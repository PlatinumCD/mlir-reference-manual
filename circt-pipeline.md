# CIRCT Pipeline Dialect

## Beginner Summary

The CIRCT `pipeline` dialect represents hardware pipelines before they have
been fully lowered to registers, wires, and modules.

A hardware pipeline splits work across stages. Each stage does some work in one
cycle, then passes values to the next stage. In final RTL, those stage
boundaries are registers. In compiler IR, however, it is useful to describe the
pipeline at several levels: first as an unscheduled dataflow graph, then as
operations assigned to stages, then as explicit stage registers, and finally as
ordinary `hw`, `seq`, and `comb` operations.

The `pipeline` dialect is CIRCT's abstraction for that middle space. It lets
the compiler talk about pipeline stages directly, without immediately spelling
out every register and control signal.

The main container operations are `pipeline.unscheduled` and
`pipeline.scheduled`. An unscheduled pipeline is a graph of work that still
needs to be assigned to stages. A scheduled pipeline has blocks that represent
stages. Inside a scheduled pipeline, `pipeline.stage` connects one stage to the
next, `pipeline.src` marks values that cross stage boundaries before registers
are materialized, and `pipeline.return` produces the pipeline outputs.

## Why This Dialect Exists

Pipelines are everywhere in hardware design. They are used to improve
throughput by allowing multiple inputs to be processed at the same time, each in
a different stage. Without a pipeline-specific IR, a compiler has two awkward
choices:

- represent the design as ordinary combinational logic and lose the stage
  structure;
- lower directly to registers and make later retiming or scheduling much
  harder.

The `pipeline` dialect exists to keep stage structure explicit while still
allowing transformations to reshape it.

The dialect separates three important ideas:

- what computation belongs to the pipeline;
- which stage each computation belongs to;
- which cross-stage values become registers, pass-through values, or control
  signals.

This matters because scheduling and register materialization are different
problems. A scheduler may decide that an add belongs in stage 1 and a multiply
belongs in stage 3. A later pass then decides how values flow from stage to
stage and where registers must be built.

## When It Matters

You will usually see `pipeline` in hardware generation and HLS-like flows, not
in a software frontend.

It matters when a compiler needs to:

- turn a dataflow graph into a multi-stage hardware pipeline;
- preserve pipeline stages while still optimizing the operations inside them;
- delay explicit register construction until after scheduling decisions;
- represent multi-cycle operations inside a pipeline;
- model stallable and non-stallable stages;
- lower a staged pipeline into HW, Seq, and Comb dialect IR.

A typical flow looks like this:

```text
dataflow or hardware IR
  -> pipeline.unscheduled
  -> pipeline-schedule-linear
  -> pipeline.scheduled with pipeline.stage and pipeline.src
  -> pipeline-explicit-regs
  -> pipeline.scheduled with explicit stage arguments and register lists
  -> lower-pipeline-to-hw
  -> hw + seq + comb
```

The dialect is especially useful when you want the compiler to reason about the
pipeline as a pipeline, rather than as a pile of already-lowered registers.

## When To Use It

Use `pipeline` when building or debugging CIRCT passes that create staged
hardware.

Use it for:

- compiler-generated pipelines;
- HLS scheduling experiments;
- explicitly staged datapaths;
- checking how values move across stage boundaries;
- modeling a pipeline before lowering it to RTL;
- testing stall, reset, go, and done control behavior;
- exploring retiming or register-materialization algorithms.

Do not use it as a general-purpose hardware module dialect. `hw`, `comb`,
`seq`, `sv`, and related CIRCT dialects are the right layers for general
structural hardware.

Do not use it as the final representation that a Verilog backend should emit
directly. The normal path is to lower `pipeline` to HW/Seq/Comb and then lower
or export those dialects.

## Core Concepts

### Pipeline Phases

The dialect is easiest to understand as a sequence of phases.

In the unscheduled phase, `pipeline.unscheduled` contains a single graph-like
body. The operations describe the work to perform, but not yet the stage where
each operation will run.

In the scheduled phase, `pipeline.scheduled` contains multiple blocks. Each
block is a pipeline stage. A `pipeline.stage` terminator says which block is
next. A `pipeline.return` terminator ends the last stage and produces data
outputs plus the pipeline `done` signal.

In the register-materialized phase, the pipeline is still represented by
`pipeline.scheduled`, but cross-stage values have been made explicit. Stage
blocks receive block arguments, and `pipeline.stage` lists values in `regs` and
`pass` groups. Values in `regs` become stage registers; values in `pass` move
through a stage without being registered.

After that, `lower-pipeline-to-hw` lowers the pipeline abstraction into
ordinary hardware operations.

### Control Signals

Pipeline container operations carry data inputs and outputs, but they also
carry hardware control.

The `clock` is required. The `go` signal starts or enables a new item entering
the pipeline. The `done` result reports when an item reaches the end. `reset`
is optional. `stall` is optional and is used to stop the pipeline or selected
stages from advancing.

The source TableGen notes that the current scheduled pipeline operation is
designed for initiation interval 1. That means the intended steady-state model
is that the pipeline can accept a new input every cycle. Pipelines with larger
initiation intervals need extra user-managed readiness state.

### Stages And SSA Values

Before explicit registers are materialized, a value defined in an earlier stage
may be referenced from a later stage through `pipeline.src`.

That does not mean the value is a normal free wire. It means "this value crosses
one or more stage boundaries, and a later pass must decide how to route it."
The `pipeline.src` operation also acts as a barrier against canonicalizations
that would incorrectly move or merge operations across stage boundaries.

After `pipeline-explicit-regs`, those implicit crossings become explicit block
arguments and `pipeline.stage` operands. At that point, the invariant is simple:
any value used inside a stage must be defined inside that stage, either by an
operation in the stage or by a block argument.

### Latency

`pipeline.latency` wraps multi-cycle work. It tells the register materializer
that the value is not available immediately in the next stage. Until the
declared latency has elapsed, the value must be passed through stages rather
than registered as an available result.

The terminator for that region is `pipeline.latency.return`.

### Stallability

`pipeline.scheduled` can carry a `stallability` attribute when it has a stall
signal. Each bit corresponds to a stage. A set bit means the stage is stallable;
an unset bit marks a non-stallable stage.

Non-stallable stages are useful for resources that cannot immediately stop once
started. This has a real design implication: downstream logic must be able to
absorb the values that continue to flow out after a stall is asserted.

## Types And Attributes

The Pipeline dialect does not define a large custom type system. It reuses
ordinary MLIR and CIRCT hardware types for data values. The clock operand uses
the Seq dialect clock type, and control signals such as `go`, `done`, `reset`,
and `stall` are `i1`.

Important attributes include:

| Attribute | Where it appears | Meaning |
| --- | --- | --- |
| `name` | `pipeline.unscheduled`, `pipeline.scheduled` | Optional human-readable pipeline name used for generated names during lowering. |
| `inputNames` | Pipeline container ops | Names for pipeline inputs. |
| `outputNames` | Pipeline container ops | Names for pipeline outputs. |
| `stallability` | `pipeline.scheduled` | Optional per-stage bits describing which stages can stall. |
| `registerNames` | `pipeline.stage` | Optional names for registers emitted at a stage boundary. |
| `passthroughNames` | `pipeline.stage` | Optional names for pass-through stage values. |
| `operator_lib` | `pipeline.unscheduled` used by scheduling | Symbol reference to an SSP operator library used by `pipeline-schedule-linear`. |
| `ssp.operator_type` | Operations inside an unscheduled pipeline | Operator type used by the linear scheduler to look up latency. |

## Operation Inventory

The Pipeline dialect currently defines 7 operations.

| Operation | Purpose |
| --- | --- |
| `pipeline.unscheduled` | Pipeline container for a graph of operations that has not yet been assigned to stages. |
| `pipeline.scheduled` | Pipeline container whose body is divided into ordered stage blocks. |
| `pipeline.src` | References a value from an earlier stage before explicit registers have been materialized. |
| `pipeline.stage` | Terminator for a scheduled stage; identifies the next stage and lists registered or pass-through values after materialization. |
| `pipeline.return` | Terminator for `pipeline.unscheduled` and `pipeline.scheduled`; returns pipeline data results. |
| `pipeline.latency` | Region wrapper for multi-cycle work with a declared latency. |
| `pipeline.latency.return` | Terminator for the body of `pipeline.latency`. |

## Transformations And Conversions

Pipeline has two dialect-local transformation passes and one main conversion
pass.

| Pass | Kind | What it does |
| --- | --- | --- |
| `pipeline-schedule-linear` | Transformation | Converts `pipeline.unscheduled` operations into `pipeline.scheduled` operations using operation latencies from an SSP operator library. |
| `pipeline-explicit-regs` | Transformation | Makes cross-stage def-use chains explicit by adding stage block arguments and `pipeline.stage` `regs` or `pass` operands. |
| `lower-pipeline-to-hw` | Conversion | Lowers `pipeline.scheduled` operations inside HW modules into HW, Seq, and Comb operations. |

### `pipeline-schedule-linear`

This pass schedules an unscheduled pipeline. It expects the
`pipeline.unscheduled` operation to have an `operator_lib` attribute pointing to
an `ssp.operator_library`. Non-return operations inside the pipeline are
expected to carry `ssp.operator_type` attributes so the pass can find operator
latencies.

The pass builds a scheduling problem, solves it with CIRCT's scheduling
utilities, creates a `pipeline.scheduled`, creates stage blocks, inserts
`pipeline.stage` terminators, and inserts `pipeline.src` operations for values
used across stage boundaries.

### `pipeline-explicit-regs`

This pass works on scheduled pipelines. It walks the stages in order and finds
operands that cross from earlier stages into later stages. It then routes those
values through intervening stages.

For normal values, this creates register crossings. For values produced by
`pipeline.latency`, it may create pass-through crossings until the declared
latency has elapsed. Finally, it removes `pipeline.src` operations because the
stage boundary dataflow has become explicit.

### `lower-pipeline-to-hw`

This conversion pass lowers scheduled pipeline operations inside `hw.module`
operations. It emits hardware-level structure for data registers, valid/done
control, stall behavior, reset behavior, and clock enables.

The pass has two important options:

| Option | Meaning |
| --- | --- |
| `clock-gate-regs` | Clock-gate each register instead of using the default input-muxing style. |
| `enable-poweron-values` | Add power-on values to pipeline control registers. |

The conversion depends on the `hw`, `comb`, and `seq` dialects.

## How To Read Pipeline IR

A minimal scheduled pipeline has a container, an entry stage, and a return:

```mlir
%out, %done = pipeline.scheduled(%a : i32 = %arg0)
    clock(%clk) reset(%rst) go(%go) entryEn(%entry)
    -> (out : i32) {
  pipeline.return %a : i32
}
```

A multi-stage scheduled pipeline uses `pipeline.stage` to branch to the next
stage. Before explicit registers are materialized, later stages use
`pipeline.src` to refer to values from earlier stages:

```mlir
%out, %done = pipeline.scheduled(%a : i32 = %arg0, %b : i32 = %arg1)
    clock(%clk) reset(%rst) go(%go) entryEn(%entry)
    -> (out : i32) {
^bb0(%a0 : i32, %b0 : i32, %entry0 : i1):
  %sum = comb.add %a0, %b0 : i32
  pipeline.stage ^bb1

^bb1(%stage1_en : i1):
  %sum1 = pipeline.src %sum : i32
  pipeline.return %sum1 : i32
}
```

After `pipeline-explicit-regs`, the same idea is expressed through stage block
arguments and `pipeline.stage` operands:

```mlir
pipeline.stage ^bb1 regs(%sum : i32)
^bb1(%sum_s0 : i32, %stage1_en : i1):
  pipeline.return %sum_s0 : i32
```

The important mental model is that stage blocks are not arbitrary control-flow
blocks. They are ordered hardware pipeline stages.

## What It Implies

Using `pipeline` means the design has explicit cycle structure. Operations in
different stages are separated by time, not just by textual order. A value that
crosses stages may require one or more registers. A `pipeline.latency` result
may not be usable until enough stages have passed.

It also means optimization must be stage-aware. A normal canonicalization that
moves a producer across a block boundary could change the hardware timing.
`pipeline.src`, stage verifiers, and the explicit-register pass exist to keep
that timing discipline visible.

Stall behavior is also part of the semantics. A pipeline with no stall signal is
continuous. A pipeline with a stall signal needs lowered control logic. A
pipeline with non-stallable stages can continue producing values after a stall,
so downstream buffering becomes part of the design contract.

## Beginner Checklist

When reading a Pipeline dialect example, ask:

- Is the container `pipeline.unscheduled` or `pipeline.scheduled`?
- If it is unscheduled, what operator library and latencies will schedule it?
- How many stage blocks are present?
- Which values cross stage boundaries through `pipeline.src`?
- Has `pipeline-explicit-regs` already run?
- Are `pipeline.stage` operands in `regs` or `pass` groups?
- Does the pipeline have `stall`, `reset`, and `go` signals?
- Does `stallability` mark any non-stallable stages?
- Are there `pipeline.latency` operations that delay value availability?
- Has the design been lowered with `lower-pipeline-to-hw` yet?

For a beginner, the central idea is this: `pipeline` is not just syntax for
registers. It is a staged hardware abstraction that lets CIRCT schedule,
materialize, and lower pipelines in separate, understandable steps.

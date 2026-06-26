# CIRCT `llhd` Dialect

The CIRCT `llhd` dialect is a low-level hardware description dialect for event-based simulation and for lowering procedural hardware descriptions toward structural hardware.

For a beginner, the useful mental model is this: LLHD models hardware the way Verilog and VHDL simulation models hardware. Signals change over time. Processes wake up when observed values change or when time advances. Drives schedule future signal updates. Later passes can recognize which processes are really combinational logic, which ones are registers, and which signals can be promoted into ordinary SSA values.

The local checkout defines 24 `llhd` operations, two custom type definitions, one custom time attribute, and 10 LLHD passes.

## When LLHD Is Important

`llhd` is important when CIRCT needs to represent procedural, simulator-like hardware behavior before it can be lowered to cleaner structural hardware. This is especially relevant for SystemVerilog-style processes, VHDL-style time and signal behavior, testbench-like event scheduling, and transformations that turn edge-sensitive procedural descriptions into registers or combinational logic.

Use this dialect when you need to answer questions like:

- Which values are signals rather than ordinary SSA values?
- Which process wakes up when a signal changes?
- Is a drive immediate, delta-delayed, epsilon-delayed, or physically delayed?
- Has procedural control flow been wrapped in `llhd.combinational` yet?
- Has a process been recognized as combinational logic?
- Has a sequential process been converted into `seq` registers?
- Have signal probes and drives been promoted into normal values?

You usually inspect LLHD after parsing or lowering procedural HDL-like constructs, and before final hardware lowering removes event-queue concepts.

## Why It Is Needed

Hardware description languages often describe behavior procedurally. A Verilog `always` block or a VHDL process is not just a collection of wires. It has a sensitivity list, can suspend, can assign signals with delays, and can observe simulation time. The compiler needs a representation for this before it can decide whether the behavior is combinational, sequential, or simulation-only.

LLHD provides that layer. It has a time type, signal references, probe and drive operations, process operations, wait terminators, and coroutine support. This lets CIRCT preserve event-queue semantics while still using MLIR regions, SSA values, symbols, and passes.

The implication is that LLHD is a staging dialect. Seeing LLHD usually means the compiler has not fully committed to structural registers, wires, muxes, or pure combinational expressions yet.

## Types And Attributes

LLHD defines `!llhd.time`, a simulation time type. A time value combines a real-time quantity, a delta step, and an epsilon slot. The real-time part represents physical time such as nanoseconds. The delta step models infinitesimal simulation steps. The epsilon value models an absolute time slot within a delta step, which is useful for SystemVerilog scheduling regions.

LLHD also defines `!llhd.ref<T>`, a reference to a signal or variable carrying a nested type `T`. A signal operation returns a reference. Probe and drive operations use that reference to read from or write to the signal.

The custom time attribute is `#llhd.time<...>`. For example, `#llhd.time<1ns, 0d, 0e>` represents a one-nanosecond time value with zero delta and zero epsilon. `llhd.constant_time` materializes such an attribute as an SSA value.

Most payload data uses regular MLIR or CIRCT hardware types. Signals can carry integer types, arrays, structs, unions, and other hardware value types through `!llhd.ref<...>`.

## Operation Inventory

This checkout defines 24 `llhd` operations.

### Time Operations

```text
llhd.constant_time
llhd.current_time
llhd.time_to_int
llhd.int_to_time
llhd.delay
```

`llhd.constant_time` creates an SSA time value from a `#llhd.time` attribute. `llhd.current_time` reads the current simulation time and has a memory-read side effect so it is not incorrectly moved across waits. `llhd.time_to_int` converts time to an integer number of femtoseconds, and `llhd.int_to_time` performs the inverse conversion. `llhd.delay` propagates value changes after a specified time delay.

These operations are the part of LLHD that has no direct equivalent in ordinary combinational hardware. They preserve event-queue timing until the compiler can remove or lower it.

### Signal Creation, Probing, Driving, And Projection

```text
llhd.sig
llhd.prb
llhd.drv
llhd.output
llhd.global_signal
llhd.get_global_signal
llhd.sig.extract
llhd.sig.array_slice
llhd.sig.array_get
llhd.sig.struct_extract
```

`llhd.sig` creates a signal initialized with a value and returns a `!llhd.ref<T>`. `llhd.prb` probes a signal and returns its current value. `llhd.drv` drives a new value onto a signal after a time value, optionally guarded by an enable. `llhd.output` is a shorthand that creates a signal and continuously drives it from a value after a delay.

`llhd.global_signal` declares a symbol-visible global signal. `llhd.get_global_signal` gets a reference to such a global signal.

The projection operations create references to parts of signals. `llhd.sig.extract` extracts a bit range from an integer signal. `llhd.sig.array_slice` gets a consecutive subrange from an array signal. `llhd.sig.array_get` gets one array element. `llhd.sig.struct_extract` gets a named struct field.

For a beginner, the important distinction is that LLHD separates the signal reference from the value carried by the signal. `llhd.prb` reads the value. `llhd.drv` schedules a write.

### Processes And Terminators

```text
llhd.process
llhd.final
llhd.combinational
llhd.wait
llhd.halt
llhd.yield
```

`llhd.process` is a concurrent process that runs during simulation. It can suspend with `llhd.wait`, resume when observed values change, and optionally produce result values that are held while the process is suspended.

`llhd.final` is a process that runs at the end of simulation. It cannot use `llhd.wait`, because there is no later time slot to resume into. It must eventually terminate with `llhd.halt`.

`llhd.combinational` is a procedural region that executes at the beginning of simulation and again whenever values used in the body change. It terminates with `llhd.yield`, which returns values from the combinational region.

`llhd.wait` suspends a process or coroutine until observed values change or a delay passes, then resumes at a successor block. `llhd.halt` suspends a process or coroutine forever. `llhd.yield` returns values from a combinational process or a global signal initializer.

### Coroutines

```text
llhd.coroutine
llhd.call_coroutine
llhd.return
```

`llhd.coroutine` defines a suspendable subroutine. It is the lowered form of a SystemVerilog task-like construct. A coroutine can wait and halt like a process.

`llhd.call_coroutine` calls a coroutine from a process or another coroutine. The caller suspends until the callee returns.

`llhd.return` returns from a coroutine to its caller.

### Complete Op List

For reference, the exact LLHD operation names in this checkout are:

```text
llhd.call_coroutine, llhd.combinational, llhd.constant_time, llhd.coroutine, llhd.current_time, llhd.delay, llhd.drv, llhd.final, llhd.get_global_signal, llhd.global_signal, llhd.halt, llhd.int_to_time, llhd.output, llhd.prb, llhd.process, llhd.return, llhd.sig, llhd.sig.array_get, llhd.sig.array_slice, llhd.sig.extract, llhd.sig.struct_extract, llhd.time_to_int, llhd.wait, llhd.yield
```

## Transformations And Conversions

The local checkout defines 10 LLHD passes. They do not form one mandatory pipeline, but they fit together as a lowering story from procedural/event-queue IR to structural hardware IR.

`llhd-wrap-procedural-ops` wraps procedural operations such as `func.call` and `scf` operations directly inside an `hw.module` body into `llhd.combinational` regions. This gives those operations an SSACFG region where inlining and control-flow conversion can work.

`llhd-inline-calls` inlines `func.call` operations nested inside `llhd.combinational`, `llhd.process`, and `llhd.final` operations in hardware modules. It expects procedural wrapping to have happened first, because graph regions cannot directly absorb arbitrary inlined control flow.

`llhd-unroll-loops` unrolls control-flow loops inside `llhd.combinational` regions when their bounds are statically known. It replicates loop bodies and replaces induction variables with constants.

`llhd-remove-control-flow` removes acyclic control flow inside `llhd.combinational` regions. It merges blocks into the entry block and replaces block arguments with muxes. This requires loops to be gone and operations to be side-effect free, because conditionally executed blocks are moved into unconditional position.

`llhd-lower-processes` converts process operations to combinational operations where possible. It recognizes processes whose wait/control-flow shape describes combinational behavior, ensures all outside operands are observed, replaces the process with `llhd.combinational`, and replaces the wait with `llhd.yield`.

`llhd-deseq` converts sequential processes to registers. It analyzes past and present values, control conditions, and process structure to build `seq` registers and related logic. The LLHD docs note an important reset implication: the pass follows the common synthesis-tool behavior of mapping edge-sensitive reset descriptions to level-sensitive register resets without requiring the reset value to be proven constant.

`llhd-sig2reg` promotes eligible LLHD signals to SSA values. It traces signal aliases, probes, drives, and projected intervals. If drives do not overlap in unsupported ways and delays are known, the pass replaces signal behavior with value-level logic.

`llhd-mem2reg` promotes memory and signal slots into SSA values using reaching definitions and block arguments. It understands blocking-like drives, delta-delayed drives, and drive conditions.

`llhd-hoist-signals` hoists probes and promotes drives to process results when doing so is semantically safe. It avoids moving probes that would change sampling behavior across `llhd.wait`.

`llhd-combine-drives` combines scalar or field-level drives into aggregate drives. If drives cover all fields or elements of an aggregate signal, the pass can build a single aggregate value and drive the whole signal.

The exact pass names covered in this chapter are:

```text
llhd-combine-drives, llhd-deseq, llhd-hoist-signals, llhd-inline-calls, llhd-lower-processes, llhd-mem2reg, llhd-remove-control-flow, llhd-sig2reg, llhd-unroll-loops, llhd-wrap-procedural-ops
```

## What It Implies

Seeing `llhd.sig`, `llhd.prb`, and `llhd.drv` means the IR still has signal-reference semantics. A value is not simply computed and assigned; it may be probed from a signal and driven after a simulation time.

Seeing `llhd.process` and `llhd.wait` means event-queue behavior is still explicit. The process may suspend and resume later based on observed values or time.

Seeing `llhd.combinational` means some procedural logic has been isolated into an SSACFG region and may be ready for inlining, loop unrolling, control-flow removal, or conversion to ordinary combinational hardware.

Seeing `llhd.coroutine` means task-like behavior has not yet been inlined or otherwise lowered away.

Seeing `llhd-deseq` in a pipeline means the compiler is trying to recognize sequential processes and emit concrete registers. Seeing `llhd-sig2reg`, `llhd-mem2reg`, or `llhd-hoist-signals` means the flow is trying to remove signal references and promote behavior into SSA or process results.

## How To Read LLHD IR

Start with signals. Find `llhd.sig` and `llhd.global_signal` operations, then trace probes and drives. A probe reads the signal's current value. A drive schedules a future value for the signal.

Next, read time values. A `#llhd.time<0ns, 0d, 0e>` style value is not just a number. It carries physical time, delta, and epsilon scheduling information.

Then inspect processes. A simple combinational-style process often waits on the same values it reads. A sequential process often compares current and past signal values or responds to edge-like conditions. A final process is simulation cleanup or end-of-run checking.

Finally, check which passes have already run. If control-flow operations remain inside `llhd.combinational`, `llhd-unroll-loops` or `llhd-remove-control-flow` may still be needed. If `llhd.process` remains, `llhd-lower-processes` or `llhd-deseq` may not have recognized it yet.

## Minimal Example

This simplified fragment creates a signal, probes it, and drives a new value after zero time:

```mlir
%false = hw.constant false
%true = hw.constant true
%time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
%sig = llhd.sig name "flag" %false : i1
%old = llhd.prb %sig : i1
llhd.drv %sig, %true after %time : i1
```

Read this as "make a signal initialized to false, read its current value, and schedule true to be driven onto it." In lower structural IR, the compiler would prefer ordinary values, registers, wires, and muxes. LLHD keeps the event-queue meaning available until passes can prove how to lower it.

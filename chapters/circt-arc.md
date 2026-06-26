# CIRCT `arc` Dialect

## Beginner Summary

The CIRCT `arc` dialect is a simulation-oriented hardware dialect for
representing state transfer in a circuit. Its core abstraction is an "arc": a
function-like piece of logic that computes the next value of state from current
inputs and current state. Around that, the dialect models simulation state,
memories, clocks, coroutine-style processes, tracing taps, runtime metadata,
and lowering paths to functions and LLVM.

For a beginner, the useful mental model is:

```text
hw / llhd / seq / comb design
  -> convert-to-arcs
  -> arc.define + arc.state + arc.model
  -> Arc optimization and state lowering
  -> explicit state storage, runtime metadata, funcs
  -> lower-arc-to-llvm
```

Arc is not a high-level hardware design dialect for authoring circuits. It is a
canonical, compiler-friendly representation used by CIRCT's arcilator flow to
turn hardware into executable simulation code.

## Why This Dialect Exists

Hardware simulation needs a different shape than source RTL. Source-oriented IR
has modules, instances, wires, registers, memories, clocks, procedural regions,
and hierarchy. A fast software simulator wants explicit update functions,
explicit state storage, predictable evaluation order, and a way to connect the
generated code to a runtime.

The `arc` dialect exists to bridge that gap. It outlines logic between state
elements into reusable arc definitions, represents registers and memories as
state-transfer operations, and then progressively lowers that model into
functions, storage layouts, runtime descriptors, and LLVM-compatible code.

The dialect also gives CIRCT places to optimize simulation-specific structure:
deduplicating equivalent arcs, inlining tiny arcs, splitting loops, inferring
enables and resets, turning suitable logic into lookup tables, vectorizing
repeated scalar operations, and preserving observability with taps.

## When It Matters

You will see `arc` in CIRCT's arcilator pipeline and in tests that exercise
simulation lowering. It matters when you are debugging:

- how registers became state-transfer functions;
- why a model's state storage has a particular layout;
- how memory read and write ports are scheduled;
- how clock trees, initial blocks, final blocks, and coroutine processes lower;
- how signal tracing metadata is preserved;
- why a simulator output contains a generated runtime model descriptor;
- where large logic cones were deduplicated, inlined, vectorized, or converted
  to lookup tables.

If `hw`, `seq`, and `comb` are still close to structural hardware, `arc` is the
point where the compiler starts thinking like a simulator generator.

## When To Use It

Most users should not write Arc IR by hand. Use it when working on CIRCT's
simulation compiler, debugging the arcilator pipeline, adding support for a new
simulation feature, or investigating why generated simulation code behaves a
certain way.

Use Arc concepts when you need to answer questions like:

- What state exists in this model?
- Which arc computes the next value for a state element?
- What latency is associated with a state transfer?
- Which values are trace taps?
- How are arrays, memories, or coroutine processes represented before LLVM
  lowering?
- Which pass is responsible for converting this construct into functions or
  runtime calls?

For ordinary hardware transformations, prefer the higher-level CIRCT dialects.
For final executable simulation lowering, Arc is the dialect to inspect.

## Core Types And Attributes

`!arc.state<T>` is a reference-like handle to a stored value of type `T`. It is
read with `arc.state_read` and updated with `arc.state_write`.

`!arc.memory<N x T, A>` represents a memory with `N` words of word type `T` and
address type `A`. The dialect has high-level memory ports and lower-level
memory read/write operations.

`!arc.storage` and `!arc.storage<N>` represent simulation storage regions.
Storage operations allocate or address subregions that later hold state,
memories, arrays, and runtime model data.

`!arc.sim.instance<@model>` is a simulation instance handle used by
`arc.sim.*` operations.

`!arc.coroutine_pc<@coro>` and `!arc.coroutine_state<@coro>` are opaque
coroutine program counter and persisted-state types. `arc-lower-coroutines`
replaces them with concrete state and PC representations.

`!arc.arrayref<N x T>` is a reference-like array type used when arrays have been
bufferized.

`#arc.trace_tap` records trace metadata: signal type, state offset, and one or
more names. The dialect also defines enum attributes for `arc.zero_count` and
the `arc-lower-vectorizations` mode option.

## Complete Operation Inventory

The current `arc` dialect defines 57 concrete operations:

```text
arc.alloc_memory
arc.alloc_state
arc.alloc_storage
arc.arrayref.alloc
arc.arrayref.copy
arc.arrayref.create
arc.arrayref.from_array
arc.arrayref.get
arc.arrayref.inject
arc.arrayref.slice
arc.arrayref.to_array
arc.call
arc.coroutine.call
arc.coroutine.define
arc.coroutine.halt
arc.coroutine.instance
arc.coroutine.pc_is_halt
arc.coroutine.pc_is_return
arc.coroutine.return
arc.coroutine.start_pc
arc.coroutine.undefined_state
arc.coroutine.yield
arc.current_time
arc.define
arc.execute
arc.final
arc.get_next_wakeup
arc.initial
arc.lut
arc.memory
arc.memory_read
arc.memory_read_port
arc.memory_write
arc.memory_write_port
arc.model
arc.output
arc.root_input
arc.root_output
arc.runtime.model
arc.set_next_wakeup
arc.sim.emit
arc.sim.get_next_wakeup
arc.sim.get_port
arc.sim.get_time
arc.sim.instantiate
arc.sim.set_input
arc.sim.set_time
arc.sim.step
arc.state
arc.state_read
arc.state_write
arc.storage.get
arc.tap
arc.terminate
arc.vectorize
arc.vectorize.return
arc.zero_count
```

## How To Read The Operation Groups

Arc definitions and calls are built around `arc.define`, `arc.call`,
`arc.output`, and `arc.state`. `arc.define` is a function-like state-transfer
arc. `arc.state` instantiates a state element whose next value is computed by an
arc. `arc.call` invokes an arc without creating state, and `arc.output`
terminates an arc or other Arc region.

Model and storage operations describe the simulator object. `arc.model`
contains the stratified model body. `arc.alloc_state`, `arc.alloc_memory`,
`arc.alloc_storage`, `arc.root_input`, `arc.root_output`, and `arc.storage.get`
allocate or address pieces of model storage. `arc.state_read`,
`arc.state_write`, `arc.current_time`, `arc.get_next_wakeup`,
`arc.set_next_wakeup`, and `arc.terminate` interact with stored simulation
state.

Memory operations come in two levels. `arc.memory`, `arc.memory_read_port`, and
`arc.memory_write_port` represent higher-level memory structures and ports.
`arc.memory_read` and `arc.memory_write` are lower-level explicit memory
accesses produced later in state lowering.

Clock-tree and lifecycle regions use `arc.initial` and `arc.final` for code that
runs at the start and end of simulation. `arc.execute` embeds an SSACFG region
inside Arc's graph-style regions.

Coroutine operations model suspendable processes. The `arc.coroutine.*` family
defines coroutines, calls them, yields, returns, halts, checks PC sentinel
values, creates initial PC/state values, and runs a coroutine continuously in an
`hw.module`.

Simulation orchestration operations are `arc.sim.*`. They instantiate models,
set inputs, read ports, step simulation, emit values to a driver, get or set
time, and query the next wakeup.

Optimization helper operations include `arc.lut`, `arc.zero_count`,
`arc.vectorize`, `arc.vectorize.return`, and `arc.tap`. Arrayref operations
represent bufferized arrays with reference semantics.

## Transformations And Conversions

The Arc dialect itself registers these 30 passes:

```text
arc-add-taps
arc-allocate-state
arc-canonicalizer
arc-dedup
arc-find-initial-vectors
arc-generate-driver
arc-infer-memories
arc-infer-state-properties
arc-inline
arc-insert-runtime
arc-latency-retiming
arc-lower-arcs-to-funcs
arc-lower-arrays
arc-lower-clocks-to-funcs
arc-lower-coroutines
arc-lower-lut
arc-lower-state
arc-lower-vectorizations
arc-lower-verif-simulations
arc-make-tables
arc-merge-ifs
arc-merge-taps
arc-mux-to-control-flow
arc-print-cost-model
arc-remove-i0-types
arc-resolve-xmr
arc-simplify-variadic-ops
arc-split-funcs
arc-split-loops
arc-strip-sv
```

The entry conversion is `convert-to-arcs`, declared in CIRCT's conversion pass
table rather than the Arc pass table. It outlines logic between registers into
`arc.define` state-transfer arcs and replaces the original register behavior
with arc invocations and latency. It rewrites selected LLHD combinational
operations and then outlines operations into arcs. Its `tap-registers` option
controls whether registers become observable.

The exit conversion is `lower-arc-to-llvm`. It lowers the state-transfer Arc
representation to LLVM-compatible IR using `cf`, `scf`, `func`, and LLVM
dialect support.

The arcilator pipeline ties these pieces together. It runs simulation and
memory preparation, converts to arcs, optionally deduplicates, flattens modules,
canonicalizes, optimizes arcs, lowers state, allocates state storage, lowers
arcs and clocks to functions, inserts runtime support, optionally lowers arrays,
and finally lowers Arc to LLVM.

## What The Major Passes Do

State and runtime passes are the backbone of the dialect. `arc-lower-state`
turns state operations into explicit reads and writes grouped by clock tree.
`arc-allocate-state` computes storage layout for model state. `arc-insert-runtime`
adds structures and calls needed by the Arc runtime library. `arc-generate-driver`
emits a driver function that instantiates a model and repeatedly advances
simulation time until no wakeup remains.

Optimization passes clean up the state-transfer representation. `arc-dedup`
merges equivalent arcs, even when they differ only by constants that can be
outlined. `arc-inline` inlines tiny arcs. `arc-split-loops` breaks zero-latency
feedback loops through arcs. `arc-latency-retiming` pushes latency through the
design. `arc-canonicalizer`, `arc-merge-ifs`, `arc-simplify-variadic-ops`,
`arc-mux-to-control-flow`, `arc-strip-sv`, `arc-remove-i0-types`, and
`arc-resolve-xmr` prepare the IR for cleaner simulation lowering.

Specialized lowering passes handle nontrivial constructs. `arc-lower-coroutines`
turns coroutine definitions into state-machine functions with explicit state and
program counters. `arc-lower-arrays` uses bufferized arrayrefs. `arc-lower-lut`
lowers lookup tables into `hw` and `comb`. `arc-lower-vectorizations` removes
`arc.vectorize` boundaries in stages or fully, using either packed integers or
SIMD vector types. `arc-lower-arcs-to-funcs` and `arc-lower-clocks-to-funcs`
convert remaining Arc callable structure into functions.

Observability and analysis passes include `arc-add-taps`, `arc-merge-taps`,
`arc-infer-memories`, `arc-infer-state-properties`, `arc-find-initial-vectors`,
`arc-make-tables`, and `arc-print-cost-model`.

## What The Dialect Implies

Seeing Arc means the compiler is no longer just preserving source hardware
structure. It is preparing a simulator. State is becoming explicit storage,
clocked updates are becoming scheduled state writes, and hierarchy is being
reshaped into model and function boundaries that a runtime can execute.

It also implies that observability is an active concern. Taps, trace tap
metadata, root inputs, root outputs, and runtime model metadata exist because a
simulator must expose values to users and trace files, not merely compute final
outputs.

The practical implication is that Arc IR should be read in terms of execution:
what state exists, what reads happen before writes, which arc computes which
next value, how time advances, and how the generated runtime will call into the
model.

## How To Use It In Practice

When reading an Arc file, start at `arc.model` and list its storage, root inputs,
root outputs, initial regions, final regions, and clock-like regions. Then find
the `arc.define` operations and see which `arc.state`, `arc.call`, or memory
port operations use them.

For correctness bugs, inspect `arc.state_read`, `arc.state_write`, memory
ports, enables, resets, wakeups, and coroutine PCs. For performance bugs, look
for too many small arcs, missed deduplication, large unsplit functions, missed
vectorization, or expensive mux/control-flow shapes.

The shortest beginner rule is: Arc is the circuit after it has become a
simulation problem. Read it as a state machine plus runtime interface, not as
source RTL.

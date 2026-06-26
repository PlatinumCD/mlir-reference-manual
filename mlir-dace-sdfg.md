# MLIR-DaCe sdfg Dialect

The MLIR-DaCe `sdfg` dialect represents Stateful DataFlow Graphs inside MLIR. An SDFG is a data-centric program representation: instead of making control flow the main structure and treating memory traffic as a consequence of instructions, it makes data movement, arrays, streams, states, and state transitions explicit.

This dialect matters because MLIR and DaCe approach optimization from different angles. Ordinary MLIR pipelines often begin with control-centric IR: functions, loops, branches, calls, loads, and stores. DaCe uses SDFGs to expose how data moves through a program and where parallelism, locality, and communication can be optimized. MLIR-DaCe is a bridge between those worlds. It lets a compiler move from MLIR dialects such as `func`, `scf`, `memref`, `arith`, `math`, LLVM, and `linalg` toward an SDFG form, or lower SDFG structure back to generic MLIR dialects.

For a beginner, the key idea is that `sdfg` is not just another loop dialect. It is a graph dialect. A top-level SDFG contains states. States contain dataflow nodes such as tasklets, maps, loads, stores, copies, streams, and library calls. Edges between states carry conditions and symbol assignments. This makes the dialect good at describing programs where the important question is "what data exists, where does it move, and under what state transition?" rather than only "which instruction executes next?"

## When To Use It

Use `sdfg` when a compiler pipeline wants to expose data-centric structure to DaCe or to apply transformations that depend on explicit data movement. It is useful for array-heavy numerical programs, stencil-like computations, linear algebra kernels, streaming computations, and programs where symbolic sizes determine the shape of data movement.

You normally do not start by hand-writing large `sdfg` modules. A frontend or conversion pass builds the dialect from more familiar MLIR. You inspect it when debugging the boundary between MLIR and DaCe, when studying whether loops became SDFG maps, when checking how loads and stores became memlet-like movement, or when verifying that state transitions and symbolic sizes were preserved.

The dialect is needed because `scf.for`, `memref.load`, and `memref.store` do not directly say enough about dataflow intent. They say that a loop iterates and memory is accessed. An SDFG map says that a parametric parallel scope exists. An SDFG state says that a subgraph of data movement happens as one control state. An SDFG edge says that execution can move from one state to another under a condition while assigning symbols. Those are stronger compiler facts.

## Operation Inventory

The `sdfg` dialect defines these operations:

```text
sdfg.alloc
sdfg.alloc_symbol
sdfg.consume
sdfg.copy
sdfg.edge
sdfg.libcall
sdfg.load
sdfg.map
sdfg.nested_sdfg
sdfg.return
sdfg.sdfg
sdfg.state
sdfg.store
sdfg.stream_length
sdfg.stream_pop
sdfg.stream_push
sdfg.subview
sdfg.sym
sdfg.tasklet
sdfg.view_cast
```

The dialect also defines two main public types: `!sdfg.array<...>` and `!sdfg.stream<...>`. Both are built from an internal sized type that carries an element type, symbolic dimensions, integer dimensions, and shape information. This is how the dialect can represent arrays and streams whose sizes may be partly symbolic.

## Graph Structure

`sdfg.sdfg` is the top-level SDFG operation. It is a module-level symbol table operation with one body region. Its attributes include an integer ID, an optional entry state symbol, and the number of arguments. Its region contains states, interstate edges, top-level allocations, and top-level symbols. If the `entry` attribute is present, it names the first state; otherwise, the first state in the body is treated as the start state by the translator.

`sdfg.nested_sdfg` is the scoped version of the same idea. It can appear inside a state, map, or consume scope. Use it when a part of a larger dataflow graph should itself contain states and edges. Nested SDFGs are important because real programs often contain structured computations that should be optimized as subgraphs without flattening every state into one global graph.

`sdfg.state` represents a state region inside an SDFG. A state is a symbol, has an ID, and owns a single block of dataflow operations. Within one state, the compiler sees data movement and computation nodes. Between states, the compiler sees explicit transitions through `sdfg.edge`.

`sdfg.edge` represents an interstate transition. It names source and destination states by symbol reference, has an `assign` attribute for symbol assignments, has a `condition` attribute with default true behavior, and may carry an optional reference value. Edges are where control-like behavior enters the SDFG model. A loop in source code may become states connected by edges that update a loop symbol and test a condition.

## Data Containers And Memlet-Like Movement

`sdfg.alloc` creates an SDFG array or stream value. It can appear at top level, in nested SDFGs, or inside a state. Its optional name records the container name, and its `transient` unit attribute marks temporary storage. The distinction between persistent arguments and transient arrays matters because DaCe optimizations often reason about which data is external and which data can be introduced, removed, or relocated by the compiler.

`sdfg.load` reads an element from an `!sdfg.array` at index operands and returns the array element type. `sdfg.store` writes a value to an array at index operands. Both are constrained so the loaded or stored scalar type matches the array element type. `sdfg.copy` represents a copy from one SDFG array to another. Beginners can read these as the basic data movement operations: load a scalar, store a scalar, or copy a container.

`sdfg.subview` returns a view of an array using offsets, sizes, and strides. `sdfg.view_cast` changes the array view type without describing a new allocation. These are useful when a program needs to express that the same underlying data is being seen through a different shape or slice. In a data-centric IR, this is not a minor detail: views and subviews affect dependence analysis, data movement, and storage layout.

## Computation Scopes

`sdfg.tasklet` represents a pure computation region with operands, results, and a body. The body usually contains ordinary MLIR computation, followed by `sdfg.return`. Tasklets are the local compute units of an SDFG. They are intentionally separated from array movement: values are loaded or connected into the tasklet, the tasklet computes, and results are stored or connected outward.

`sdfg.return` terminates a tasklet and returns its computed values. Its operands must match the tasklet result expectations. When reading SDFG IR, tasklets are often where familiar arithmetic or scalar logic still appears.

`sdfg.map` represents a parametric loop-like scope. It has entry and exit IDs, lower bounds, upper bounds, and steps. It implements MLIR's loop-like interface. A map is one of the most important SDFG concepts: it describes a structured repeated computation that can often be parallelized or transformed. Where `scf.for` emphasizes sequential loop control, a map emphasizes a range of data-parallel work.

`sdfg.consume` represents a stream-processing scope. It has entry and exit IDs, an optional number of processing elements, an optional condition, and a stream type. Its region has access to a processing element value and the popped stream element. Use it for computations driven by streams rather than statically bounded array iteration.

`sdfg.libcall` represents a direct call to a DaCe library node, such as a BLAS node. It implements MLIR's call operation interface, carries a string callee, and has variadic operands and results. This lets the dialect keep high-level library intent instead of immediately lowering a library call into scalar code.

## Streams And Symbols

`sdfg.stream_push` pushes a value into an `!sdfg.stream`, and `sdfg.stream_pop` pops one value from a stream. Their type constraints ensure that the pushed or popped value matches the stream element type. `sdfg.stream_length` returns an `i32` length for a stream. These operations make streaming dataflow explicit enough for a compiler to distinguish it from random array access.

`sdfg.alloc_symbol` declares a symbolic name. `sdfg.sym` materializes a symbolic arithmetic expression as an integer or index value. Symbols are important because SDFGs frequently describe shapes, bounds, and conditions that are not compile-time constants. A symbol such as `N` can appear in array dimensions, map bounds, or interstate assignments.

## Transformations And Conversions

MLIR-DaCe registers three main conversion passes in `sdfg-opt`.

`convert-to-sdfg` converts generic MLIR into the SDFG dialect. Its TableGen summary says it converts SCF, Arith, Math, and Memref dialects to SDFG. The implementation also contains conversion patterns for `func.func`, `func.call`, `func.return`, many `memref` operations, `scf.for`, `scf.while`, `scf.condition`, `scf.if`, `scf.yield`, several LLVM pointer and memory operations, and a fallback pattern that wraps remaining operations as tasklets. It also converts memref-like and LLVM pointer-like types into SDFG array types where appropriate. The pass has a `main-func-name` option to specify which function should be treated as the main function.

`linalg-to-sdfg` is registered as a module pass for converting `linalg` to SDFG. In this checkout, the pattern population function for that pass is empty, so the pass is part of the intended interface but should not be read as a complete Linalg lowering by itself.

`lower-sdfg` converts SDFG back to more generic MLIR: `func`, `cf`, `memref`, `arith`, and `scf`. Its patterns lower top-level and nested SDFGs to functions, states to blocks, edges to branches, arrays to memrefs, loads and stores to memref operations, tasklets to functions, tasklet returns to returns, maps to parallel-style control, and symbols to ordinary arithmetic operations through the symbolic parser. The implementation contains a placeholder-style conversion for consume nodes, so stream-consume lowering is less complete than core arrays, states, and maps.

MLIR-DaCe also registers an `mlir-to-sdfg` translation in `sdfg-translate`. That translation walks a module containing exactly one top-level `sdfg.sdfg`, collects states, edges, allocations, symbols, tasklets, maps, consumes, copies, loads, stores, stream operations, library calls, and nested SDFGs, and emits SDFG JSON. This is the bridge from MLIR syntax to the DaCe SDFG representation.

## How To Read sdfg IR

Start at `sdfg.sdfg` and identify the entry state. Then read each `sdfg.state` as a dataflow subgraph. Inside the state, look for allocations, loads, stores, maps, tasklets, library calls, and stream operations. Those tell you what data is used and how computation is organized.

Next, read `sdfg.edge` operations between states. Their source, destination, condition, and assignments tell you how the graph moves through control states. This is where loop-carried symbolic updates and branch-like behavior often appear.

Finally, decide which direction the pipeline is moving. If the program is running `convert-to-sdfg` or `mlir-to-sdfg`, the goal is to expose data-centric structure for DaCe. If it is running `lower-sdfg`, the goal is to return to ordinary MLIR control, memory, and function constructs. The implication is that `sdfg` is a bridge dialect: it preserves enough MLIR structure to interoperate with MLIR passes, while expressing enough SDFG structure to make data-centric optimization possible.

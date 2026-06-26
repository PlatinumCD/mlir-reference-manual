# StableHLO `interpreter` Dialect

The StableHLO `interpreter` dialect contains operations that support the StableHLO reference interpreter. It is not part of the StableHLO spec itself. Instead, it provides interpreter-only functionality for running multi-process test programs, printing runtime tensors, and recording intermediate tensor values.

For a beginner, the most important distinction is this: StableHLO operations describe portable computation, while the `interpreter` dialect describes helper behavior for the reference interpreter. A `stablehlo.add` operation is part of the program being specified. An `interpreter.probe` operation is a tool for observing a value while the reference interpreter runs that program.

This dialect matters because StableHLO needs a readable, executable reference for its semantics. Compiler developers, framework integrators, and backend authors can use the interpreter to compare expected behavior with what another runtime or compiler backend produces. The `interpreter` dialect gives that reference implementation a small amount of IR-level control without adding debugging and test machinery to the StableHLO specification.

## Operation Inventory

The `interpreter` dialect defines these operations:

```text
interpreter.run_parallel
interpreter.print
interpreter.probe
```

It also defines one helper attribute class:

```text
Array of FlatSymbolRefArrayAttr
```

That helper attribute is used by `interpreter.run_parallel` to hold a two-dimensional grid of function symbol references.

The dialect does not define custom public types in this checkout. Its operations work with ordinary StableHLO tensor values, StableHLO tokens, tuples indirectly through surrounding StableHLO operations, and function symbol references. The runtime value model used by the interpreter is implemented in C++ classes such as `Tensor`, `Token`, `Tuple`, and `InterpreterValue`, not as separate MLIR types owned by the `interpreter` dialect.

## The Interpreter Role

The StableHLO reference interpreter evaluates MLIR functions by tracking a mapping from SSA values to runtime values. Function arguments are mapped to input tensors, operations are visited in order, operands are looked up in the runtime map, and operation-specific evaluation code computes new runtime values for the results.

The implementation is intentionally written for readability. For example, an elementwise operation can allocate a result tensor, iterate over result indices, read elements from operand tensors, compute a new element, and write it into the result tensor. That style is slower than a production tensor runtime, but speed is not the main goal. The interpreter is a reference for what StableHLO means.

The `interpreter` dialect is the narrow set of operations that this reference engine needs beyond ordinary StableHLO. It lets the interpreter model distributed process grids, debug output, and serialized probes while still keeping those features visible as MLIR operations.

## `interpreter.run_parallel`

`interpreter.run_parallel` runs a two-dimensional grid of StableHLO processes. The dimensions are replicas and partitions. Its `programs` attribute is a two-dimensional array of function symbol references, where each entry names the function to execute for one process in the grid.

```mlir
%results:2 = "interpreter.run_parallel"() {
  programs = [[@foo], [@bar]]
} : () -> (tensor<ui32>, tensor<ui32>)
```

The operation has variadic inputs and variadic results. Because each function in the process grid can have a different signature, the operation flattens all process inputs into one operand list and flattens all process outputs into one result list. The order is row-major over the process grid: first replica 0 partition 0, then replica 0 partition 1, and so on.

The verifier checks several structural facts:

```text
programs must be non-empty
all rows of the programs grid must have the same size
every referenced function must exist
the number of operands must match the sum of all referenced function arguments
the number of results must match the sum of all referenced function results
infeed, if present, must be non-empty and reference valid functions
infeed functions must return exactly one tensor
```

At runtime, the evaluator builds a `ProcessGrid`, creates one `Process` for each replica and partition coordinate, and evaluates the referenced functions in a thread pool. This is how the reference interpreter tests collective and communication-like StableHLO behavior. Operations such as all-reduce, all-gather, send, receive, infeed, outfeed, replica ID, and partition ID need a notion of multiple processes to be meaningful.

Use `interpreter.run_parallel` when a StableHLO test needs to execute the same or different programs across a process grid. Do not use it as a general parallel programming abstraction. It is an interpreter support operation for reference testing, not a portable StableHLO computation operation.

## `interpreter.print`

`interpreter.print` prints a tensor value to stdout while the reference interpreter is running.

```mlir
interpreter.print %operand : tensor<i1>
```

It takes one tensor operand and produces no result. The evaluator prints the SSA value name followed by the tensor contents. This is useful when debugging small programs and trying to understand what the interpreter sees at a particular point.

The operation is deliberately simple. It is not a logging framework, and it is not meant for collecting large amounts of data. Printing every element of a large tensor can be noisy and slow. For systematic capture of intermediate values, `interpreter.probe` is the better operation.

## `interpreter.probe`

`interpreter.probe` records a tensor value while preserving the value in the IR. It has the same operand and result type, so users of the original value can be rewired to use the probe result.

```mlir
%result = interpreter.probe %operand, probe_id = "probe0" : tensor<3xi32>
```

The operation writes the runtime tensor value to disk in NumPy `.npy` format. The output directory comes from the interpreter configuration, exposed by the translate tool through its probe output option. The interpreter also appends metadata to an index file so the serialized file can be associated with the original probe ID and tensor type.

`probe_id` should be unique for each probe operation in a module. A probe may execute more than once, especially inside loops or repeatedly called functions. In that case, each execution gets a separate serialized file. The implementation uses an increasing file ID for the actual filename, which avoids unsafe filenames even when the logical probe ID contains characters that would be awkward on a filesystem.

The evaluator adds the probe result to the interpreter scope with the same runtime value as the operand. That means `interpreter.probe` observes the value but does not change the program's dataflow semantics.

Use `interpreter.probe` when comparing the StableHLO reference interpreter against another runtime. A test can instrument intermediate values, run the interpreter to produce `.npy` files, run another backend, and compare the captured tensors.

## Probe Instrumentation Pass

The dialect registers one transformation pass:

```text
interpreter-instrument-with-probe
```

This is a `ModuleOp` pass that inserts `interpreter.probe` operations after suitable operations in a StableHLO program. The pass has one option:

```text
useDebugInfo
```

When `useDebugInfo` is false, generated probes use names such as `probe1`, `probe2`, and so on. When `useDebugInfo` is true and a result operation has a named location, the pass uses the location name as the base of the probe ID, then appends a numeric suffix.

The pass deliberately skips constants, operations with no results, and values that are not statically shaped ranked tensors. It only instruments tensor values whose element type is integer, floating-point, or complex. It does not instrument tokens, tuples, dynamically shaped tensors, string tensors, or operations with no tensor return values.

This pass is important because hand-placing probes is tedious and error-prone. Instrumentation can be generated across a whole module, including operations nested inside regions such as `stablehlo.if` or `stablehlo.while`. That makes it practical to capture a broad trace of intermediate tensors.

## Interpreter API Flow

The public interpreter API can evaluate a parsed MLIR module with a list of runtime inputs. The configuration includes:

```text
probeInstrumentationDir
mainFunction
fallback
```

`mainFunction` selects the entry point. If a module has exactly one function and the default name `main` is used, the interpreter can use that single function even if its symbol name is different. The API validates that the number and types of runtime inputs match the selected entry function.

Before evaluation, the API may run supporting StableHLO pipelines. If the entry function has dynamic argument shapes, it uses the runtime input values to refine those shapes through the StableHLO remove-dynamism pipeline. If quantized types are present, it runs the StableHLO quantization lowering pipeline so the interpreter can evaluate primitive math instead of unsupported quantized constructs.

The fallback mechanism handles operations that are not matched by ordinary StableHLO interpreter kernels. The default fallback recognizes the `interpreter` dialect operations, including `print`, `probe`, and `run_parallel`. The command-line translate tool also installs a fallback for StableHLO's test `check` operations so interpreter tests can assert expected results.

## Command-Line Use

The main command-line entry point is:

```text
stablehlo-translate --interpret
```

The translate registration loads the StableHLO dialect, the StableHLO `check` dialect, the `interpreter` dialect, `func`, and `quant`. It parses interpreter arguments from the command line, evaluates the module, and prints the results. An option can print results as dense MLIR attributes instead of the default interpreter value format.

This command is central to StableHLO's own interpreter tests. A test file can define a small StableHLO function, include constants and expected checks, and then run the program through `stablehlo-translate --interpret`.

The important implication is that the interpreter is not a compiler lowering target. You do not lower a production StableHLO program into `interpreter` IR. Instead, you run StableHLO through a reference evaluator, with a small helper dialect available for cases where tests need process grids, printing, or probes.

## Transformations And Conversions

The `interpreter` dialect has one direct transformation pass, `interpreter-instrument-with-probe`. It has no standalone dialect conversion pass in this checkout. There is no `convert-interpreter-to-llvm`, no `convert-interpreter-to-stablehlo`, and no production lowering pipeline whose goal is to erase interpreter operations into executable target code.

The broader interpreter API does perform preparatory transformations before execution:

```text
dynamic entry shapes -> refined static shapes
quantized StableHLO constructs -> primitive math-oriented form
ordinary StableHLO program -> runtime InterpreterValue results
```

Those are not conversions of the `interpreter` dialect itself. They are support steps that make a StableHLO module evaluable by the reference interpreter.

The `interpreter.probe` operation also participates in a data conversion at runtime: tensor values are serialized to NumPy `.npy` files and indexed with metadata. This is not an MLIR dialect conversion, but it is an important representation boundary for debugging and cross-runtime comparison.

## When To Use It

Use the `interpreter` dialect when working with the StableHLO reference interpreter. Use `interpreter.run_parallel` for tests that need a process grid. Use `interpreter.print` for quick, small-value debugging. Use `interpreter.probe` when you need durable intermediate tensor captures that can be compared against another runtime.

Use `interpreter-instrument-with-probe` when you need broad instrumentation and do not want to manually place probe operations. It is especially useful when testing backend correctness, because it can capture values from nested regions as the reference interpreter executes them.

Do not use the dialect to describe model semantics. The portable semantics belong in StableHLO. The `interpreter` dialect is intentionally outside the StableHLO spec, so a consumer that accepts StableHLO portable artifacts should not be expected to accept `interpreter.*` operations.

Do not treat interpreter results as performance signals. The interpreter prioritizes clarity and reference behavior. Its tensor storage, element handling, process-grid execution, and file I/O are designed for correctness and observability, not for matching an optimized runtime.

## How To Read interpreter IR

When you see `interpreter.run_parallel`, first inspect the `programs` grid. The shape of that attribute tells you the number of replicas and partitions. Then count the referenced functions' arguments and results to understand how the flattened operands and results are partitioned across processes.

When you see `interpreter.print`, treat it as a temporary debugging side effect. It does not produce a value and should not affect tensor computation.

When you see `interpreter.probe`, treat it as an identity value with a side effect. The result has the same type and runtime value as the input, but the interpreter writes the value to disk when the operation executes.

The main implication is that `interpreter` IR is a testing and observation layer. It gives the StableHLO project a practical reference execution environment while keeping the StableHLO specification focused on portable computation.

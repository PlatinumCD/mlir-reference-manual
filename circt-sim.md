# CIRCT sim Dialect

The CIRCT `sim` dialect models simulator-specific behavior in hardware IR. It gives CIRCT a high-level representation for things that interact with Verilog and SystemVerilog simulators: command-line plusargs, DPI-C calls, formatted printing, dynamic strings, queues, output streams, file I/O, simulation termination, pauses, and clock-triggered procedural regions.

For a beginner, the key idea is that `sim` captures simulation intent before it becomes raw SystemVerilog system tasks or simulator-specific glue. A hardware design may be synthesizable, but a testbench, debug harness, assertion environment, or co-simulation wrapper often needs printing, file descriptors, dynamic strings, DPI imports, and `$finish`-style control. The `sim` dialect keeps those concepts explicit long enough for CIRCT passes to analyze, fold, reorder, proceduralize, or lower them.

This dialect is important because simulator behavior is not ordinary hardware. Printing has ordering constraints. DPI calls have ABI direction semantics. Plusargs read command-line state. File I/O has resources. A triggered simulation block depends on a clock edge. Encoding all of this immediately as low-level SV would make analysis harder.

## Operation Inventory

The `sim` dialect defines these operations:

```text
sim.clocked_pause
sim.clocked_terminate
sim.flush
sim.fmt.bin
sim.fmt.char
sim.fmt.concat
sim.fmt.current_time
sim.fmt.dec
sim.fmt.exp
sim.fmt.flt
sim.fmt.gen
sim.fmt.hex
sim.fmt.hier_path
sim.fmt.literal
sim.fmt.oct
sim.fmt.string
sim.func.dpi
sim.func.dpi.call
sim.get_file
sim.pause
sim.plusargs.test
sim.plusargs.value
sim.print
sim.proc.print
sim.queue.cmp
sim.queue.concat
sim.queue.delete
sim.queue.empty
sim.queue.from_array
sim.queue.get
sim.queue.insert
sim.queue.pop_back
sim.queue.pop_front
sim.queue.push_back
sim.queue.push_front
sim.queue.resize
sim.queue.set
sim.queue.size
sim.queue.slice
sim.stderr_stream
sim.stdout_stream
sim.string.concat
sim.string.get
sim.string.int_to_string
sim.string.length
sim.string.literal
sim.sv.channel_to_output_stream
sim.sv.fclose
sim.sv.fflush_all
sim.sv.fopen
sim.terminate
sim.triggered
```

The dialect defines these main types:

```text
!sim.fstring
!sim.dstring
!sim.queue<element-type, bound>
!sim.dpi_functy<...>
!sim.output_stream
```

`!sim.fstring` is a formatted-string fragment or concatenation. `!sim.dstring` is a runtime-length string with value semantics. `!sim.queue` models SystemVerilog-like queues. `!sim.dpi_functy` preserves DPI-C argument names, types, and directions. `!sim.output_stream` models a simulator output destination.

## Plusargs And DPI

`sim.plusargs.test` and `sim.plusargs.value` wrap SystemVerilog `$test$plusargs` and `$value$plusargs`. They are pure from the IR perspective because they produce values from a format string instead of exposing all the wires, registers, initial blocks, and ifdefs needed by a concrete SV implementation. `sim.plusargs.test` returns whether a plusarg was found. `sim.plusargs.value` returns both a found flag and the parsed value.

`sim.func.dpi` declares a DPI-C imported function. Its type stores first-class argument directions: `in`, `out`, `inout`, `return`, and `ref`. This is more precise than a plain function type because DPI direction affects both call operands and the generated C/SystemVerilog ABI. `sim.func.dpi.call` calls a DPI function and can optionally carry a clock and enable. Without a clock, the call behaves combinationally as inputs change. With a clock, the call happens on a positive clock edge and results behave like registered state.

These ops are important in co-simulation and simulator integration. They let hardware IR call into external C code while retaining enough semantic detail to lower correctly to SV DPI imports and calls.

## Formatting, Printing, And Streams

The `sim.fmt.*` operations build `!sim.fstring` values. `sim.fmt.literal` is a raw fragment. `sim.fmt.string` formats a dynamic string. `sim.fmt.hex`, `sim.fmt.oct`, `sim.fmt.bin`, `sim.fmt.dec`, and `sim.fmt.char` format integer values. `sim.fmt.exp`, `sim.fmt.flt`, and `sim.fmt.gen` format floating-point values. `sim.fmt.current_time` represents the current simulation time. `sim.fmt.hier_path` represents the current hierarchical instance path. `sim.fmt.concat` joins fragments.

`sim.print` is non-procedural: it prints on a given clock edge when a condition is true. It can optionally print to an output stream. `sim.proc.print` is the procedural version used inside procedural regions. `sim.get_file` creates an output stream from a formatted file name. `sim.stdout_stream` and `sim.stderr_stream` produce standard output and standard error stream resources. `sim.flush` flushes an output stream.

The distinction between formatted strings and dynamic strings matters. A formatted string is a recipe for printing values with format specifiers. A dynamic string is a runtime value that behaves like a string.

## Dynamic Strings And Queues

Dynamic string operations include `sim.string.literal`, `sim.string.length`, `sim.string.concat`, `sim.string.int_to_string`, and `sim.string.get`. They represent SystemVerilog-style runtime strings. Constant forms have folding support, so static string computations can often collapse before lowering.

Queue operations model SystemVerilog queue behavior with value semantics. `sim.queue.empty` creates an empty queue. `sim.queue.size` returns its length. `sim.queue.push_back`, `sim.queue.push_front`, `sim.queue.pop_back`, and `sim.queue.pop_front` model insertion and removal at the ends. `sim.queue.insert`, `sim.queue.delete`, `sim.queue.set`, `sim.queue.get`, and `sim.queue.slice` model indexed access and updates. `sim.queue.concat` joins queues, `sim.queue.resize` changes bounds, `sim.queue.from_array` creates a queue from an array, and `sim.queue.cmp` compares queues for equality or inequality.

Queues are useful for testbench-like code and simulator models where dynamic collections appear, but they are not normal hardware memories.

## Simulation Control And File I/O

`sim.terminate` and `sim.pause` are procedural simulation-control operations. They correspond to `$finish` or `$fatal`, and `$stop`-style behavior. `sim.clocked_terminate` and `sim.clocked_pause` are non-procedural forms guarded by clock and condition.

`sim.triggered` is a procedural region that executes on a positive clock edge, optionally guarded by a condition. It provides a structured representation for clocked simulator behavior before lowering to `sv.always`.

The `sim.sv.*` operations model SystemVerilog file-oriented system tasks and conversions. `sim.sv.fopen` opens a file and returns a descriptor. `sim.sv.fclose` closes it. `sim.sv.fflush_all` flushes all open files. `sim.sv.channel_to_output_stream` turns a SystemVerilog channel or file descriptor into `!sim.output_stream`.

## Local Transformations

`sim-fold-value-formatters` converts constant formatted values to literals. It uses the fold behavior of formatter-producing operations to replace constant formatted fragments with literal `!sim.fstring` values where possible.

`sim-proceduralize` transforms non-procedural simulation operations with clock and enable into procedural operations wrapped in a procedural region. For example, clocked prints or control operations can become operations inside a `sim.triggered` region. This pass depends on `hw`, `seq`, and `scf` because it has to build clocked procedural structure.

`sim-squash-triggered` merges multiple `sim.triggered` operations in the same block that share a clock. If original triggers have different conditions, the pass may materialize `scf.if` inside the merged body. It has a `convert-to-hw` option that converts squashed `sim.triggered` ops to `hw.triggered`.

`sim-lower-dpi-func` lowers `sim.func.dpi` declarations into `func.func` for simulation flows. It preserves the DPI function shape while moving the declaration toward a lower-level function representation.

## Conversion To SystemVerilog

The conversion pass `lower-sim-to-sv` lowers simulator-specific Sim operations to SV, HW, Seq, Comb, and Emit constructs. It is the main exit path from the dialect.

The Sim-to-SV type converter maps `!sim.output_stream` to `i32`, because SystemVerilog channels and file descriptors are represented as 32-bit values. Standard output and standard error lower to fixed multichannel descriptor constants: `sim.stdout_stream` to `32'h8000_0001` and `sim.stderr_stream` to `32'h8000_0002`.

`sim.plusargs.test` lowers to an SV initial block using `test$plusargs`, with a register that is read back as the result. `sim.plusargs.value` lowers to SV logic around `value$plusargs`, including synthesis guards and dummy assignments to avoid undriven warnings.

`sim.triggered` lowers to an `sv.always` region on the positive edge of the converted clock. If it has a condition, the body is placed inside an `sv.if`. The lowering also uses synthesis guards where needed.

`sim.func.dpi` declarations lower to SV DPI import declarations and associated emitted fragments with include guards. `sim.func.dpi.call` lowers to procedural SV function calls. Clocked calls go into positive-edge always blocks; unclocked calls go into `always_comb`. Enable operands become SV if conditions, and disabled unclocked results are assigned X values.

Formatted print lowering builds SV format strings and arguments, lowers procedural printing to file writes, and cleans up dead formatting operations afterward. `sim.flush` lowers to `sv.fflush`. Simulation-control ops lower to `sv.finish`, `sv.fatal`, or `sv.stop` variants.

## When To Use It

Use `sim` when the design or test harness needs simulator behavior that should remain analyzable before SV emission: plusargs, formatted output, runtime strings, queues, DPI imports, file I/O, and simulation termination. It is also useful when another dialect, such as Arc or Verif, needs to express simulation-only behavior before a later pass emits SV.

Do not use `sim` for synthesizable datapath or control logic. Use `hw`, `seq`, `comb`, `sv`, `firrtl`, or other CIRCT hardware dialects for that. `sim` is for the surrounding simulator interface and testbench-like behavior.

When reading Sim IR, first group operations by resource: command-line state, DPI, strings, queues, output streams, file descriptors, and clocked procedural regions. Then check whether the pipeline will proceduralize, squash triggers, or lower directly to SV. The implication is that `sim` keeps simulator semantics structured until the compiler is ready to emit concrete SystemVerilog constructs.

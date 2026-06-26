# CIRCT `seq` Dialect

The CIRCT `seq` dialect models digital sequential logic. In hardware terms, it is where CIRCT represents state, clocks, registers, FIFOs, memories, and initialization values before those constructs are lowered to SystemVerilog or to another hardware-oriented backend.

For a beginner, the central distinction is simple: `comb` describes combinational logic, while `seq` describes logic that depends on time. A register stores a value across cycles. A memory keeps state. A FIFO remembers which elements have been written and read. Clock operations describe how sequential elements are driven.

`seq` is not meant to be a Verilog syntax dialect. It is a hardware semantics dialect. It preserves the idea of "this is a register" or "this is a FIRRTL memory" before the compiler chooses a concrete implementation such as `sv.reg`, `sv.always_ff`, a generated memory module, or a behavioral simulation model.

## When Seq Is Important

`seq` is important whenever CIRCT IR contains stateful hardware. You will see it after frontends or higher-level dialects have lowered into a hardware form, but before final emission or formal lowering has erased sequential abstractions.

Common sources include FIRRTL lowering, FSM lowering, Calyx-to-HW lowering, Handshake-to-HW lowering, DC-to-HW lowering, Pipeline-to-HW lowering, LTL lowering, AIGER import, and several simulation or verification flows. These pipelines use `seq` because registers and memories need a shared representation across CIRCT.

Use this dialect when you need to answer questions like:

- Which values are stored across clock cycles?
- What clock drives a register, memory port, FIFO, or clock gate?
- Is a reset present, and what reset value is used?
- Is this memory a high-level abstract memory, a FIRRTL-flavored memory, or a generated simulation memory?
- Are clock gates, clock dividers, or clock casts still explicit?
- Has initialization already been lowered to SystemVerilog `initial` behavior?

## Why It Is Needed

Hardware state is not a normal SSA value. SSA values describe dataflow in one instant. A register, by contrast, connects a value from one clock edge to a value visible in the next cycle. A memory has ports, read and write timing, read-under-write behavior, write-under-write behavior, masks, and initialization.

Without `seq`, CIRCT would have to lower state immediately into low-level SystemVerilog operations. That would make optimization and analysis harder because the compiler would have to recover register and memory intent from procedural blocks. `seq` keeps the stateful construct explicit while still fitting into MLIR's operation and type system.

The implication is that `seq` sits between abstract hardware dialects and concrete emission. It is high-level enough to analyze and transform state, but concrete enough to lower to RTL.

## Type Inventory

The local `seq` dialect defines four types.

`!seq.clock` is a dedicated type for clock-carrying values. It prevents a clock from being confused with an arbitrary `i1` signal during the stages where clock semantics matter. Cast operations convert between `i1` and `!seq.clock` when needed.

`!seq.hlmem<...>` is a high-level multidimensional memory type. It has a static shape and an element type. Address types are derived from the memory dimensions, so each address operand has the bit width needed to index its dimension.

`!seq.firmem<depth x width>` is a FIRRTL-flavored memory type. It stores memory parameters needed by FIRRTL-style ports and lowering, including depth, data width, and optionally mask width.

`!seq.immutable<T>` wraps a value that is immutable after initialization. It is used with `seq.initial` and register initial values. The wrapper says that a value is produced during initialization and can later be unwrapped or used as an initial register value.

## Attribute Inventory

The dialect defines clock and memory attributes.

`#seq.clock_constant<low>` and `#seq.clock_constant<high>` represent constant clock values. The `seq.const_clock` operation materializes these.

Read-under-write behavior is represented by the enum values `undefined`, `old`, and `new`. Write-under-write behavior is represented by `undefined` and `port_order`. These appear on `seq.firmem` and guide memory generation.

`#seq.firmem.init<filename, isBinary, isInline>` records initialization information for a FIRRTL memory. It models `$readmemh` and `$readmemb` style initialization. The filename identifies the data source, `isBinary` distinguishes binary from hexadecimal interpretation, and `isInline` controls whether initialization is emitted in the memory model or split out into a bound module.

## Operation Inventory

The local `seq` dialect defines 22 operations.

### Registers and Shift Registers

`seq.compreg` is the core computational register. It captures an input value on the positive edge of a clock and produces the stored data. It may have a reset and reset value, and it may have an immutable initial value. The operation is reset-style agnostic; lowering decides how to implement reset behavior.

`seq.compreg.ce` is a computational register with an explicit clock-enable signal. When the enable is asserted, the input is captured. This form is convenient for mapping to target primitives or later lowering.

`seq.shiftreg` represents a shift register with a fixed number of elements. It shifts its input through a chain of registers when its clock enable is asserted. Optional reset and power-on values apply to the shift elements.

`seq.firreg` is a FIRRTL-flavored register. It preserves behavior expected by FIRRTL lowering, including synchronous or asynchronous reset, preset values, names, and inner symbols. It lowers through specialized FIR register lowering rather than through the generic computational register path.

### High-Level FIFO and Memory

`seq.fifo` is a high-level FIFO. It has input data, read enable, write enable, clock, reset, depth, read latency, and optional almost-full and almost-empty thresholds. It returns output data plus full and empty flags, with optional almost-full and almost-empty flags.

`seq.hlmem` allocates a high-level memory. It produces a handle of `!seq.hlmem<...>` and is paired with structural port operations.

`seq.read` is a read port for `seq.hlmem`. It takes a memory handle, one address per memory dimension, an optional read-enable signal, and a latency attribute. It returns the read data.

`seq.write` is a write port for `seq.hlmem`. It takes a memory handle, addresses, input data, write enable, and latency. It has no result because it updates memory state.

### FIRRTL-Flavored Memories

`seq.firmem` declares a FIRRTL-flavored memory. It records read latency, write latency, read-under-write behavior, write-under-write behavior, optional name, optional inner symbol, optional initialization attribute, optional prefix, and optional output file. Its result is a `!seq.firmem` memory handle.

`seq.firmem.read_port` reads from a `seq.firmem`. It takes the memory, address, clock, and optional enable. If the enable is omitted, it behaves like a constant true enable.

`seq.firmem.write_port` writes to a `seq.firmem`. It takes the memory, address, clock, optional enable, data, and optional mask. A mask is only valid if the memory type specifies a mask width.

`seq.firmem.read_write_port` is a combined read/write port. Its mode operand selects read or write behavior. It also supports optional enable and optional mask.

### Clock Operations

`seq.const_clock` produces a constant clock value, either low or high.

`seq.to_clock` casts an `i1` wire value to `!seq.clock`.

`seq.from_clock` casts a `!seq.clock` value back to `i1`.

`seq.clock_gate` safely gates a clock with an enable signal and optional test-enable signal. If enabled, the output follows the input clock. If disabled, the output is held low. The enable is sampled at the rising edge of the input clock.

`seq.clock_mux` selects between two clocks based on an `i1` condition.

`seq.clock_div` produces a clock divided by a power of two.

`seq.clock_inv` inverts a clock.

### Initialization Operations

`seq.initial` produces values of `!seq.immutable<T>`. It contains a single block and is used to build initialization-time values for registers and other stateful constructs.

`seq.yield` terminates a `seq.initial` region and yields the initialized values.

`seq.from_immutable` unwraps an immutable value back to the underlying wire type. During Seq-to-SV lowering, immutable values are mapped into `sv.initial` behavior or register initialization.

## Transformations and Conversions

The dialect has seven Seq-owned transform passes.

`lower-seq-compreg-ce` rewrites `seq.compreg.ce` into `seq.compreg` by folding the clock enable into the next-state value:

```text
next := mux(clock_enable, next, current)
```

`lower-seq-shiftreg` lowers `seq.shiftreg` into a chain of `seq.compreg.ce` operations. This is a conservative fallback lowering; target-specific flows may replace shift registers with device primitives instead.

`lower-seq-fifo` lowers `seq.fifo` into registers, a high-level memory, combinational pointer/count logic, and verification assertions. It creates read and write pointers, count state, full/empty flags, optional almost-full/empty flags, and protocol assertions for invalid reads or writes.

`lower-seq-hlmem` lowers `seq.hlmem`, `seq.read`, and `seq.write` to a simple behavioral SystemVerilog memory implementation. The local fallback supports unidimensional memories and handles read latency by registering addresses or read data as needed.

`externalize-clock-gate` replaces `seq.clock_gate` operations with instances of an external clock-gate module. Options control the external module name, port names, test-enable port name, and instance name. This is useful when a technology library provides the real clock-gating cell.

`hw-memory-sim` turns generated FIRRTL memory modules into simulation models. It can add read-enable behavior, randomization code, `$readmemh` or `$readmemb` style initialization, mux pragmas, macro-replacement preparation, and synthesis workarounds.

`seq-reg-of-vec-to-mem` recognizes register arrays that behave like memories and rewrites them to `seq.firmem` with read and write ports. It looks for patterns built from array reads, array injects, mux-controlled updates, and a shared clock.

There are also two direct Seq conversion passes.

`lower-seq-to-sv` lowers the remaining Seq dialect to SystemVerilog-oriented IR. It maps `!seq.clock` to `i1`, lowers `seq.compreg` and `seq.compreg.ce` to `sv.reg` plus `sv.always_ff` or `sv.always`, lowers clock gates/inverters/muxes/dividers, lowers constant clocks, lowers immutable initialization into `sv.initial`, and invokes FIR memory/register lowering utilities. Options control register randomization, memory randomization, separate always blocks, and use of `always_ff`.

`lower-seq-firmem` lowers `seq.firmem` memories to instances of generated HW modules. It groups memories by configuration, creates `hw.module.generated` declarations with a FIRRTL memory schema, and replaces memory declarations and ports with module instances.

Seq also participates in broader conversions. `lower-firrtl-to-hw` creates `seq.firreg`, `seq.firmem`, FIR memory ports, and clock operations from FIRRTL constructs. FSM lowering creates `seq.compreg` state and variable registers. Calyx, Handshake, DC, Pipeline, and LTL lowering use Seq for registers, memories, and clocks. Verification and export paths such as HW-to-BTOR2, HW-to-SMT, AIGER export/import, Arc lowering, and Sim-to-SV either consume Seq operations or legalize clock casts around them.

## What It Implies

Seeing `seq.compreg` means the design has a register that stores a value across cycles. Check the input, clock, reset, reset value, initial value, and name. The input is the next-state value; the result is the current stored value.

Seeing `seq.firreg` means the register came from a FIRRTL-sensitive path. Pay attention to sync versus async reset, preset, name, and inner symbol, because these affect emitted SystemVerilog and randomization behavior.

Seeing `seq.hlmem` means memory is still high-level and abstract. Look for `seq.read` and `seq.write` users to understand its ports. Seeing `seq.firmem` means the memory is in the FIRRTL-flavored representation, with explicit read, write, or read-write ports.

Seeing `!seq.clock` means the compiler is still treating clocks as a special kind of value. Casts through `seq.to_clock` and `seq.from_clock` show where a normal `i1` signal crosses into or out of clock semantics.

Seeing `seq.initial` means initialization is still represented structurally. Its body computes immutable values, and `seq.yield` returns them. Later lowering will merge initial regions and emit SystemVerilog initialization code.

## How To Read Seq IR

Start by finding the clocks. Registers, memory ports, FIFOs, and clock operations all depend on them. If a clock is produced by `seq.to_clock`, trace the underlying `i1` source.

Next, read registers as feedback points. A `seq.compreg` result may feed logic that eventually computes its next input. The operation itself marks the cycle boundary.

For memories, first identify the memory declaration, then inspect all users of its handle. A high-level memory uses `seq.read` and `seq.write`; a FIRRTL memory uses `seq.firmem.read_port`, `seq.firmem.write_port`, and `seq.firmem.read_write_port`.

Finally, inspect lowering state. If `seq.fifo`, `seq.shiftreg`, or `seq.hlmem` still exist, higher-level lowering remains. If only `seq.compreg`, `seq.firreg`, clocks, and `seq.firmem` remain, the IR is closer to SystemVerilog or verification lowering.

## Minimal Example

This small register captures `%next` on `%clk` and resets to zero:

```mlir
%zero = hw.constant 0 : i8
%q = seq.compreg %next, %clk reset %rst, %zero : i8
```

Read it as "on a clock edge, store `%next`; when reset is active, store `%zero`; expose the current stored value as `%q`." A later lowering can turn this into an `sv.reg` and an `always_ff` block, but while it remains in `seq`, the compiler still sees it as a register.

# CIRCT CHIRRTL Dialect

## Beginner Summary

The CIRCT `chirrtl` dialect is a small companion dialect to FIRRTL for
high-level behavioral memories.

It models memories before the compiler has decided exactly what FIRRTL memory
ports they need. A CHIRRTL memory starts as a behavioral object, and separate
operations declare and access ports. Later, the `firrtl-lower-chirrtl` pass
infers the concrete port kinds and rewrites the CHIRRTL memory into standard
FIRRTL memory operations.

Think of `chirrtl` as the "memory inference" layer for FIRRTL. It lets frontend
or imported FIRRTL-style IR say "there is a memory, and here are its port
uses," then lets the compiler determine whether each port is read-only,
write-only, read-write, or debug.

## Why This Dialect Exists

Hardware memories are not just arrays. Their observable behavior depends on
ports, clocks, enable signals, read latency, write latency, masks, write modes,
read-under-write behavior, and whether a port is used for reads, writes, or
both.

In source-level FIRRTL and Chisel-style designs, memory intent can be more
behavioral than structural. A designer may describe a memory and use a port,
while the compiler is still expected to infer the exact FIRRTL memory port
shape.

The `chirrtl` dialect exists to hold that pre-inference representation.

It fills the gap between:

- high-level memory declarations that are easy for a frontend to emit;
- standard FIRRTL `firrtl.mem` operations with fully determined port bundles;
- later CIRCT lowering to HW, SV, or generated memory modules.

Without CHIRRTL, a frontend would need to eagerly know whether each memory port
is read, write, read-write, or debug. CHIRRTL lets CIRCT derive that from actual
uses.

## When It Matters

You will see `chirrtl` early in CIRCT FIRRTL pipelines, before memory lowering
has completed.

A typical path looks like this:

```text
CHIRRTL memory ops
  -> firrtl-lower-chirrtl
  -> standard FIRRTL memories and ports
  -> firrtl-flatten-memory / firrtl-lower-memory / memory cleanup
  -> FIRRTLToHW
  -> HW/SV
```

The dialect matters most when debugging memory behavior. If a memory port is
inferred incorrectly, or if a port is never enabled, the issue is usually
visible around `chirrtl.memoryport` and `chirrtl.memoryport.access` before
`firrtl-lower-chirrtl` rewrites them.

## When To Use It

Use `chirrtl` when representing FIRRTL/Chisel-style behavioral memories whose
ports still need inference.

Use it for:

- combinational memories with read latency `0` and write latency `1`;
- sequential memories with read and write latency `1`;
- memory ports whose final kind can be inferred from reads and writes;
- debug ports that observe memory contents;
- FIRRTL import paths that preserve high-level memory intent.

Do not use `chirrtl` once the memory interface is already structural. At that
point, use regular FIRRTL memory operations.

Do not use it as a generic memory abstraction for software-like arrays. It is
hardware-memory specific and depends on FIRRTL types, clocks, directions, and
memory semantics.

## Core Concepts

### Behavioral Memories

The main CHIRRTL type is `!chirrtl.cmemory<element-type, element-count>`.

It represents a behavioral memory with an element type and depth, but without
fixed ports. For example:

```mlir
!chirrtl.cmemory<uint<32>, 16>
!chirrtl.cmemory<bundle<a : uint<1>>, 16>
```

The element type is a FIRRTL base type. The element count is the memory depth.

### Memory Ports

The second CHIRRTL type is `!chirrtl.cmemoryport`.

It represents a declared port on a CHIRRTL memory. A port has a data result and
a port-token result. The data result is used by ordinary FIRRTL operations.
The token result is passed to `chirrtl.memoryport.access` to define address,
clock, and enable behavior.

### Port Kind Inference

`chirrtl.memoryport` has a declared direction attribute, but the lowering pass
also inspects uses of the port data.

The port can become:

- `Read`, if the data is only read;
- `Write`, if the data is only written;
- `ReadWrite`, if the data is both read and written;
- deleted, if the port is unused and has no meaningful inferred direction.

This is the main service CHIRRTL provides.

### Combinational And Sequential Memories

CHIRRTL distinguishes:

- combinational memory, created by `chirrtl.combmem`;
- sequential memory, created by `chirrtl.seqmem`.

Combinational memories have read latency `0` and write latency `1`.
Sequential memories have both read latency and write latency `1`.

Sequential memories also carry FIRRTL read-under-write behavior, represented
by the `RUWBehavior` attribute.

### Enable Inference

The lowering pass wires memory enables when it creates FIRRTL memory ports.
Most port kinds are enabled at the port access point.

Sequential read ports are trickier. The pass may infer enable points from the
operation that defines the address:

- if the address is a wire or register, enable is associated with valid drives
  to that address;
- if the address is a node, enable is placed at the node;
- otherwise, the pass may fail to infer an enable and warn that the port is
  never enabled.

This behavior is one reason CHIRRTL is a temporary dialect. It captures source
intent until the compiler can make these port and enable decisions.

## Types

| Type | Meaning |
| --- | --- |
| `!chirrtl.cmemory<element, depth>` | Behavioral memory with FIRRTL element type and fixed depth. |
| `!chirrtl.cmemoryport` | Token-like handle for a declared memory port. |

## Operations

The CHIRRTL dialect currently defines five operations:

| Operation | Purpose |
| --- | --- |
| `chirrtl.combmem` | Define a behavioral combinational memory. |
| `chirrtl.seqmem` | Define a behavioral sequential memory. |
| `chirrtl.memoryport` | Declare a port on a CHIRRTL memory. |
| `chirrtl.memoryport.access` | Provide address and clock for a memory port access. |
| `chirrtl.debugport` | Declare a debug observation port on a CHIRRTL memory. |

### `chirrtl.combmem`

`chirrtl.combmem` defines a combinational behavioral memory.

It produces a `!chirrtl.cmemory` value. A combinational memory has read latency
`0` and write latency `1`. It may carry a name, name-kind, annotations, an
inner symbol, an initialization attribute, and a prefix.

Example shape:

```mlir
%mem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 256>
```

Use this operation when reads are intended to be combinational before memory
lowering determines concrete ports.

### `chirrtl.seqmem`

`chirrtl.seqmem` defines a sequential behavioral memory.

It also produces a `!chirrtl.cmemory`, but it models a memory whose reads and
writes both have latency `1`. It carries read-under-write behavior, because
sequential memory semantics depend on what happens when the same address is
read and written in the same cycle.

Example shape:

```mlir
%mem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 256>
```

Use it when a memory should be treated as a clocked memory rather than a
combinational lookup.

### `chirrtl.memoryport`

`chirrtl.memoryport` declares a port on a `chirrtl.combmem` or
`chirrtl.seqmem`.

It takes a `!chirrtl.cmemory` and returns two values:

- the port data value, whose type is the memory element type;
- a `!chirrtl.cmemoryport` handle used by `chirrtl.memoryport.access`.

The operation also has a direction such as `Read`, `Write`, `ReadWrite`, or
`Infer`. If the direction is `Infer`, the lowering pass decides the final kind
from uses of the data value.

Example shape:

```mlir
%data, %port = chirrtl.memoryport Infer %mem {name = "p0"}
  : (!chirrtl.cmemory<uint<8>, 256>)
    -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
```

### `chirrtl.memoryport.access`

`chirrtl.memoryport.access` enables a declared memory port at a particular
address and clock.

It takes:

- the `!chirrtl.cmemoryport` handle;
- a FIRRTL integer index;
- a FIRRTL clock.

Example shape:

```mlir
chirrtl.memoryport.access %port[%addr], %clock
  : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
```

The operation does not itself read or write data. Instead, it supplies the
access metadata that the lowering pass uses to wire the generated FIRRTL memory
port's `addr`, `en`, and `clk` fields.

### `chirrtl.debugport`

`chirrtl.debugport` declares a debug memory port.

It takes a CHIRRTL memory and returns a FIRRTL reference type. In lowering, it
becomes a debug port on the generated FIRRTL memory. Use it for observation and
debugging paths rather than normal read/write hardware behavior.

## Transformations And Conversions

### `firrtl-lower-chirrtl`

This is the key CHIRRTL pass.

It runs on `firrtl.module` operations. It finds `chirrtl.combmem` and
`chirrtl.seqmem` operations, collects their `chirrtl.memoryport` and
`chirrtl.debugport` users, infers port kinds, and creates standard FIRRTL
memory operations.

Important rewrite behavior:

- portless memories are deleted;
- unused ports are deleted;
- CHIRRTL memory ports are sorted by name before the FIRRTL memory is created;
- combinational memories become FIRRTL memories with read latency `0` and
  write latency `1`;
- sequential memories become FIRRTL memories with read latency `1` and write
  latency `1`;
- read ports produce FIRRTL `data` fields;
- write ports produce FIRRTL `data` and `mask` fields;
- read-write ports produce FIRRTL `rdata`, `wmode`, `wdata`, and `wmask`
  fields;
- debug ports become debug memory ports;
- annotations, inner symbols, prefixes, names, and memory initialization
  attributes are preserved where applicable.

This pass is a conversion from CHIRRTL to FIRRTL. After it succeeds, the
CHIRRTL memory operations should be gone.

### Port Use Rewriting

While lowering, the pass tracks how port data values are used.

If a use writes to the port data, it rewrites the destination to the generated
FIRRTL write-data field and drives the appropriate mask. If the port is
read-write, it also drives write mode.

If a use reads from the port data, it rewrites the source to the generated
FIRRTL read-data field.

For aggregate element access, such as `firrtl.subfield`, `firrtl.subindex`, or
`firrtl.subaccess`, the pass may clone the access operation so reads go through
read-data paths and writes go through write-data/mask paths.

### Related Memory Passes

These are not CHIRRTL dialect passes, but they commonly appear after CHIRRTL
has been lowered:

- `firrtl-flatten-memory`: flattens aggregate memory element types to UInt and
  inserts bitcasts around accesses;
- `firrtl-lower-memory`: lowers FIRRTL memory operations to generated modules;
- `firrtl-add-seqmem-ports`: adds extra ports to lowered memory modules when
  annotations request it;
- `firrtl-mem-to-reg-of-vec`: implements combinational memories with registers
  and vectors;
- `firrtl-to-hw` / FIRRTLToHW conversion: eventually lowers FIRRTL structures
  toward CIRCT HW and SV.

The important ordering point is that CHIRRTL comes before these. It first turns
behavioral memory and port inference into ordinary FIRRTL memory structure.

## What It Means In A Pipeline

Seeing `chirrtl` in IR means the compiler is still preserving behavioral memory
intent.

The IR is saying:

- this is a memory with a type and depth;
- these operations declare possible ports;
- these uses imply whether a port is read, write, or read-write;
- the compiler still needs to create the exact FIRRTL memory port bundle.

Once `firrtl-lower-chirrtl` runs, the IR becomes less behavioral and more
structural. The memory has explicit FIRRTL ports, fields, masks, enables, and
latencies.

## Common Mistakes

Do not confuse `chirrtl.combmem` with a final hardware memory. It is a
behavioral memory declaration waiting for lowering.

Do not expect `chirrtl.memoryport.access` to be the actual read or write. The
read or write is inferred from how the port data value is used elsewhere.

Do not assume an inferred port will always become read-only. If the same port
data is both read and written, lowering creates a read-write port.

Do not leave CHIRRTL around late in the pipeline. Most later FIRRTL and HW
passes expect standard FIRRTL memory operations, not CHIRRTL memory inference
ops.

## Source Map

| Topic | Source path |
| --- | --- |
| Dialect definition | `circt/include/circt/Dialect/FIRRTL/CHIRRTLDialect.td` |
| CHIRRTL types | `circt/include/circt/Dialect/FIRRTL/CHIRRTLTypes.td` |
| CHIRRTL operations | `circt/include/circt/Dialect/FIRRTL/CHIRRTLOps.td` |
| Dialect implementation | `circt/lib/Dialect/FIRRTL/CHIRRTLDialect.cpp` |
| Type implementation | `circt/lib/Dialect/FIRRTL/CHIRRTLTypes.cpp` |
| Lowering pass declaration | `circt/include/circt/Dialect/FIRRTL/Passes.td` |
| CHIRRTL lowering implementation | `circt/lib/Dialect/FIRRTL/Transforms/LowerCHIRRTL.cpp` |
| Related FIRRTL memory lowering | `circt/lib/Dialect/FIRRTL/Transforms/LowerMemory.cpp` |
| FIRRTL to HW conversion | `circt/lib/Conversion/FIRRTLToHW/LowerToHW.cpp` |
| Tests | `circt/test/Dialect/FIRRTL/lower-chirrtl.mlir` and `circt/test/Dialect/FIRRTL/lower-chirrtl-errors.mlir` |

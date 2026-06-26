# IREE `vm` Dialect

## Beginner Summary

The IREE `vm` dialect is the low-level virtual-machine IR used near the end of
IREE compilation. It represents functions, globals, scalar computation, buffers,
lists, references, control flow, runtime imports, and module metadata in a form
that can be serialized to IREE VM bytecode or lowered further to C through
EmitC.

For a beginner, the key idea is:

```text
util / func / arith / math / HAL-facing compiler IR
  -> iree-vm-conversion
  -> vm.module, vm.func, vm.call, vm.buffer.*, vm.list.*, scalar vm ops
  -> VM transformation pipeline
  -> VM bytecode or VM C module
```

The VM dialect is not a frontend ML dialect. It is IREE's portable runtime
module dialect. Earlier IREE dialects decide what work exists and how it should
be scheduled. The VM dialect explains how the host-side runtime module will
execute that work, call imported runtime functions, manage references, and
encode a module for deployment.

## Why This Dialect Exists

IREE needs a compact portable representation for runtime modules. A compiled
program may need host code for initialization, parameter loading, dispatch
launches, shape arithmetic, error handling, and calls into runtime modules such
as HAL, IO parameters, or VMVX. The `vm` dialect gives IREE a common target for
that host/runtime part of the program.

The dialect deliberately keeps its value model small. It supports integer and
floating-point scalar operations for offsets, sizes, conditions, and simple
runtime calculations. It also models references through VM ref types, so
ownership and reference cleanup can be reasoned about by compiler passes before
the program is encoded for the runtime.

This is why `vm` is important even if most tensor computation has already been
lowered elsewhere. The VM module is where IREE packages the callable program:
exports visible to users, imports visible to runtime modules, global
initialization, rodata, scalar control flow, ref lifetime cleanup, and bytecode
ordinal assignment.

## When It Matters

You will usually see `vm` late in the IREE compiler pipeline. By the time IR is
in this dialect, the compiler is no longer describing high-level tensor
semantics. It is describing a runtime module that can be loaded and invoked.

The dialect matters when you are debugging:

- final IREE VM bytecode contents;
- generated VM C output;
- runtime import calls into HAL or other IREE modules;
- global initialization and deinitialization;
- reference lifetime cleanup;
- stack/register pressure in VM functions;
- shape or offset arithmetic that remains on the host side;
- why a compiled module exports or imports a particular function.

If earlier dialects answer "what computation should run?", the VM dialect
answers "what runtime module will IREE actually load and execute?"

## When To Use It

Most users should not write VM IR directly. Use `vm` when you are inspecting
IREE compiler output, working on the VM bytecode backend, working on the VM C
backend, debugging runtime imports, or adding conversion support from another
dialect into IREE's runtime module format.

Use it for:

- representing final runtime module functions and globals;
- lowering `func`, `arith`, `math`, `affine`, and IREE `util` constructs into a
  portable runtime form;
- representing runtime imports and exports;
- encoding scalar control flow and scalar arithmetic;
- carrying rodata and module initialization metadata;
- preparing a module for bytecode serialization or C emission.

Do not use it as the main IR for tensor optimizations. At the VM level, tensor
work should generally have become dispatches, runtime calls, buffers, or scalar
metadata operations.

## Core Types And Attributes

The VM type system is intentionally runtime-oriented.

`!vm.ref<T>` represents an `iree_vm_ref_t`-style reference to a runtime object.
The dialect also has predicates for any VM ref and for typed references. These
refs are important because the compiler can track ownership, insert cleanup, and
mark last uses before bytecode emission.

`!vm.buffer` represents a byte buffer. Buffer operations load, store, copy,
fill, hash, compare, clone, allocate, and query length.

`!vm.list<T>` represents a VM list reference. List operations allocate, reserve,
resize, get, set, and query size for primitive and ref element values.

The dialect uses i32 and i64 integers, f32 and f64 floats, and a 32-bit integer
condition convention where zero is false and nonzero is true. It also defines
symbol-reference attributes for globals and functions, ordinal attributes used
by bytecode layout, and `#vm.ordinal_counts` metadata recording counts for
imports, exports, internals, globals, rodata, and mutable data.

## Operation Inventory

The current local IREE source defines 313 concrete `vm` operations. The list
below is complete for this checkout.

```text
vm.abs.f32
vm.abs.f64
vm.abs.i32
vm.abs.i64
vm.add.f32
vm.add.f64
vm.add.i32
vm.add.i64
vm.and.i32
vm.and.i64
vm.assign.ref
vm.atan.f32
vm.atan.f64
vm.atan2.f32
vm.atan2.f64
vm.bitcast.f32.i32
vm.bitcast.f64.i64
vm.bitcast.i32.f32
vm.bitcast.i64.f64
vm.br
vm.br_table
vm.break
vm.buffer.alloc
vm.buffer.clone
vm.buffer.compare
vm.buffer.copy
vm.buffer.fill.f32
vm.buffer.fill.f64
vm.buffer.fill.i16
vm.buffer.fill.i32
vm.buffer.fill.i64
vm.buffer.fill.i8
vm.buffer.hash
vm.buffer.length
vm.buffer.load.f32
vm.buffer.load.f64
vm.buffer.load.i16.s
vm.buffer.load.i16.u
vm.buffer.load.i32
vm.buffer.load.i64
vm.buffer.load.i8.s
vm.buffer.load.i8.u
vm.buffer.store.f32
vm.buffer.store.f64
vm.buffer.store.i16
vm.buffer.store.i32
vm.buffer.store.i64
vm.buffer.store.i8
vm.call
vm.call.variadic
vm.call.variadic.yieldable
vm.call.yieldable
vm.cast.any.ref
vm.cast.f32.si32
vm.cast.f32.si64
vm.cast.f32.ui32
vm.cast.f32.ui64
vm.cast.f64.si64
vm.cast.f64.ui64
vm.cast.ref.any
vm.cast.si32.f32
vm.cast.si64.f32
vm.cast.si64.f64
vm.cast.ui32.f32
vm.cast.ui64.f32
vm.cast.ui64.f64
vm.ceil.f32
vm.ceil.f64
vm.check.eq
vm.check.ne
vm.check.nearly_eq
vm.check.nz
vm.cmp.eq.f32.near
vm.cmp.eq.f32.o
vm.cmp.eq.f32.u
vm.cmp.eq.f64.near
vm.cmp.eq.f64.o
vm.cmp.eq.f64.u
vm.cmp.eq.i32
vm.cmp.eq.i64
vm.cmp.eq.ref
vm.cmp.gt.f32.o
vm.cmp.gt.f32.u
vm.cmp.gt.f64.o
vm.cmp.gt.f64.u
vm.cmp.gt.i32.s
vm.cmp.gt.i32.u
vm.cmp.gt.i64.s
vm.cmp.gt.i64.u
vm.cmp.gte.f32.o
vm.cmp.gte.f32.u
vm.cmp.gte.f64.o
vm.cmp.gte.f64.u
vm.cmp.gte.i32.s
vm.cmp.gte.i32.u
vm.cmp.gte.i64.s
vm.cmp.gte.i64.u
vm.cmp.lt.f32.o
vm.cmp.lt.f32.u
vm.cmp.lt.f64.o
vm.cmp.lt.f64.u
vm.cmp.lt.i32.s
vm.cmp.lt.i32.u
vm.cmp.lt.i64.s
vm.cmp.lt.i64.u
vm.cmp.lte.f32.o
vm.cmp.lte.f32.u
vm.cmp.lte.f64.o
vm.cmp.lte.f64.u
vm.cmp.lte.i32.s
vm.cmp.lte.i32.u
vm.cmp.lte.i64.s
vm.cmp.lte.i64.u
vm.cmp.nan.f32
vm.cmp.nan.f64
vm.cmp.ne.f32.o
vm.cmp.ne.f32.u
vm.cmp.ne.f64.o
vm.cmp.ne.f64.u
vm.cmp.ne.i32
vm.cmp.ne.i64
vm.cmp.ne.ref
vm.cmp.nz.f32.o
vm.cmp.nz.f32.u
vm.cmp.nz.f64.o
vm.cmp.nz.f64.u
vm.cmp.nz.i32
vm.cmp.nz.i64
vm.cmp.nz.ref
vm.cond_br
vm.cond_break
vm.cond_fail
vm.const.f32
vm.const.f32.zero
vm.const.f64
vm.const.f64.zero
vm.const.i32
vm.const.i32.zero
vm.const.i64
vm.const.i64.zero
vm.const.ref.rodata
vm.const.ref.zero
vm.cos.f32
vm.cos.f64
vm.ctlz.i32
vm.ctlz.i64
vm.discard.refs
vm.div.f32
vm.div.f64
vm.div.i32.s
vm.div.i32.u
vm.div.i64.s
vm.div.i64.u
vm.erf.f32
vm.erf.f64
vm.exp.f32
vm.exp.f64
vm.exp2.f32
vm.exp2.f64
vm.expm1.f32
vm.expm1.f64
vm.export
vm.ext.f32.f64
vm.ext.i16.i32.s
vm.ext.i16.i32.u
vm.ext.i16.i64.s
vm.ext.i16.i64.u
vm.ext.i32.i64.s
vm.ext.i32.i64.u
vm.ext.i8.i32.s
vm.ext.i8.i32.u
vm.ext.i8.i64.s
vm.ext.i8.i64.u
vm.fail
vm.floor.f32
vm.floor.f64
vm.fma.f32
vm.fma.f64
vm.fma.i32
vm.fma.i64
vm.func
vm.global.address
vm.global.f32
vm.global.f64
vm.global.i32
vm.global.i64
vm.global.load.f32
vm.global.load.f64
vm.global.load.i32
vm.global.load.i64
vm.global.load.indirect.f32
vm.global.load.indirect.f64
vm.global.load.indirect.i32
vm.global.load.indirect.i64
vm.global.load.indirect.ref
vm.global.load.ref
vm.global.ref
vm.global.store.f32
vm.global.store.f64
vm.global.store.i32
vm.global.store.i64
vm.global.store.indirect.f32
vm.global.store.indirect.f64
vm.global.store.indirect.i32
vm.global.store.indirect.i64
vm.global.store.indirect.ref
vm.global.store.ref
vm.import
vm.import.resolved
vm.initializer
vm.list.alloc
vm.list.get.f32
vm.list.get.f64
vm.list.get.i32
vm.list.get.i64
vm.list.get.ref
vm.list.reserve
vm.list.resize
vm.list.set.f32
vm.list.set.f64
vm.list.set.i32
vm.list.set.i64
vm.list.set.ref
vm.list.size
vm.log.f32
vm.log.f64
vm.log10.f32
vm.log10.f64
vm.log1p.f32
vm.log1p.f64
vm.log2.f32
vm.log2.f64
vm.max.f32
vm.max.f64
vm.max.i32.s
vm.max.i32.u
vm.max.i64.s
vm.max.i64.u
vm.min.f32
vm.min.f64
vm.min.i32.s
vm.min.i32.u
vm.min.i64.s
vm.min.i64.u
vm.module
vm.module_terminator
vm.mul.f32
vm.mul.f64
vm.mul.i32
vm.mul.i64
vm.neg.f32
vm.neg.f64
vm.not.i32
vm.not.i64
vm.optimization_barrier
vm.or.i32
vm.or.i64
vm.pow.f32
vm.pow.f64
vm.print
vm.rem.f32
vm.rem.f64
vm.rem.i32.s
vm.rem.i32.u
vm.rem.i64.s
vm.rem.i64.u
vm.return
vm.rodata
vm.rodata.inline
vm.rodata.table.inline
vm.round.f32
vm.round.f32.even
vm.round.f64
vm.round.f64.even
vm.rsqrt.f32
vm.rsqrt.f64
vm.select.f32
vm.select.f64
vm.select.i32
vm.select.i64
vm.select.ref
vm.shl.i32
vm.shl.i64
vm.shr.i32.s
vm.shr.i32.u
vm.shr.i64.s
vm.shr.i64.u
vm.sin.f32
vm.sin.f64
vm.sqrt.f32
vm.sqrt.f64
vm.sub.f32
vm.sub.f64
vm.sub.i32
vm.sub.i64
vm.switch.f32
vm.switch.f64
vm.switch.i32
vm.switch.i64
vm.switch.ref
vm.tanh.f32
vm.tanh.f64
vm.trace
vm.trunc.f64.f32
vm.trunc.i16.i8
vm.trunc.i32.i16
vm.trunc.i32.i8
vm.trunc.i64.i16
vm.trunc.i64.i32
vm.trunc.i64.i8
vm.xor.i32
vm.xor.i64
vm.yield
```

## How To Read The Operation Groups

Structural operations are the module shell: `vm.module`, `vm.func`,
`vm.import`, `vm.export`, `vm.initializer`, and `vm.module_terminator`. These
define what the runtime module contains, what it exposes, and what external
runtime functions it expects.

Global and rodata operations model module state. `vm.global.*` declares mutable
or reference globals. `vm.global.load.*` and `vm.global.store.*` access them.
The indirect forms access globals through an address value. `vm.rodata`,
`vm.rodata.inline`, and `vm.rodata.table.inline` represent immutable data that
can be hoisted, deduplicated, and serialized into the module.

Reference operations model runtime object ownership. `vm.assign.ref`,
`vm.cast.any.ref`, `vm.cast.ref.any`, `vm.const.ref.zero`,
`vm.const.ref.rodata`, `vm.discard.refs`, and the ref comparison/select/switch
ops make reference values explicit enough for the compiler to analyze last uses
and cleanup obligations.

Buffer and list operations are runtime containers. `vm.buffer.*` is byte-buffer
storage with primitive loads and stores. `vm.list.*` is a VM list abstraction
with get and set variants for i32, i64, f32, f64, and refs.

Scalar integer and floating-point operations cover the arithmetic that remains
in the VM module: constants, add/sub/mul/div/rem, min/max, shifts, bitwise ops,
casts, truncations, extensions, comparisons, selects, switches, and math
functions. They are used for shape math, offset math, runtime decisions, and
small host-side calculations.

Control-flow and call operations define executable behavior. `vm.br`,
`vm.cond_br`, `vm.br_table`, `vm.call`, `vm.call.variadic`,
`vm.call.yieldable`, `vm.return`, `vm.fail`, `vm.cond_fail`, and `vm.yield`
describe ordinary branches, runtime calls, resumable calls, returns, errors,
and fiber yields.

Debug and compiler-hint operations include `vm.trace`, `vm.print`, `vm.break`,
`vm.cond_break`, and `vm.optimization_barrier`. These are useful during
debugging or during optimization staging, but some are removed or stripped
before final output depending on target options.

## Transformations And Conversions

The central lowering pass is `iree-vm-conversion`. It converts from a mix of
MLIR and IREE dialects into `vm`. In this checkout it registers conversions for
IREE `util`, `arith`, `math`, `func`, affine-to-standard patterns, and any used
dialect that implements IREE's VM conversion dialect interface. The conversion
pass also builds VM import declarations, configures index bit width, controls
the f32 and f64 VM extensions, and can optimize for smaller VM stack usage.

The core VM transform passes are:

```text
iree-vm-annotate-functions
iree-vm-conversion
iree-vm-convert-to-yieldable-calls
iree-vm-deduplicate-rodata
iree-vm-drop-empty-module-initializers
iree-vm-drop-optimization-barriers
iree-vm-drop-unused-calls
iree-vm-global-initialization
iree-vm-hoist-inlined-rodata
iree-vm-materialize-ref-discards
iree-vm-ordinal-allocation
iree-vm-reify-rodata-tables
iree-vm-resolve-rodata-loads
iree-vm-sink-defining-ops
```

`iree-vm-transformation-pipeline` is the full pipeline registration for VM
lowering and cleanup. It performs inlining and symbol DCE, verifies and combines
initializers, lowers loops and affine pieces, propagates subranges, runs
`iree-vm-conversion`, reifies and hoists rodata, resolves rodata loads, creates
global initialization functions, canonicalizes, annotates yield and unwind
requirements, converts calls to yieldable forms, optionally sinks definitions
for stack size, drops optimization barriers, and inserts explicit ref discards.

The VM-to-C path uses `iree-convert-vm-to-emitc`. It rewrites VM functions and
ops into EmitC using the calling convention expected by IREE's C module output.
The C target also uses `iree-vm-drop-excluded-exports` to remove exports that
should not be emitted in that target.

The target translations registered by the VM dialect are:

```text
iree-vm-ir-to-bytecode-module
iree-vm-ir-to-c-module
```

The first serializes a `vm.module` to IREE VM bytecode. The second translates a
`vm.module` to a C module through the VM-to-EmitC conversion path.

## What The Dialect Implies

Seeing `vm` means the IR is close to a runtime artifact. Operation choices now
affect bytecode size, VM stack usage, import ABI shape, ref cleanup, and whether
the module can be interpreted, serialized, or emitted as C.

It also means many high-level compiler freedoms are gone. Tensor algebra should
not be rediscovered from VM scalar operations. Device scheduling should mostly
have been decided by earlier IREE dialects. VM passes still optimize, but they
optimize runtime-module concerns: symbol ordinals, rodata layout, call cleanup,
global initialization, dead calls, barriers, and reference lifetimes.

The most important practical implication is that VM IR is both compiler IR and
runtime ABI preparation. A `vm.import` is not just a call placeholder; it is an
ABI contract with a runtime module. A `vm.export` is not just a function name;
it is part of the callable module interface. A `vm.discard.refs` is not just a
cleanup marker; it controls how references are released or transferred at
runtime.

## How To Use It In Practice

When reading VM IR, start at `vm.module`. Look for `vm.export` to see what
outside callers can invoke. Look for `vm.import` to see which runtime modules
the compiled program depends on. Then inspect `vm.func` bodies and separate
scalar bookkeeping from runtime calls.

For performance or size questions, inspect rodata handling, call counts,
globals, and the number of temporary scalar values. Passes such as
`iree-vm-deduplicate-rodata`, `iree-vm-hoist-inlined-rodata`,
`iree-vm-drop-unused-calls`, `iree-vm-sink-defining-ops`, and
`iree-vm-materialize-ref-discards` are specifically about making the final
runtime module smaller or easier to execute.

For correctness questions, focus on imports, exports, global initialization,
failure paths, yieldable calls, and ref ownership. These are the areas where a
VM module most directly interacts with the runtime.

For backend work, remember that VM has two main output paths in this source
tree: bytecode and C. Bytecode relies on opcode encoding and ordinal allocation.
C output relies on conversion to EmitC and the generated module structure. The
same VM IR must therefore be precise enough to support both a compact
serialized runtime format and a static C representation.

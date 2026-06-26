# IREE `vmvx` Dialect

## Beginner Summary

The IREE `vmvx` dialect represents VM-based vector and linear-algebra
extensions for IREE's portable reference target.

VMVX stands for Virtual Machine-based Vector eXtensions. It is designed as a
small virtual ISA extension to the IREE VM. Instead of lowering everything to
scalar bytecode loops, IREE can lower selected operations to VMVX calls that are
implemented by a runtime module.

For a beginner, the key idea is:

```text
tensor / linalg / vector / HAL executable code
  -> VMVX-compatible buffer code
  -> vmvx dialect operations
  -> VM imports such as vmvx.add.2d.f32 or vmvx.copy.2d.x32
  -> runtime VMVX module functions
```

VMVX is not a frontend ML dialect. It is a target-side dialect used when IREE is
building executable code for the portable `vmvx` backend.

## Why This Dialect Exists

IREE has a VM bytecode runtime. Plain scalar VM bytecode is portable, but
running all tensor code as scalar loops would be too expensive. VMVX provides a
small set of runtime-backed operations for common strided buffer work.

The dialect exists to bridge compiler IR and runtime module exports.

It lets IREE represent:

- strided unary elementwise operations;
- strided binary elementwise operations;
- strided buffer copies;
- simple 2D fills;
- late-bound buffer descriptors;
- raw interface binding buffers for VMVX calling conventions.

This keeps the VM target portable while avoiding a fully scalar expansion for
every operation.

## When It Matters

You will see `vmvx` late in the IREE compilation flow, after high-level tensor
and Linalg operations have been selected for a VMVX executable target.

A simplified flow looks like this:

```text
stablehlo / linalg / tensor / vector
  -> IREE flow / stream / HAL executable
  -> VMVX codegen configuration and lowering
  -> bufferization and loop/vector cleanup
  -> iree-vmvx-materialize-constants
  -> iree-vmvx-conversion
  -> vmvx operations
  -> VM import calls
  -> VMVX runtime module
```

VMVX is useful for testing, portability, embedded deployments, and reference
execution paths. It is often easier to bring up than target-specific native code
generation because the runtime module implements the core operations.

## When To Use It

Use `vmvx` when working on IREE's VMVX target backend or debugging a VMVX
compiled module.

Use it for:

- portable execution through IREE's VM runtime;
- reference-style backend testing;
- lowering strided 2D elementwise and copy operations;
- connecting HAL executable bindings to VM runtime buffers;
- replacing selected Linalg or vector code with VMVX microkernel calls;
- investigating how a VMVX `.vmfb` module calls runtime exports.

Do not use VMVX as a source dialect. A frontend should emit StableHLO, TOSA,
Linalg, or another high-level input representation.

Do not use VMVX as a general CPU optimization dialect. It is tied to IREE's VM
runtime import model and portable microkernel module.

## Core Concepts

### VM Runtime Imports

VMVX operations lower to calls into a VM module named `vmvx`.

The compiler embeds declarations from `vmvx.imports.mlir`. The runtime side has
matching exports in `runtime/src/iree/modules/vmvx/exports.inl` and
implementations in the VMVX runtime module.

Examples of runtime export names include:

- `add.2d.f32`;
- `add.2d.i32`;
- `copy.2d.x32`;
- `fill.2d.x32`;
- `abs.2d.f32`;
- `mmt4d`;
- `pack` and `unpack`.

The dialect op is therefore not the final executable instruction. It is a
compiler-level form that can be converted into the right VM import call.

### Buffer-Oriented ABI

VMVX ops do not traffic in shaped tensor values. They operate on buffers,
offsets, strides, sizes, scalar values, and element type attributes.

For example, `vmvx.binary` carries:

- an opcode such as `add` or `mul`;
- left-hand-side buffer, offset, and strides;
- right-hand-side buffer, offset, and strides;
- output buffer, offset, and strides;
- sizes;
- an element type.

This is much closer to a runtime ABI than to a high-level tensor IR.

### Late Buffer Descriptors

`vmvx.get_buffer_descriptor` delays the decision of where a buffer really comes
from. It queries a base buffer, offset, sizes, and strides from a memref-like
source. Canonicalization can move it through view-like operations, and a later
pass resolves it to concrete sources.

This keeps buffer plumbing flexible while earlier lowering stages are still
rewriting views and subspans.

### HAL Interface Bindings

VMVX executable functions use a VMVX-specific calling convention. HAL interface
bindings, constants, and workgroup values need to become VMVX-compatible values.

`vmvx.get_raw_interface_binding_buffer` exists for cases where lowering needs
the raw backing buffer for a binding rather than a memref subspan.

## Types And Attributes

The VMVX dialect mostly uses aliases and constraints rather than many custom
types.

| Type or attribute | Meaning |
| --- | --- |
| `!util.buffer` | The runtime buffer type used by VMVX buffer operands. |
| `index` | Used for VMVX offsets, sizes, and strides. |
| `i8`, `i16`, `i32`, `i64`, `f32`, `f64` | Supported element types in the inspected TableGen constraints. |
| element type attribute | Records the element type used by `vmvx.binary`, `vmvx.copy`, and `vmvx.unary`. |
| device and host size aliases | Size-like aliases over `index` used by the VMVX ABI. |

## Operation Inventory

The VMVX dialect currently defines six operations.

| Operation | Purpose |
| --- | --- |
| `vmvx.get_buffer_descriptor` | Late-bind a memref-like source to a base buffer, offset, sizes, and strides. |
| `vmvx.get_raw_interface_binding_buffer` | Get the raw VMVX buffer associated with a HAL interface binding. |
| `vmvx.binary` | Run a strided 2D-style binary elementwise runtime operation over two input buffers and one output buffer. |
| `vmvx.copy` | Copy strided data from one buffer to another. |
| `vmvx.fill2d` | Fill a 2D tile in an output buffer with a scalar. |
| `vmvx.unary` | Run a strided unary elementwise runtime operation over one input buffer and one output buffer. |

## Transformations And Conversions

VMVX support is split between dialect transforms, codegen passes, conversion
patterns, and registered pipelines.

### Dialect Passes

| Pass | What it does |
| --- | --- |
| `iree-vmvx-conversion` | Converts from multiple dialects into VMVX-compatible IR. |
| `iree-vmvx-materialize-constants` | Materializes executable constant global values for VMVX lowering. |
| `iree-vmvx-resolve-buffer-descriptors` | Resolves remaining `vmvx.get_buffer_descriptor` operations. |

The conversion pass uses pattern families such as:

- `populateHALToVMVXPatterns`;
- `populateStandardToVMVXPatterns`;
- generic structural conversion patterns;
- TOSA rescale to arithmetic conversion patterns where needed.

### Codegen Passes

| Pass | What it does |
| --- | --- |
| `iree-vmvx-assign-constant-ordinals` | Assigns executable constant ordinals across VMVX variants. |
| `iree-vmvx-select-lowering-strategy` | Selects a VMVX lowering pipeline attribute for an executable variant. |
| `iree-vmvx-link-executables` | Links VMVX HAL executables in the top-level module. |
| `iree-vmvx-lower-executable-target` | Lowers an executable target using the selected pipeline attribute. |
| `iree-vmvx-lower-linalg-microkernels` | Lowers Linalg operations to the VMVX microkernel library. |

### VM Conversion

`populateVMVXToVMPatterns` converts VMVX operations into calls to VM imports.
For example:

- `vmvx.binary` selects an import name based on opcode, rank, and element type;
- `vmvx.copy` selects a typed copy import;
- `vmvx.fill2d` selects a typed 2D fill import;
- `vmvx.unary` selects an import name based on opcode and element type.

This is the point where `vmvx` IR becomes ordinary VM-level calls.

### Registered Pipelines

| Pipeline | Purpose |
| --- | --- |
| `iree-vmvx-configuration-pipeline` | Runs the full VMVX dialect configuration pipeline. |
| `iree-vmvx-transformation-pipeline` | Runs the full VMVX dialect transformation pipeline. |
| `iree-codegen-vmvx-configuration-pipeline` | Runs VMVX codegen configuration. |
| `iree-codegen-vmvx-lowering-pipeline` | Runs VMVX codegen lowering. |
| `iree-codegen-vmvx-linking-pipeline` | Links VMVX HAL executables and assigns constants. |

## How To Read VMVX IR

Start by reading buffer triples: buffer, offset, and strides.

Then read the size operands. VMVX operations are explicit about the region of
memory they operate over. If the operation is `vmvx.binary` or `vmvx.unary`,
also read the opcode and element type attribute. Those decide which runtime
import the conversion will target.

Example shape:

```mlir
vmvx.binary op(add : f32)
  lhs(%lhs_buffer offset %lhs_offset strides [%lhs_s0, %lhs_s1] : !util.buffer)
  rhs(%rhs_buffer offset %rhs_offset strides [%rhs_s0, %rhs_s1] : !util.buffer)
  out(%out_buffer offset %out_offset strides [%out_s0, %out_s1] : !util.buffer)
  sizes(%m, %n)
```

This says "call a strided 2D add implementation over these buffers." It does
not describe a high-level tensor add anymore.

## What It Implies

Using VMVX implies that IREE has already selected a portable VM target path.

It also implies that:

- tensors have been lowered toward buffers;
- memory access is ABI-like and explicit;
- operation names must correspond to runtime imports;
- adding a new VMVX operation requires compiler, import, export, and runtime
  changes;
- the result is portable but not the same as native CPU or GPU codegen.

VMVX is therefore best understood as IREE's portable runtime-backed execution
layer for selected vector and linear algebra work.

## Source Files Inspected

This chapter was written from the local IREE source:

- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/IR/VMVXBase.td`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/IR/VMVXOps.td`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/README.md`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/vmvx.imports.mlir`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/Transforms/Passes.td`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/Transforms/Passes.cpp`
- `IREE/iree/compiler/src/iree/compiler/Codegen/VMVX/Passes.td`
- `IREE/iree/compiler/src/iree/compiler/Codegen/VMVX/Passes.cpp`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/Conversion/HALToVMVX/ConvertHALToVMVX.cpp`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/Conversion/StandardToVMVX/ConvertStandardToVMVX.cpp`
- `IREE/iree/compiler/src/iree/compiler/Dialect/VMVX/Conversion/VMVXToVM/ConvertVMVXToVM.cpp`
- `IREE/iree/runtime/src/iree/modules/vmvx/exports.inl`

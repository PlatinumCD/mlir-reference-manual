# ONNX-MLIR `zhigh` Dialect

## Beginner Summary

The ONNX-MLIR `zhigh` dialect is the high-level dialect for IBM Z NNPA
acceleration. NNPA is exposed through ONNX-MLIR as a separate accelerator path:
eligible ONNX operations are rewritten into `zhigh` operations, optimized while
they are still tensor-like, then lowered to the lower-level `zlow` dialect and
eventually to calls into the NNPA runtime support.

For a beginner, the useful mental model is:

```text
ONNX graph
  -> rewrite-onnx-for-zhigh
  -> convert-onnx-to-zhigh
  -> zhigh tensors with NNPA layouts
  -> zhigh layout and stick/unstick cleanup
  -> ZHigh-to-ZLow conversion through NNPA lowering hooks
  -> zlow memref/runtime-oriented operations
```

ZHigh is not a generic neural-network dialect. It exists to represent the part
of an ONNX model that is suitable for NNPA execution, with the data layouts,
data conversions, and operation forms expected by that accelerator path.

## Why This Dialect Exists

ONNX describes portable tensor operations. NNPA execution requires more
specific information: which tensors are in the accelerator's internal layout,
which values have been converted to DLFLOAT16, which operations are legal for a
given NNPA level, and where layout conversion should happen.

The `zhigh` dialect gives the compiler a high-level place to express those
decisions. It is still close to ONNX operations such as `MatMul`, `Conv2D`,
`Relu`, `Softmax`, `LSTM`, and `GRU`, but it attaches NNPA-specific tensor
encoding and layout semantics. This lets ONNX-MLIR optimize placement before
committing to lower-level memref shapes and runtime-call details.

The dialect is also a filter. If an ONNX operation is not suitable for NNPA
because of shape, layout, datatype, architecture level, or performance
heuristics, it can remain ONNX or be reconstructed back to ONNX so the normal
CPU path handles it.

## When It Matters

You will see `zhigh` when compiling with the NNPA accelerator enabled. It
matters when:

- an ONNX operation has been selected for NNPA execution;
- tensors are being converted into or out of NNPA layouts;
- DLFLOAT16 conversion is visible in the IR;
- `Stick` and `Unstick` operations surround NNPA work;
- lightweight operations may be moved back to ONNX to avoid expensive
  stick/unstick traffic;
- ZHigh operations are about to lower into `zlow` operations;
- NNPA profiling or signatures are requested at the ZHigh stage.

Use ZHigh when debugging NNPA placement and high-level accelerator lowering. Do
not use it as a general replacement for ONNX or Krnl. If the hardware path is
not enabled, or the operation is not supported by NNPA, the ordinary ONNX to
Krnl path is the relevant one.

## Core Attribute And Tensor Model

The central attribute is:

```text
#zhigh.layout<{dataLayout = "...", quantizedType = "..."}>
```

The attribute is implemented as `ZTensorEncodingAttr`. Its `dataLayout`
describes the logical layout of the pre-stickified tensor. The supported layout
enumerators include:

```text
UNDEFINED, 1D, 2D, 2DS, 3D, 3DS, 4D, 4DS,
NCHW, NHWC, HWCK, FICO, ZRH, BFICO, BZRH
```

Its optional `quantizedType` can describe quantized tensor roles such as
DLFLOAT16, INT8, or WEIGHTS.

A tensor with this encoding is a ztensor. The tensor element type is commonly
`f16`, representing the accelerator-oriented DLFLOAT16 storage path. The type's
shape remains the logical tensor shape that the compiler reasons about, while
the encoding tells later lowering how to allocate and call the lower-level NNPA
operations.

## Complete Operation Inventory

The current `zhigh` dialect defines 40 operations:

```text
zhigh.Add
zhigh.AvgPool2D
zhigh.BatchNorm
zhigh.Conv2D
zhigh.DLF16ToF32
zhigh.Div
zhigh.Exp
zhigh.ExtendedLayoutTransform
zhigh.F32ToDLF16
zhigh.FixGRUY
zhigh.FixGRUYh
zhigh.GRU
zhigh.Gelu
zhigh.InvSqrt
zhigh.LSTM
zhigh.LeakyRelu
zhigh.Log
zhigh.MatMul
zhigh.Max
zhigh.MaxPool2D
zhigh.MeanReduce2d
zhigh.Min
zhigh.Mul
zhigh.QuantizedMatMul
zhigh.QuantizedStick
zhigh.ReduceMax
zhigh.ReduceMin
zhigh.Relu
zhigh.Reshape
zhigh.Sigmoid
zhigh.Softmax
zhigh.Sqrt
zhigh.Stick
zhigh.StickForGRU
zhigh.StickForLSTM
zhigh.StickifiedConstant
zhigh.StickifiedConstantOfShape
zhigh.Sub
zhigh.Tanh
zhigh.Unstick
```

## Operation Groups

`zhigh.Stick` converts a CPU-format tensor into an NNPA ztensor layout.
`zhigh.Unstick` converts a ztensor back to ordinary tensor layout. These
operations are the main boundary markers between normal ONNX-style tensor data
and NNPA-formatted data.

`zhigh.F32ToDLF16` and `zhigh.DLF16ToF32` represent data conversion between
ordinary `f32` tensors and the DLFLOAT16-oriented path. `zhigh.QuantizedStick`
does the analogous job for quantized inputs.

Elementwise binary operations include `zhigh.Add`, `zhigh.Sub`, `zhigh.Mul`,
`zhigh.Div`, `zhigh.Min`, and `zhigh.Max`. These generally require matching
layouts on operands and results.

Unary math and activation operations include `zhigh.Log`, `zhigh.Exp`,
`zhigh.InvSqrt`, `zhigh.Relu`, `zhigh.Gelu`, `zhigh.Sqrt`, `zhigh.Tanh`, and
`zhigh.Sigmoid`. `zhigh.LeakyRelu` carries its alpha behavior through a
dedicated operation.

Neural-network layer operations include `zhigh.MatMul`,
`zhigh.QuantizedMatMul`, `zhigh.Conv2D`, `zhigh.BatchNorm`, `zhigh.MaxPool2D`,
`zhigh.AvgPool2D`, `zhigh.Softmax`, `zhigh.ReduceMax`, `zhigh.ReduceMin`, and
`zhigh.MeanReduce2d`.

Sequence-model operations include `zhigh.LSTM`, `zhigh.GRU`,
`zhigh.StickForLSTM`, `zhigh.StickForGRU`, `zhigh.FixGRUY`, and
`zhigh.FixGRUYh`. The stick-for-RNN operations prepare packed gate layouts such
as FICO and ZRH that the NNPA path expects.

Constant and layout helper operations include `zhigh.StickifiedConstant`,
`zhigh.StickifiedConstantOfShape`, `zhigh.Reshape`, and
`zhigh.ExtendedLayoutTransform`.

## Transformations And Conversions

The main ZHigh-related passes and lowering hooks are:

```text
device-placement
nnpa-quant-ops-selection
generate-config-file
rewrite-onnx-for-zhigh
convert-onnx-to-zhigh
zhigh-layout-prop
zhigh-decompose-stick-unstick
zhigh-recompose-to-stick-unstick
convert-zhigh-to-onnx
constprop-zhigh
zhigh-scrub-disposable
fusion-op-stick-unstick
convert-onnx-to-krnl
```

`device-placement` decides which ONNX operations should try the NNPA path.
`nnpa-quant-ops-selection` configures quantized operation selection. If
requested, `generate-config-file` writes the selected NNPA configuration.

`rewrite-onnx-for-zhigh` prepares ONNX graphs for NNPA lowering. The compiler
runs it repeatedly because it can create shape-related ONNX operations, and
those may become constants after shape simplification.

`convert-onnx-to-zhigh` is the main conversion from ONNX to ZHigh. It uses
combined patterns first, such as replacing ONNX `MatMul` plus `Add` with a
single ZHigh operation when legal, then applies single-op patterns. Dynamic
legality checks decide whether each ONNX op is actually suitable for the NNPA
path.

`zhigh-layout-prop` moves layout conversions across operations where possible,
so unnecessary `Unstick` plus `Stick` pairs can disappear or move away from
critical NNPA subgraphs.

`zhigh-decompose-stick-unstick` breaks stick/unstick operations into layout
transform and data conversion pieces. After canonicalization,
`zhigh-recompose-to-stick-unstick` combines remaining pieces back into the
compact ZHigh operations. This two-phase strategy exposes optimization
opportunities without losing the final high-level form.

`convert-zhigh-to-onnx` reconstructs ONNX operations from some ZHigh patterns.
This is intentionally a fallback optimization: if a lightweight operation would
require costly `Stick` and `Unstick` traffic, CPU execution can be better than
NNPA execution.

`constprop-zhigh` performs ZHigh constant propagation, including constant
stickification on big-endian hosts. `zhigh-scrub-disposable` replaces
disposable element attributes with dense attributes. `fusion-op-stick-unstick`
fuses suitable ONNX operations with surrounding stick/unstick structure.

There is no separate command-line pass named `convert-zhigh-to-zlow` in the
same style as `convert-onnx-to-zhigh`. Instead, the NNPA accelerator registers
ZHigh-to-ZLow rewrite patterns through its ONNX-to-Krnl conversion hook.
During `convert-onnx-to-krnl`, ZHigh operations are lowered to `zlow`
operations using `populateZHighToZLowConversionPattern`.

## What The Dialect Implies

Seeing `zhigh` means the compiler believes this part of the model is currently
on the NNPA path. The operation is still high-level enough to resemble ONNX,
but its tensor types and attributes now carry accelerator layout meaning.

Seeing `zhigh.Stick` or `zhigh.Unstick` means there is a boundary between CPU
format and NNPA format. Too many such boundaries can erase accelerator gains,
which is why the compiler has layout propagation, decomposition/recomposition,
fusion, and ZHigh-to-ONNX fallback passes.

Seeing a ztensor type with `#zhigh.layout` means later lowering must allocate
and pass data in an NNPA-compatible layout. It is not just an ordinary ranked
tensor with a decorative attribute.

## How To Use It In Practice

When reading ZHigh IR, start with the stick/unstick boundaries. They show where
data enters and leaves the NNPA formatted world. Then inspect the chain of
ZHigh compute operations between those boundaries. Long chains are usually
better than isolated single operations because they amortize layout conversion.

Next, inspect layouts. Elementwise operations usually require matching operand
and result layouts, while pooling, convolution, matmul, and recurrent
operations have more specialized layout expectations.

Finally, remember that ZHigh is not the end. If compilation continues, ZHigh
will lower to ZLow during the ONNX-to-Krnl phase, and ZLow will later lower to
LLVM/runtime calls through the NNPA accelerator hooks.

## Source Pointers

The core dialect definition is in
`src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td`. Per-operation implementations
live under `src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps`. ONNX-to-ZHigh
lowering lives in `src/Accelerators/NNPA/Conversion/ONNXToZHigh`. ZHigh-to-ZLow
rewrite patterns live in `src/Accelerators/NNPA/Conversion/ZHighToZLow`.
ZHigh-specific transformations live in
`src/Accelerators/NNPA/Transform/ZHigh`.

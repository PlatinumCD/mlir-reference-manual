# ONNX-MLIR `onnx` Dialect

## Beginner Summary

The ONNX-MLIR `onnx` dialect is the compiler's MLIR representation of an ONNX
model.

ONNX is a model interchange format. It describes neural-network and machine
learning computations with operators such as `Conv`, `MatMul`, `Relu`,
`Softmax`, `Gather`, `Reshape`, `Loop`, and `If`. ONNX-MLIR imports that model
format into MLIR as operations prefixed with `onnx.`.

For a beginner, the most important idea is that the ONNX dialect is a frontend
dialect. It preserves the model-level meaning of the original ONNX graph before
the compiler lowers it to loop nests, accelerator operations, StableHLO, TOSA,
Linalg, LLVM, or runtime calls.

An ONNX model node like `Conv` becomes an MLIR operation such as `onnx.Conv`.
The operation still means convolution in the ONNX sense: it has ONNX attributes,
ONNX tensor semantics, ONNX broadcasting rules, and ONNX shape rules. Later
passes progressively rewrite or lower that high-level meaning into executable
code.

## Why This Dialect Exists

ONNX is broad. It includes tensor math, neural-network layers, control flow,
sequences, optionals, quantization, recurrent networks, classical machine
learning operators, and many versioned compatibility forms. Lowering all of
that directly to LLVM would make the compiler hard to understand and hard to
debug.

The `onnx` dialect exists so ONNX-MLIR can preserve the source model's intent
while still using MLIR infrastructure.

It gives the compiler a place to:

- represent imported ONNX graphs faithfully;
- run ONNX-aware shape inference;
- fold constant ONNX subgraphs;
- decompose complex ONNX ops into simpler ONNX ops;
- recompose beneficial patterns such as larger matrix or convolution forms;
- preserve node names and model signatures for debugging;
- choose CPU, StableHLO, TOSA, Linalg, or accelerator lowering paths;
- keep optional inputs and ONNX-specific sequence types visible until lowering.

This dialect is needed because ONNX operators are not just generic tensor
operations. For example, `onnx.Resize`, `onnx.Pad`, `onnx.Conv`,
`onnx.BatchNormalization`, and `onnx.LSTM` all carry ONNX-specific attributes
and edge cases. The ONNX dialect is where those details remain explicit.

## When It Matters

You see the ONNX dialect near the beginning of an ONNX-MLIR compilation.

A typical CPU-oriented flow is:

```text
.onnx model
  -> onnx dialect
  -> ONNX-to-ONNX transforms
  -> shape inference and constant propagation
  -> convert-onnx-to-krnl
  -> krnl / affine / scf / vector / memref
  -> llvm
```

Alternative flows can lower selected ONNX operations to StableHLO, TOSA,
Linalg, or NNPA accelerator dialects:

```text
onnx
  -> convert-onnx-to-stablehlo
  -> stablehlo

onnx
  -> convert-onnx-to-tosa
  -> tosa

onnx
  -> convert-onnx-to-linalg
  -> linalg

onnx
  -> rewrite-onnx-for-zhigh
  -> convert-onnx-to-zhigh
  -> zhigh / zlow
```

It matters most when you are debugging model import, shape inference, operator
coverage, constant propagation, ONNX canonicalization, or the boundary between
frontend model semantics and lower-level code generation.

## When To Use It

Use the `onnx` dialect when you want to inspect or transform a model at ONNX
semantic level.

Use it for:

- imported `.onnx` or `.onnxtext` models;
- model-level graph rewriting;
- shape inference and dimension analysis;
- optional-input handling;
- decomposition of complex ops into simpler ONNX ops;
- preserving ONNX node names for diagnostics;
- selecting which ops go to CPU or accelerator paths;
- tests that compare against ONNX operator semantics.

Do not use it as a general tensor compiler IR after model semantics are no
longer needed. Once the compiler has enough shape and type information, it
usually lowers to `krnl`, `stablehlo`, `tosa`, `linalg`, or accelerator dialects.

Do not expect every `onnx.*` op to lower through every backend path. The normal
CPU path has broad ONNX coverage through `convert-onnx-to-krnl`; StableHLO,
TOSA, Linalg, and NNPA conversion paths intentionally cover selected subsets.

## Core Concepts

### ONNX Semantics In MLIR

Most generated operations are direct MLIR representations of ONNX operators.
The operation name keeps the ONNX spelling, such as `onnx.Add`,
`onnx.ReduceMean`, or `onnx.SequenceInsert`.

Attributes, operands, and results follow the imported ONNX operator schema.
Many ops implement shape inference interfaces so ONNX-MLIR can refine unranked
or dynamic tensor types into ranked tensor types when enough information is
available.

### Optional Inputs

ONNX operators often have optional inputs or outputs. ONNX-MLIR represents a
missing optional value with `onnx.NoValue`, whose result has MLIR `NoneType`.

This is more explicit than simply deleting the operand. It lets patterns match
the same operand positions that the ONNX schema uses, even when one of those
positions is absent.

### Shape Inference

Shape inference is central to ONNX-MLIR. Many lowerings require ranked tensors,
and many optimizations become possible only when dimensions are known.

The `shape-inference` pass asks ONNX ops and their shape helpers to refine
result types. Other passes such as `simplify-shape-related-ops-onnx`,
`onnx-dim-analysis`, and `remove-same-onnx-dim` make dynamic shape information
more usable.

### ONNX-to-ONNX Rewriting

The ONNX dialect is not just imported and immediately lowered. It is also
rewritten while remaining ONNX.

`decompose-onnx` turns complex or inconvenient ops into simpler ONNX patterns.
`recompose-onnx` recognizes patterns that should become a larger op again.
`constprop-onnx` folds constant ONNX computations. `onnx-op-transform` and
`onnx-hybrid-transform` run repeated combinations of these steps until a
threshold or fixed point is reached.

### Versioned Ops

Some operations have versioned spellings such as `onnx.ClipV6`,
`onnx.ResizeV11`, `onnx.PadV18`, or `onnx.ReduceMeanV13`. These exist because
ONNX operator semantics have changed across opsets.

Versioned operations let the compiler preserve the behavior of an older ONNX
model without pretending it means exactly the same thing as the newest schema.

## Types And Attributes

The ONNX dialect primarily uses MLIR tensor types. It also defines ONNX-specific
types and attributes:

| Type or attribute | Meaning |
| --- | --- |
| `!onnx.String` | ONNX string scalar element type. |
| `!onnx.Seq<...>` | ONNX sequence type; usually a sequence of tensors. |
| `!onnx.Opt<...>` | ONNX optional type around a tensor or sequence. |
| `NoneType` via `onnx.NoValue` | Sentinel for missing optional ONNX values. |
| `#onnx.encoding<...>` | Tensor layout encoding, such as standard layout, `NCHWxC`, or `KCNMxCyK`. |

The tensor encoding attribute is used when ONNX-MLIR represents custom data
layouts, especially for convolution-oriented transformations. `onnx.LayoutTransform`
can move tensors between layouts.

## Operation Inventory

This checkout defines 259 `onnx.*` operations: 243 generated from ONNX operator
definitions and 16 ONNX-MLIR helper operations.

### Generated ONNX Ops

These are the generated ONNX-schema operations in
`src/Dialect/ONNX/ONNXOps.td.inc`.

```text
onnx.Abs
onnx.Acos
onnx.Acosh
onnx.Adagrad
onnx.Adam
onnx.Add
onnx.And
onnx.ArgMax
onnx.ArgMin
onnx.ArrayFeatureExtractor
onnx.Asin
onnx.Asinh
onnx.Atan
onnx.Atanh
onnx.Attention
onnx.AveragePool
onnx.BatchNormalization
onnx.Bernoulli
onnx.Binarizer
onnx.BitShift
onnx.BitwiseAnd
onnx.BitwiseNot
onnx.BitwiseOr
onnx.BitwiseXor
onnx.BlackmanWindow
onnx.Cast
onnx.CastLike
onnx.CastMap
onnx.CategoryMapper
onnx.Ceil
onnx.Celu
onnx.CenterCropPad
onnx.Clip
onnx.ClipV11
onnx.ClipV12
onnx.ClipV6
onnx.Col2Im
onnx.Compress
onnx.Concat
onnx.ConcatFromSequence
onnx.Constant
onnx.ConstantOfShape
onnx.Conv
onnx.ConvInteger
onnx.ConvTranspose
onnx.Cos
onnx.Cosh
onnx.CumSum
onnx.DFT
onnx.DFTV17
onnx.DeformConv
onnx.DepthToSpace
onnx.DequantizeLinear
onnx.Det
onnx.DictVectorizer
onnx.Div
onnx.Dropout
onnx.DynamicQuantizeLinear
onnx.Einsum
onnx.Elu
onnx.Equal
onnx.Erf
onnx.Exp
onnx.Expand
onnx.EyeLike
onnx.FeatureVectorizer
onnx.Flatten
onnx.Floor
onnx.GRU
onnx.Gather
onnx.GatherElements
onnx.GatherND
onnx.Gelu
onnx.Gemm
onnx.GlobalAveragePool
onnx.GlobalLpPool
onnx.GlobalMaxPool
onnx.Gradient
onnx.Greater
onnx.GreaterOrEqual
onnx.GridSample
onnx.GridSampleV16
onnx.GroupNormalization
onnx.GroupNormalizationV18
onnx.HammingWindow
onnx.HannWindow
onnx.HardSigmoid
onnx.HardSwish
onnx.Hardmax
onnx.Identity
onnx.If
onnx.Imputer
onnx.InstanceNormalization
onnx.IsInf
onnx.IsNaN
onnx.LRN
onnx.LSTM
onnx.LabelEncoder
onnx.LayerNormalization
onnx.LeakyRelu
onnx.Less
onnx.LessOrEqual
onnx.LinearClassifier
onnx.LinearRegressor
onnx.Log
onnx.LogSoftmax
onnx.Loop
onnx.LpNormalization
onnx.LpPool
onnx.MatMul
onnx.MatMulInteger
onnx.Max
onnx.MaxPool
onnx.MaxRoiPool
onnx.MaxUnpool
onnx.Mean
onnx.MeanVarianceNormalization
onnx.MelWeightMatrix
onnx.Min
onnx.Mish
onnx.Mod
onnx.Momentum
onnx.Mul
onnx.Multinomial
onnx.Neg
onnx.NegativeLogLikelihoodLoss
onnx.NonMaxSuppression
onnx.NonZero
onnx.Normalizer
onnx.Not
onnx.OneHot
onnx.OneHotEncoder
onnx.Optional
onnx.OptionalGetElement
onnx.OptionalHasElement
onnx.Or
onnx.PRelu
onnx.Pad
onnx.PadV11
onnx.PadV13
onnx.PadV18
onnx.PadV2
onnx.Pow
onnx.QLinearConv
onnx.QLinearMatMul
onnx.QuantizeLinear
onnx.RNN
onnx.RandomNormal
onnx.RandomNormalLike
onnx.RandomUniform
onnx.RandomUniformLike
onnx.Range
onnx.Reciprocal
onnx.ReduceL1
onnx.ReduceL1V13
onnx.ReduceL2
onnx.ReduceL2V13
onnx.ReduceLogSum
onnx.ReduceLogSumExp
onnx.ReduceLogSumExpV13
onnx.ReduceLogSumV13
onnx.ReduceMax
onnx.ReduceMaxV13
onnx.ReduceMaxV18
onnx.ReduceMean
onnx.ReduceMeanV13
onnx.ReduceMin
onnx.ReduceMinV13
onnx.ReduceMinV18
onnx.ReduceProd
onnx.ReduceProdV13
onnx.ReduceSum
onnx.ReduceSumSquare
onnx.ReduceSumSquareV13
onnx.ReduceSumV11
onnx.Relu
onnx.Reshape
onnx.Resize
onnx.ResizeV10
onnx.ResizeV11
onnx.ResizeV13
onnx.ResizeV18
onnx.ReverseSequence
onnx.RoiAlign
onnx.Round
onnx.STFT
onnx.SVMClassifier
onnx.SVMRegressor
onnx.Scaler
onnx.Scan
onnx.Scatter
onnx.ScatterElements
onnx.ScatterND
onnx.Selu
onnx.SequenceAt
onnx.SequenceConstruct
onnx.SequenceEmpty
onnx.SequenceErase
onnx.SequenceInsert
onnx.SequenceLength
onnx.SequenceMap
onnx.Shape
onnx.Shrink
onnx.Sigmoid
onnx.Sign
onnx.Sin
onnx.Sinh
onnx.Size
onnx.Slice
onnx.Softmax
onnx.SoftmaxCrossEntropyLoss
onnx.SoftmaxV11
onnx.Softplus
onnx.Softsign
onnx.SpaceToDepth
onnx.Split
onnx.SplitToSequence
onnx.SplitV11
onnx.SplitV13
onnx.Sqrt
onnx.Squeeze
onnx.SqueezeV11
onnx.StringNormalizer
onnx.Sub
onnx.Sum
onnx.Tan
onnx.Tanh
onnx.TfIdfVectorizer
onnx.ThresholdedRelu
onnx.Tile
onnx.TopK
onnx.Transpose
onnx.TreeEnsembleClassifier
onnx.TreeEnsembleRegressor
onnx.Trilu
onnx.Unique
onnx.Unsqueeze
onnx.UnsqueezeV11
onnx.Upsample
onnx.UpsampleV7
onnx.Where
onnx.Xor
onnx.ZipMap
```

### ONNX-MLIR Helper Ops

These are defined in `src/Dialect/ONNX/AdditionalONNXOps.td`. Some assist
lowering, some represent missing optional values, and some encode compiler
patterns that are not direct ONNX model nodes.

```text
onnx.BatchNormalizationInferenceMode
onnx.ConcatShapeTranspose
onnx.Custom
onnx.Dim
onnx.DimGroup
onnx.EntryPoint
onnx.Im2Col
onnx.LayoutTransform
onnx.MaxPoolSingleOut
onnx.NoValue
onnx.PrintSignature
onnx.RMSLayerNormalization
onnx.Return
onnx.ShapeTransform
onnx.UpsampleAndPad
onnx.Yield
```

## Operation Families

The full list is large, but most operations fall into a few learnable groups.

| Family | Examples | What to learn first |
| --- | --- | --- |
| Elementwise math | `onnx.Add`, `onnx.Mul`, `onnx.Exp`, `onnx.Tanh`, `onnx.Relu` | Broadcasting, element types, constant folding. |
| Tensor shape and movement | `onnx.Reshape`, `onnx.Transpose`, `onnx.Gather`, `onnx.Slice`, `onnx.Concat` | Shape inference and dynamic dimensions. |
| Reductions | `onnx.ReduceMean`, `onnx.ReduceSum`, `onnx.ArgMax`, `onnx.TopK` | Axis handling, keepdims, opset differences. |
| Neural-network layers | `onnx.Conv`, `onnx.AveragePool`, `onnx.BatchNormalization`, `onnx.LayerNormalization` | Layouts, padding, kernels, and lowering strategy. |
| Matrix and attention | `onnx.MatMul`, `onnx.Gemm`, `onnx.Einsum`, `onnx.Attention`, `onnx.Softmax` | Dense linear algebra and transformer-style graphs. |
| Recurrent and control flow | `onnx.RNN`, `onnx.GRU`, `onnx.LSTM`, `onnx.If`, `onnx.Loop`, `onnx.Scan` | Regions/subgraphs and sequence dimensions. |
| Quantization | `onnx.QuantizeLinear`, `onnx.DequantizeLinear`, `onnx.QLinearConv`, `onnx.QLinearMatMul` | Scale, zero point, integer tensor semantics. |
| Sequences and optionals | `onnx.SequenceInsert`, `onnx.SequenceAt`, `onnx.Optional`, `onnx.NoValue` | Non-tensor values in an ONNX graph. |
| Classical ML | `onnx.TreeEnsembleClassifier`, `onnx.SVMRegressor`, `onnx.LabelEncoder`, `onnx.Normalizer` | ONNX is broader than neural networks. |
| Compiler helpers | `onnx.EntryPoint`, `onnx.Return`, `onnx.Dim`, `onnx.LayoutTransform` | ONNX-MLIR infrastructure around the model. |

## Transformations And Conversions

ONNX-MLIR has a rich set of ONNX-level transformations before final lowering.

### ONNX-to-ONNX Transform Passes

| Pass | What it does |
| --- | --- |
| `shape-inference` | Refines ONNX result types using ONNX shape helpers. |
| `constprop-onnx` | Folds constant ONNX computations. |
| `decompose-onnx` | Rewrites complex ONNX ops into simpler ONNX subgraphs. |
| `recompose-onnx` | Recognizes useful ONNX subgraphs and replaces them with larger ONNX ops. |
| `conv-opt-onnx` | Performs convolution-oriented ONNX optimizations, including optional SIMD data layout handling. |
| `onnx-op-transform` | Iteratively runs ONNX decomposition, recomposition, shape inference, convolution optimization, and constant propagation. |
| `onnx-hybrid-transform` | Configurable transform that can combine shape inference, canonicalization, constant propagation, decomposition, convolution-to-matmul, and recomposition. |
| `simplify-shape-related-ops-onnx` | Simplifies ONNX shape-related operations and propagates useful shape information. |
| `remove-same-onnx-dim` | Removes duplicated `onnx.Dim` operations that refer to the same dimension. |
| `onnx-dim-analysis` | Analyzes dynamic dimensions and relationships among ONNX tensor dimensions. |
| `replace-op-with-its-operand` | Replaces selected one-result ONNX ops with one of their operands, controlled by node-name regex options. |
| `set-onnx-node-name` | Adds `onnx_node_name` attributes when missing. |
| `onnx-cse-with-node-name` | Runs CSE while preserving ONNX node-name attributes. |
| `append-decoding-strategy` | Appends decoding logic such as greedy, top-k, or top-p for decoder models. |
| `standard-func-return` | Replaces `onnx.Return` with standard `func.return` before lowerings that expect Func dialect returns. |
| `scrub-disposable` | Cleans disposable ONNX constant data structures. |
| `instrument-onnx-runtime-signature` | Instruments ONNX runtime signature information. |
| `onnx-pre-krnl-verify` | Verifies ONNX IR before lowering to Krnl. |
| `test-onnx-reify-result-shapes` | Test-only pass for `reifyResultShapes` support. |

### Conversion Passes

| Pass | Target | What it does |
| --- | --- | --- |
| `convert-onnx-to-krnl` | `krnl` | Main CPU-oriented lowering path from ONNX frontend ops to ONNX-MLIR's loop-building Krnl dialect. |
| `convert-onnx-to-linalg` | `linalg` | Lowers selected ONNX ops to Linalg; has options for choosing which ops use this path. |
| `convert-onnx-to-stablehlo` | `stablehlo` | Lowers supported ONNX math, NN, RNN, and tensor ops to StableHLO. |
| `convert-onnx-to-tosa` | `tosa` | Lowers a supported subset of ONNX ops to TOSA. |
| `rewrite-onnx-for-zhigh` | ONNX/NNPA preparation | Rewrites ONNX patterns into forms suitable for IBM z/Architecture NNPA ZHigh lowering. |
| `convert-onnx-to-zhigh` | `zhigh` | Converts supported ONNX ops to NNPA ZHigh ops. |
| `convert-zhigh-to-onnx` | `onnx` | Rewrites some ZHigh stick/unstick patterns back to ONNX operations. |

Related NNPA passes such as `device-placement`, `nnpa-quant-ops-selection`, and
`generate-config-file` are not ONNX dialect passes in the narrow sense, but they
shape which ONNX subgraphs are placed on the accelerator path.

## How To Read ONNX IR

An imported model often looks like a `func.func` containing `onnx.*` operations:

```mlir
func.func @main_graph(%arg0: tensor<1x3x224x224xf32>,
                      %arg1: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> {
  %none = onnx.NoValue
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {
    auto_pad = "NOTSET",
    dilations = [1, 1],
    group = 1 : si64,
    kernel_shape = [7, 7],
    pads = [3, 3, 3, 3],
    strides = [2, 2]
  } : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, none)
    -> tensor<1x64x112x112xf32>
  onnx.Return %0 : tensor<1x64x112x112xf32>
}
```

The important things to notice are:

- ONNX optional inputs may appear as `onnx.NoValue`;
- ONNX attributes remain attached to the op;
- result types may become more precise after `shape-inference`;
- `onnx.Return` is still an ONNX dialect operation until
  `standard-func-return` runs;
- the op is still high level, not yet a loop nest.

## What It Implies

Using the ONNX dialect means the compiler is still preserving ONNX model
semantics. That is good for debugging and model-level optimization, but it also
means many details are still unresolved: dynamic shapes may remain symbolic,
optional inputs may be present, layout choices may not yet be final, and backend
selection may not yet be complete.

Once an ONNX op lowers to Krnl, StableHLO, TOSA, Linalg, or ZHigh, the compiler
has made a stronger commitment about how to implement it. Before that point,
ONNX-to-ONNX passes can still reshape the graph at the semantic level.

For a beginner, the central model is:

```text
onnx = meaning-preserving model IR
krnl/linalg/stablehlo/tosa/zhigh = implementation-oriented IR
llvm/runtime = executable code boundary
```

## Beginner Checklist

When reading ONNX-MLIR ONNX IR, ask:

- Which ONNX ops are still present?
- Are the tensors ranked or still unranked?
- Are dimensions static, dynamic, or tied by `onnx.Dim` / `onnx.DimGroup`?
- Are optional operands represented with `onnx.NoValue`?
- Has `shape-inference` run?
- Has `constprop-onnx` folded constants?
- Has `decompose-onnx` expanded complex ops?
- Has `recompose-onnx` rebuilt larger efficient patterns?
- Is the model headed to `convert-onnx-to-krnl`, `convert-onnx-to-stablehlo`, `convert-onnx-to-tosa`, `convert-onnx-to-linalg`, or `convert-onnx-to-zhigh`?
- Are node names preserved for debugging?

The ONNX dialect is the place to learn the model as the model. Lowering comes
later.

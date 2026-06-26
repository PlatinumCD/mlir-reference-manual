# ONNX-MLIR `zlow` Dialect

The `zlow` dialect is ONNX-MLIR's low level dialect for IBM Z Neural Network Processing Assist, usually abbreviated NNPA. In the ONNX-MLIR stack, a model starts in ONNX dialect, eligible operations are moved into the NNPA-oriented `zhigh` dialect, and then `zhigh` is lowered to `zlow`. After that, `zlow` is lowered through LLVM-facing code that calls the zDNN runtime.

For a beginner, the important idea is that `zlow` is not a general tensor dialect. It is a hardware and runtime boundary dialect. It records the operations, layouts, temporary buffers, shape descriptors, and data conversions needed to issue zDNN work. A `zlow` program is close enough to the runtime that many operations use explicit memrefs, side effects, output buffers, shape buffers, and layout strings instead of the value-style tensor semantics seen earlier in a compiler pipeline.

## When This Dialect Is Important

Use `zlow` when studying how ONNX-MLIR targets NNPA, how zTensor layout is represented, or how a high level neural network operation becomes an explicit runtime call. It is important after graph-level legality and profitability choices have already happened. By the time code is in `zlow`, the compiler has committed to using NNPA-compatible operations for the selected regions.

You usually do not hand-write `zlow` as an application developer. You inspect it when debugging NNPA lowering, layout conversions, quantization, missing runtime calls, or memory traffic around stick and unstick operations. It is also useful for learning how an MLIR dialect can model a concrete accelerator ABI: inputs are memrefs, outputs are explicit, shape operands are passed as `memref<...xi64>`, and attributes encode zDNN choices such as layout, activation behavior, direction, and quantized type.

The dialect implies that ordinary tensor meaning has already been specialized. For example, a `memref<f16>` in `zlow` may be a container for DLFloat16 data because MLIR does not natively model that type in this source tree. Likewise, a zTensor is not just an array with a shape; it is a memory object in a zDNN layout such as `1D`, `2D`, `2DS`, `3D`, `3DS`, `4D`, or `NCHW`.

## Operation Inventory

The `zlow` dialect defines these operations:

```text
zlow.add
zlow.sub
zlow.mul
zlow.div
zlow.log
zlow.exp
zlow.invsqrt
zlow.min
zlow.max
zlow.leakyrelu
zlow.relu
zlow.gelu
zlow.tanh
zlow.sigmoid
zlow.softmax
zlow.sqrt
zlow.reducemax
zlow.reducemin
zlow.matmul
zlow.quantizedMatmul
zlow.lstm
zlow.gru
zlow.stick
zlow.stickForLSTM
zlow.stickForGRU
zlow.quantizedStick
zlow.unstick
zlow.conv2d
zlow.avgpool2d
zlow.maxpool2d
zlow.meanreduce2d
zlow.batchnorm
zlow.dummy
zlow.dlf16_to_f32
zlow.f32_to_dlf16
zlow.vec_dlf16_to_f32
zlow.vec_f32_to_dlf16
zlow.reshape
```

## Data Model

Most `zlow` operations are memref based and implement memory effects. They read input memrefs, read shape or work area buffers when needed, and write explicit output memrefs. This is a major difference from many tensor dialect operations, which often return a new SSA tensor value. In `zlow`, the output object is commonly an operand because the generated runtime call needs a concrete destination buffer.

The central data object is a zTensor memref. The source defines a `ZMemRef` constraint for memrefs whose element type is the dialect's DLFloat16 container, represented with MLIR `f16`. Quantized paths use `ZQMemRef`, which can hold either that DLF16 container or `i8`. Scalar floating control values use rank zero memrefs, such as `memref<f32>`, for values like reciprocal scale and offset operands.

Shape operands are explicit. For example, matrix multiply uses a shape buffer describing the matrix dimensions and stacked form; recurrent operations use a shape buffer for sequence, batch, direction, and hidden dimensions; convolution and pooling use shape buffers with kernel, stride, padding, and image dimensions. The dialect is therefore teaching a key lowering lesson: as the compiler approaches a C runtime ABI, symbolic tensor information becomes concrete memory and integer data.

## Elementwise And Activation Ops

`zlow.add`, `zlow.sub`, `zlow.mul`, `zlow.div`, `zlow.min`, and `zlow.max` are binary zTensor operations. They consume two zTensor inputs, a shape buffer, an output zTensor, and a layout attribute. They map directly to zDNN elementwise APIs such as add, subtract, multiply, divide, minimum, and maximum.

`zlow.log`, `zlow.exp`, `zlow.invsqrt`, `zlow.relu`, `zlow.gelu`, `zlow.tanh`, `zlow.sigmoid`, and `zlow.sqrt` are unary zTensor operations. They carry one zTensor input, shape information, an output zTensor, and a layout. `zlow.leakyrelu` is in the same family but also carries an `alpha` attribute, with the dialect definition giving it a default of `0.01`.

The lesson from these operations is that even simple math is no longer represented as abstract scalar formulas. Each op says, "call this NNPA/zDNN operation over this zTensor layout and write this output buffer."

## Softmax And Reductions

`zlow.softmax` lowers a softmax over a zTensor input. It takes an explicit work area, a shape buffer, an output buffer, and an activation function attribute. The supported activation choices in this layer include the normal softmax behavior and a log-softmax form.

`zlow.reducemax` and `zlow.reducemin` represent zDNN reduction operations. They also use a work area, shape information, output memory, and layout metadata. Their presence shows why `zlow` is more ABI-like than algebra-like: temporary scratch space becomes part of the IR because the runtime call requires it.

## Matrix And Recurrent Ops

`zlow.matmul` represents NNPA matrix multiplication. It has zTensor inputs for left and right operands, a bias input, a shape buffer, and an output. Its attributes describe broadcasting and layout variants, including `is_bcast1`, `is_bcast23`, `is_stacked`, `transposeA`, and `transposeB`. These attributes preserve lowering decisions that are relevant to the zDNN call rather than to generic linear algebra.

`zlow.quantizedMatmul` is the quantized matrix multiply form. It has quantized zTensor inputs and explicit scale and offset operands for input, weights, bias, and output. Its attributes include quantized type strings for the operands, whether the operation is stacked or broadcast, whether the bias was precomputed, whether clipping is disabled, and whether the output should be dequantized. This operation matters when learning how quantization crosses the boundary from high level graph meaning into runtime-level metadata.

`zlow.lstm` and `zlow.gru` are recurrent neural network operations. They take input state, initial state, weights, biases, work areas, shape buffers, and output buffers. `zlow.lstm` writes both hidden and cell outputs, while `zlow.gru` writes the hidden output. Their attributes include direction, whether all time steps should be returned, and whether the operation is a previous layer form. These ops are good examples of how one MLIR operation can represent a large runtime kernel rather than a small primitive instruction.

## Layout Movement Ops

`zlow.stick` converts ordinary CPU-side tensor memory into zTensor layout. `zlow.unstick` converts zTensor layout back into ordinary CPU memory. These are central to NNPA lowering because the accelerator consumes and produces data in specific zDNN layouts, while the surrounding compiler and runtime may use ordinary dense memory.

`zlow.stickForLSTM` and `zlow.stickForGRU` are specialized stick operations for recurrent gate data. LSTM uses the four gate order F, I, C, O, while GRU uses the three gate order Z, R, H. These operations show that layout conversion can be semantic: it is not always a plain reshape or transpose, because the target runtime expects gates to be packed in a specific way.

`zlow.quantizedStick` sticks either `i8` or `f32` input data into a quantized zTensor representation. It takes reciprocal scale and offset operands and a `q_type` attribute. The verifier accepts quantized type names for DLF16, int8, and weights, and checks that the output element type and scale or offset element types are consistent with the declared quantized form.

`zlow.reshape` is a zTensor reshape. It carries input and output layouts, and the rewrite pass can lower it to `memref.reinterpret_cast` when it is only changing the view of the underlying zTensor memory. The important beginner point is that `zlow.reshape` is not a generic tensor reshape. It is a zTensor memory reinterpretation when layouts and shapes make that legal.

## Neural Network Kernel Ops

`zlow.conv2d` represents a 2D convolution. It consumes input, kernel, and bias zTensors plus a shape buffer and output zTensor. It also records kernel shape, strides, padding type, and activation function. That compact operation stands for a concrete zDNN convolution call.

`zlow.avgpool2d` and `zlow.maxpool2d` represent 2D pooling. Both carry input, shape, output, kernel shape, strides, and padding type. `zlow.meanreduce2d` represents a 2D mean reduction over a zTensor input with shape and output operands. `zlow.batchnorm` models zDNN batch normalization with input, scale-like and bias-like operands, shape information, and output.

These operations are important because they show where ONNX graph concepts become accelerator library entry points. The compiler is no longer trying to preserve the full ONNX operator surface; it is selecting the subset and form that NNPA can execute.

## Helper And Conversion Ops

`zlow.dummy` is an identity-like helper operation. It exists to make intermediate IR valid or to work around cases where another transformation cannot handle multiple dereferences from the same producer. The dialect includes a declarative rewrite rule, `RemoveDummyOpPattern`, that replaces `zlow.dummy(X)` with `X`, so this operation should disappear after canonicalization.

`zlow.dlf16_to_f32` and `zlow.f32_to_dlf16` convert scalar values between the DLF16 container representation and `f32`. `zlow.vec_dlf16_to_f32` and `zlow.vec_f32_to_dlf16` perform vectorized forms of the same conversions. These are used by data movement and stick expansion code because the surrounding CPU loops may need to load, convert, and store values explicitly. The combine rules also include `DLF16ConversionOpPattern`, which removes an unnecessary `dlf16_to_f32` followed by `f32_to_dlf16` pair.

## Transformations And Conversions

The main producer-side conversion is `populateZHighToZLowConversionPattern`. It converts NNPA `zhigh` operations into concrete `zlow` operations. The registered patterns include lowering for stick, unstick, recurrent gate sticking, stickified constants, scalar and vector data conversions, binary and unary elementwise operations, reshape, reduce min and max, softmax, mean reduce, leaky relu, matmul, LSTM, GRU, batch normalization, convolution, pooling, quantized stick, quantized matmul, and extended layout transforms.

`zlow-rewrite` is the main cleanup and simplification pass for ZLow IR. Its patterns include `ReshapeToReinterpretCastPattern`, `StickRemovalPattern`, `UnstickRemovalPattern`, `UnstickStickRemovalPattern`, `StickViewUnstickRemovalPattern`, and `UnstickLoadStoreStickRemovalPattern`. In beginner terms, this pass tries to remove layout traffic that does not need to survive. It can replace a zTensor reshape with a memref reinterpret cast, delete unused stick or unstick operations, remove an unstick followed by a stick with compatible layouts, and forward affine load/store traffic directly between stickified memrefs.

`zlow-stick-expansion` is the stick and unstick optimization pass. Depending on compiler options, it expands `zlow.stick` and `zlow.unstick` into explicit loops and conversion operations, or at least adjusts allocation sizes and alignment for runtime requirements. This is where high level layout movement can become ordinary memref, affine, and conversion code.

`zlow-dummyop-for-multideref` inserts `zlow.dummy` in repeated-operand situations so later memory normalization can proceed. Because `zlow.dummy` canonicalizes away, this pass is a temporary structural aid rather than a semantic operation that should remain visible in final code.

The consumer-side conversion is `populateZLowToLLVMConversionPattern`. It lowers `zlow` operations to LLVM dialect constructs and zDNN runtime calls. Elementwise operations are mapped to zDNN APIs such as `ZDNN_ADD`, `ZDNN_SUB`, `ZDNN_MUL`, `ZDNN_DIV`, `ZDNN_MIN`, `ZDNN_MAX`, `ZDNN_LOG`, `ZDNN_EXP`, `ZDNN_INVSQRT`, `ZDNN_RELU`, `ZDNN_GELU`, `ZDNN_TANH`, `ZDNN_SIGMOID`, and `ZDNN_SQRT`. Other lowering classes handle stick, unstick, softmax, leaky relu, reductions, matmul, quantized matmul, LSTM, GRU, convolution, pooling, mean reduction, batch normalization, and DLF16 conversions.

In the ONNX-MLIR NNPA pipeline, the accelerator registers `zlow::ZLowDialect` as legal during ONNX to Krnl lowering, populates ZHigh-to-ZLow patterns there, runs ZLow rewrite and stick optimization passes, and finally lowers remaining Krnl and all ZLow operations to LLVM. Instrumentation can also target the `ZLow` stage and match `zlow.*` operations, which makes this dialect a practical debugging checkpoint.

## How To Read ZLow IR

When reading `zlow`, start by finding the layout movement. A cluster of `zlow.stick` operations means CPU or compiler-produced data is entering zTensor layout. A `zlow.unstick` means data is leaving NNPA layout. If a model has many stick and unstick pairs, inspect whether `zlow-rewrite` can remove them or whether the program is crossing between accelerator and CPU regions too often.

Next, identify the kernel operations. Operations such as `zlow.conv2d`, `zlow.matmul`, `zlow.lstm`, `zlow.gru`, `zlow.softmax`, and `zlow.batchnorm` are the meaningful accelerator calls. Their shape and attribute operands tell you exactly which runtime variant is being requested.

Finally, check helper operations. `zlow.dummy`, DLF16 conversion operations, and vector conversion operations are usually artifacts of lowering, normalization, or explicit data movement. They matter for correctness, but they are not the neural network algorithm itself.

The broad implication is that `zlow` teaches the last MLIR step before accelerator runtime execution. It is where tensor meaning, memory layout, scratch space, quantization metadata, and runtime ABI details meet.

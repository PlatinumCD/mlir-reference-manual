# StableHLO `vhlo` Dialect

The StableHLO `vhlo` dialect is Versioned StableHLO. It exists to make StableHLO portable across time.

For a beginner, the most important point is that VHLO is not the dialect most users write directly. Normal producers write the current `stablehlo` dialect. When a StableHLO module needs to be serialized, stored, shipped to another compiler, or targeted at an older compatibility version, StableHLO can be converted into `vhlo`. VHLO gives every operation, type, and attribute an explicit versioned spelling such as `vhlo.add_v1`, `vhlo.scatter_v2`, or `!vhlo.tensor_v1`.

This makes compatibility a compiler problem instead of a guess. A producer can convert StableHLO to VHLO, ask VHLO to target a specific StableHLO version, and fail early if the program uses features that cannot be represented at that target version.

## When VHLO Is Important

VHLO is important when StableHLO crosses a version boundary. That happens when a model is serialized by one tool and consumed by another tool that may not be built from the same StableHLO revision. It also matters for test cases that verify bytecode stability, compatibility windows, and older StableHLO consumers.

Use VHLO when you need to answer questions like:

- Can this StableHLO program be consumed by a compiler that only supports an older StableHLO version?
- Which exact version of an operation is being serialized?
- Did a change to an operation require a new compatibility spelling?
- Is a type, attribute, or location legal for a requested StableHLO target version?
- Can the serialized payload round-trip back into the current StableHLO dialect?

You usually do not use VHLO to express new machine learning semantics. If you want to model tensor computation, use `stablehlo`. If you want to preserve the meaning of a StableHLO program across releases, use the VHLO conversion path.

## Why It Is Needed

StableHLO changes over time. New operations are added, attributes gain fields, types appear, and some operation semantics become more precise. If a serialized file only used the latest textual operation names, an older consumer might not know whether it can read the file safely.

VHLO solves this by making the compatibility surface add-only. Existing VHLO definitions are not changed in place. If an operation changes in a way that affects compatibility, the dialect adds a new versioned operation instead of mutating the old one. For example, `vhlo.scatter_v1` and `vhlo.scatter_v2` can both exist, with conversion patterns deciding whether a program can move between them.

The implication is simple: VHLO turns "what did this operation mean in that release?" into explicit IR. The version suffix is part of the program, not a comment.

## How To Read VHLO

Read a VHLO operation name in three parts:

- `vhlo` says this is the versioned compatibility dialect.
- The middle name, such as `add`, `scatter`, or `dot_general`, is the StableHLO concept.
- The suffix, such as `_v1` or `_v2`, is the VHLO compatibility version of that concept.

Most VHLO operations are intentionally shallow. The dialect avoids the rich StableHLO verifier and type constraints because the goal is stable serialization, not a second semantic implementation of StableHLO. Many definitions use broad placeholders such as `VHLO_AnyType`, `VHLO_AnyAttr`, and `VHLO_AnyRegion`. The exact compatibility rules live in version interfaces and conversion patterns.

This also explains why VHLO may look more verbose than StableHLO. StableHLO can say `stablehlo.exponential` with a current result-accuracy attribute. VHLO may say `vhlo.exponential_v1` or `vhlo.exponential_v2` depending on which compatibility version is needed.

## Type Inventory

The local VHLO dialect defines versioned types for scalar elements, shaped containers, quantization, functions, buffers, tokens, and compatibility helpers.

Boolean and numeric scalar types:

- `!vhlo.bool_v1`
- `!vhlo.index_v1`
- `!vhlo.i2_v1`, `!vhlo.i4_v1`, `!vhlo.i8_v1`, `!vhlo.i16_v1`, `!vhlo.i32_v1`, `!vhlo.i64_v1`
- `!vhlo.ui2_v1`, `!vhlo.ui4_v1`, `!vhlo.ui8_v1`, `!vhlo.ui16_v1`, `!vhlo.ui32_v1`, `!vhlo.ui64_v1`
- `!vhlo.bf16_v1`, `!vhlo.f16_v1`, `!vhlo.f32_v1`, `!vhlo.f64_v1`
- `!vhlo.f4E2M1FN_v1`, `!vhlo.f6E2M3FN_v1`, `!vhlo.f6E3M2FN_v1`
- `!vhlo.f8E3M4_v1`, `!vhlo.f8E4M3_v1`, `!vhlo.f8E4M3FN_v1`, `!vhlo.f8E5M2_v1`
- `!vhlo.f8E4M3FNUZ_v1`, `!vhlo.f8E4M3B11FNUZ_v1`, `!vhlo.f8E5M2FNUZ_v1`, `!vhlo.f8E8M0FNU_v1`
- `!vhlo.tf31_v1`

Compound and special types:

- `!vhlo.complex_v1` wraps a versioned floating-point element type.
- `!vhlo.tensor_v1` is the ranked tensor container.
- `!vhlo.unranked_tensor_v1` is the unranked tensor container.
- `!vhlo.buffer_v1` is a ranked buffer container.
- `!vhlo.tuple_v1` groups multiple VHLO types.
- `!vhlo.func_v1` represents function signatures.
- `!vhlo.none_v1` represents absence of a value where compatibility needs a type.
- `!vhlo.token_v1` models ordering tokens used by side-effecting StableHLO operations.
- `!vhlo.future_v1` represents asynchronous future values.
- `!vhlo.witness_v1` represents a witness value used by shape-style constraints.
- `!vhlo.quant_v1` and `!vhlo.quant_per_axis_v1` represent uniform quantized types.

The beginner rule is that VHLO types mirror the StableHLO type system but pin each piece to a compatibility version.

## Attribute Inventory

VHLO has primitive container attributes:

- `#vhlo.array_v1`
- `#vhlo.bool_v1`
- `#vhlo.dict_v1`
- `#vhlo.float_v1`
- `#vhlo.integer_v1`
- `#vhlo.string_v1`
- `#vhlo.tensor_v1`
- `#vhlo.type_v1`
- `#vhlo.type_extensions_v1`

It also has StableHLO-specific structural attributes:

- `#vhlo.output_operand_alias_v1`
- `#vhlo.result_accuracy_v1`
- `#vhlo.sub_axis_info_v1`
- `#vhlo.axis_ref_v1`
- `#vhlo.replica_group_mesh_axes_v1`
- `#vhlo.mesh_axis_v1`
- `#vhlo.mesh_v1`

And it carries enum-style attributes that mirror StableHLO enumerations:

- `#vhlo.comparison_direction_v1`
- `#vhlo.comparison_type_v1`
- `#vhlo.custom_call_api_version_v1`
- `#vhlo.fft_type_v1`
- `#vhlo.precision_v1`
- `#vhlo.rng_algorithm_v1`
- `#vhlo.rng_distribution_v1`
- `#vhlo.transpose_v1`
- `#vhlo.result_accuracy_mode_v1`

These attributes matter because compatibility is not only about operations. A program can fail to target an older version because an attribute or nested type did not exist yet.

## Operation Inventory

The local VHLO dialect defines 142 operation names. They are easier to understand by role, but the exact names matter because each name is part of the serialized compatibility contract.

### Elementwise Arithmetic, Logic, and Math

These operations cover scalar-style tensor computation: `vhlo.abs_v1`, `vhlo.add_v1`, `vhlo.and_v1`, `vhlo.atan2_v1`, `vhlo.cbrt_v1`, `vhlo.cbrt_v2`, `vhlo.ceil_v1`, `vhlo.clamp_v1`, `vhlo.compare_v1`, `vhlo.complex_v1`, `vhlo.convert_v1`, `vhlo.cosine_v1`, `vhlo.cosine_v2`, `vhlo.count_leading_zeros_v1`, `vhlo.divide_v1`, `vhlo.exponential_v1`, `vhlo.exponential_v2`, `vhlo.exponential_minus_one_v1`, `vhlo.exponential_minus_one_v2`, `vhlo.floor_v1`, `vhlo.imag_v1`, `vhlo.is_finite_v1`, `vhlo.log_v1`, `vhlo.log_v2`, `vhlo.log_plus_one_v1`, `vhlo.log_plus_one_v2`, `vhlo.logistic_v1`, `vhlo.logistic_v2`, `vhlo.maximum_v1`, `vhlo.minimum_v1`, `vhlo.multiply_v1`, `vhlo.negate_v1`, `vhlo.not_v1`, `vhlo.or_v1`, `vhlo.popcnt_v1`, `vhlo.power_v1`, `vhlo.real_v1`, `vhlo.remainder_v1`, `vhlo.round_nearest_afz_v1`, `vhlo.round_nearest_even_v1`, `vhlo.rsqrt_v1`, `vhlo.rsqrt_v2`, `vhlo.shift_left_v1`, `vhlo.shift_right_arithmetic_v1`, `vhlo.shift_right_logical_v1`, `vhlo.sign_v1`, `vhlo.sine_v1`, `vhlo.sine_v2`, `vhlo.sqrt_v1`, `vhlo.sqrt_v2`, `vhlo.subtract_v1`, `vhlo.tan_v1`, `vhlo.tan_v2`, `vhlo.tanh_v1`, `vhlo.tanh_v2`, and `vhlo.xor_v1`.

The repeated versions are the compatibility signal. For example, a newer result-accuracy field can make a math operation need a `_v2` spelling even though the user-level operation still feels like the same StableHLO op.

### Tensor Shape, Movement, and Construction

These operations build constants, reshape tensors, move dimensions, or slice/update tensor contents: `vhlo.bitcast_convert_v1`, `vhlo.broadcast_v1`, `vhlo.broadcast_in_dim_v1`, `vhlo.concatenate_v1`, `vhlo.constant_v1`, `vhlo.dynamic_broadcast_in_dim_v1`, `vhlo.dynamic_iota_v1`, `vhlo.dynamic_pad_v1`, `vhlo.dynamic_reshape_v1`, `vhlo.dynamic_slice_v1`, `vhlo.dynamic_update_slice_v1`, `vhlo.get_dimension_size_v1`, `vhlo.get_tuple_element_v1`, `vhlo.iota_v1`, `vhlo.pad_v1`, `vhlo.real_dynamic_slice_v1`, `vhlo.reshape_v1`, `vhlo.reverse_v1`, `vhlo.set_dimension_size_v1`, `vhlo.slice_v1`, `vhlo.transpose_v1`, and `vhlo.tuple_v1`.

These are common in serialized model graphs because framework frontends often express shape manipulation explicitly.

### Reductions, Windows, Sorting, and Indexing

These operations describe reductions, windowed reductions, gather/scatter, and selection patterns: `vhlo.gather_v1`, `vhlo.gather_v2`, `vhlo.dynamic_gather_v1`, `vhlo.dynamic_gather_v2`, `vhlo.map_v1`, `vhlo.reduce_v1`, `vhlo.reduce_precision_v1`, `vhlo.reduce_window_v1`, `vhlo.scatter_v1`, `vhlo.scatter_v2`, `vhlo.select_v1`, `vhlo.select_and_scatter_v1`, `vhlo.sort_v1`, and `vhlo.torch_index_select_v1`.

Gather and scatter have multiple versions because their dimension-number attributes evolved. When lowering to an older target version, this is exactly the kind of operation that may need a rewrite or may fail if new fields cannot be represented safely.

### Linear Algebra, Convolution, FFT, Random, and Quantization

These operations represent larger numerical building blocks: `vhlo.batch_norm_grad_v1`, `vhlo.batch_norm_inference_v1`, `vhlo.batch_norm_training_v1`, `vhlo.cholesky_v1`, `vhlo.convolution_v1`, `vhlo.dot_v1`, `vhlo.dot_general_v1`, `vhlo.dot_general_v2`, `vhlo.dynamic_conv_v1`, `vhlo.dynamic_conv_v2`, `vhlo.einsum_v1`, `vhlo.fft_v1`, `vhlo.rng_v1`, `vhlo.rng_bit_generator_v1`, `vhlo.triangular_solve_v1`, `vhlo.unary_einsum_v1`, `vhlo.uniform_dequantize_v1`, and `vhlo.uniform_quantize_v1`.

These are the operations most associated with machine learning computation. VHLO does not optimize them directly; it preserves their StableHLO meaning across serialization boundaries.

### Collectives, Communication, Tokens, and Async

These operations represent distributed execution, communication, or ordered side effects: `vhlo.after_all_v1`, `vhlo.all_gather_v1`, `vhlo.all_gather_v2`, `vhlo.all_reduce_v1`, `vhlo.all_reduce_v2`, `vhlo.all_to_all_v1`, `vhlo.all_to_all_v2`, `vhlo.async_done_v1`, `vhlo.async_start_v1`, `vhlo.collective_broadcast_v1`, `vhlo.collective_permute_v1`, `vhlo.create_token_v1`, `vhlo.cross-replica-sum_v1`, `vhlo.infeed_v1`, `vhlo.outfeed_v1`, `vhlo.recv_v1`, `vhlo.recv_v2`, `vhlo.reduce_scatter_v1`, `vhlo.send_v1`, and `vhlo.send_v2`.

Communication operations are sensitive to attributes such as channel handles, replica groups, and mesh axes. VHLO conversion has special handling for these because a small attribute change can affect distributed semantics.

### Control Flow, Functions, Calls, and Extension Points

These operations describe program structure and extensibility: `vhlo.call_v1`, `vhlo.case_v1`, `vhlo.composite_v1`, `vhlo.composite_v2`, `vhlo.custom_call_v1`, `vhlo.func_v1`, `vhlo.if_v1`, `vhlo.optimization_barrier_v1`, `vhlo.return_v1`, and `vhlo.while_v1`.

`vhlo.custom_call_v1` is an escape hatch for backend- or framework-defined behavior. `vhlo.composite_v1` and `vhlo.composite_v2` preserve higher-level named composites and their decomposition information. `vhlo.func_v1`, `vhlo.call_v1`, and `vhlo.return_v1` are VHLO mappings for MLIR function structure.

### Program and Device Identity

The remaining identity-style operations are `vhlo.partition_id_v1` and `vhlo.replica_id_v1`. They expose distributed execution identity values in the same compatibility-aware form as the rest of StableHLO.

## Transformations and Conversions

VHLO is mostly used through three passes.

`stablehlo-legalize-to-vhlo` converts current StableHLO IR into the latest matching VHLO spellings. It maps StableHLO operations through `MapStablehloToVhlo.h`, converts StableHLO types and attributes into VHLO equivalents, and inserts defaults for attributes that must be explicit in serialized form. The pass also has an `allow-other-dialects` option for serialization situations that must carry non-StableHLO dialects with casts.

`vhlo-to-version` converts VHLO IR to a requested target version. It checks each operation, type, attribute, and relevant location against its version range. If a rewrite exists, it applies it. If a feature is too new and cannot be represented, the pass fails. Current manual patterns include conversions for `vhlo.scatter_v1` and `vhlo.scatter_v2`, `vhlo.all_reduce_v1` and `vhlo.all_reduce_v2`, and `vhlo.composite_v1` and `vhlo.composite_v2`. Generated patterns cover many simple version moves.

`vhlo-legalize-to-stablehlo` converts compatible VHLO back into current StableHLO. This is the consumer side of the pipeline. A compiler backend generally wants to optimize the current StableHLO dialect, not VHLO itself, so this pass removes the serialization layer after compatibility has been established.

The related `stablehlo-compatibility-expander` pass is not a VHLO-only pass, but it is important in practice. It can decompose newer StableHLO features into older StableHLO constructs before version targeting. For example, a newer operation may be represented in terms of older operations if the user is willing to trade away some high-level information for wider compatibility.

A common flow is:

```text
stablehlo module
  -> stablehlo-legalize-to-vhlo
  -> vhlo-to-version='target=1.0.0'
  -> serialize

deserialize
  -> vhlo-to-version
  -> vhlo-legalize-to-stablehlo
  -> backend compilation
```

## Special Conversion Cases

Most operations convert mechanically, but some pieces need special handling because their syntax or attribute structure differs from the generic StableHLO representation.

The StableHLO-to-VHLO path handles special attributes for collectives, convolution dimension numbers, dot dimension numbers and algorithms, gather dimension numbers, scatter dimension numbers, send and receive channel handles, custom-call API versions and called computations, composite metadata, function callees, dense array attributes, and default attributes.

The versioning pass also recursively checks nested attributes and types. A ranked tensor can fail because its element type is too new. A tensor attribute can fail because its type is too new. A result-accuracy attribute can fail because its mode is too new. This recursive checking is why compatibility needs versioned attributes and types, not only versioned operations.

## What VHLO Implies

Seeing VHLO in IR implies that the program is in a serialization or compatibility phase. It is usually not the phase where high-level optimization should be happening.

Seeing an operation with multiple versions implies that StableHLO evolved around that feature. For example, `vhlo.all_reduce_v1` and `vhlo.all_reduce_v2` represent different compatibility-era shapes of the same user-facing idea. The compiler can upgrade or downgrade only when the conversion preserves meaning.

Seeing a `vhlo-to-version` failure is not just a compiler bug. It may mean the source model uses a feature that did not exist in the requested target version. In that case, the user can either choose a newer target, run a compatibility expander if one exists for that feature, or avoid the new StableHLO feature.

For maintainers, adding a new incompatible StableHLO feature implies a VHLO checklist: add the versioned op, type, or attribute; add StableHLO-to-VHLO conversion; add VHLO-to-StableHLO conversion; update mappings if a new op version supersedes an old one; add VHLO version conversion patterns if upgrade or downgrade is possible; and update compatibility and bytecode tests.

## How To Use It

As a model author or frontend user, you usually do not write `vhlo` by hand. You write or inspect StableHLO, then use StableHLO tools to serialize with a target version.

As a compiler engineer, inspect VHLO when a serialized StableHLO artifact fails to load, fails a compatibility test, or behaves differently across versions. Start by checking which operation version appears in the IR, then check whether nested attributes and types are legal for the requested target.

As a backend author, convert VHLO back to StableHLO before normal optimization. VHLO preserves compatibility; StableHLO is the dialect that carries the current semantic model and verifier.

## Minimal Example

This sketch shows the mental model rather than exact production syntax:

```mlir
%0 = stablehlo.add %lhs, %rhs : tensor<4xf32>
```

After conversion to VHLO, the operation and its type are made versioned:

```mlir
%0 = "vhlo.add_v1"(%lhs, %rhs)
  : (!vhlo.tensor_v1<4x!vhlo.f32_v1>, !vhlo.tensor_v1<4x!vhlo.f32_v1>)
    -> !vhlo.tensor_v1<4x!vhlo.f32_v1>
```

The important concept is not that users should prefer the second spelling. The important concept is that the second spelling can be checked, upgraded, downgraded, serialized, and deserialized with explicit compatibility rules.

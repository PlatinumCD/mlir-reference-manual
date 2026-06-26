# IREE `check` Dialect

The IREE `check` dialect is an assertion dialect for compiled IREE programs. It lets tests and validation modules say that a scalar, tensor, or HAL buffer view should have a particular value, then lowers those assertions into calls to IREE's optional runtime check module.

For a beginner, the important idea is that `check` is not a computation dialect. It is a testing dialect. A program still computes with normal MLIR and IREE dialects, such as `arith`, `linalg`, `tensor`, `hal`, and `flow`. The `check` operations sit next to that computation and express expectations about the results.

This matters because compiler pipelines need executable tests. A pass can be structurally correct and still generate wrong math. A backend can accept a program and still produce a wrong device result. The `check` dialect gives IREE tests a way to keep expected values inside the IR being compiled, so the same compiled artifact can compute a result and validate it at runtime.

## Operation Inventory

The `check` dialect defines these operations:

```text
check.expect_true
check.expect_false
check.expect_all_true
check.expect_eq
check.expect_eq_const
check.expect_almost_eq
check.expect_almost_eq_const
```

The dialect does not define custom types in this checkout. It works with ordinary signless integer scalars, builtin `tensor` values, and IREE HAL buffer views. The tensor and buffer-view checks can optionally take a `!hal.device` operand when the value may need to be copied from a device to host memory for inspection.

There are also no custom public attributes comparable to a layout or target attribute dialect. The main operation attributes are the constant values carried by the constant-comparison helpers and the tolerance values used by approximate equality checks.

## Scalar Checks

`check.expect_true` checks that a signless integer operand is true. In this dialect, true means nonzero. The operand can be any signless integer type, but the VM import used by the runtime path accepts an `i32`, so ordinary test IR normally uses small integer truth values.

```mlir
check.expect_true(%ok) : i32
```

`check.expect_false` checks the opposite condition. It expects the integer operand to be zero.

```mlir
check.expect_false(%failed) : i32
```

These operations are useful for scalar predicates. For example, a test can compare two scalar values with `arith.cmpi`, then pass the resulting integer predicate to `check.expect_true`. They are also useful for testing control-flow paths where the final value is a boolean-like integer.

The checks are non-fatal in the runtime implementation. They report expectation failures through the test runtime instead of modeling an MLIR exception or terminating the VM call immediately. That design is deliberate: a test can report what failed while still using ordinary compiled module execution.

## Tensor And Buffer Checks

`check.expect_all_true` checks that every element of a tensor or HAL buffer view is true. For integer-like values, true means nonzero. In the runtime module, the implementation also accepts floating-point element buffers and treats nonzero values as true.

```mlir
check.expect_all_true(%tensor) : tensor<4xi32>
check.expect_all_true<%device>(%view) : !hal.buffer_view
```

Use this operation when the program produces a vector or tensor of predicates and the test wants all lanes or all elements to pass. It is a compact way to validate elementwise comparisons.

`check.expect_eq` checks exact equality between two values of the same type. The operands can be tensors or HAL buffer views. For buffer views, the runtime checks the element type, shape, and byte contents. That makes the operation a strict equality assertion, not a numerical tolerance check.

```mlir
check.expect_eq(%lhs, %rhs) : tensor<2x2xf32>
check.expect_eq<%device>(%lhs_view, %rhs_view) : !hal.buffer_view
```

`check.expect_eq_const` is a convenience form for comparing a tensor against an inline elements attribute. It lets tests write the expected value directly inside the operation.

```mlir
check.expect_eq_const(%actual, dense<[1, 2, 3, 4]> : tensor<4xi32>)
  : tensor<4xi32>
```

The constant form is not a separate runtime check. Its canonicalization rewrites it to an `arith.constant` followed by `check.expect_eq`. This is a common MLIR pattern: the user-facing operation is convenient, but the core lowering path only needs to understand the smaller primitive set.

## Approximate Equality

`check.expect_almost_eq` checks floating-point values with a tolerance. It is meant for numerical tests where exact byte equality would be too strict because different lowering paths, device instructions, or floating-point formats may produce slightly different but acceptable results.

```mlir
check.expect_almost_eq(%actual, %expected) {atol = 1.0e-04 : f32, rtol = 0.0 : f32}
  : tensor<8xf32>
```

The comparison is shaped like NumPy's close comparison:

```text
lhs == rhs || (isfinite(rhs) && abs(lhs - rhs) <= atol + rtol * abs(rhs))
```

The default absolute tolerance is `1.0e-4`, and the default relative tolerance is `0.0`. The default is intentionally simple: it gives tests a small absolute tolerance without silently allowing differences to scale with the magnitude of the expected value. Tests can choose an explicit relative tolerance when that is the better numerical contract.

`check.expect_almost_eq_const` is the constant convenience form. Like `check.expect_eq_const`, it canonicalizes to an `arith.constant` plus the primitive approximate-equality operation, preserving the tolerance attributes.

Use approximate equality for floating-point tensors, especially when the tested path goes through vectorization, GPU code generation, lower precision formats, or target-specific math libraries. Do not use it for integer exactness tests or for cases where bitwise floating-point identity is the requirement.

## Canonicalization

The dialect has two important canonicalization patterns:

```text
check.expect_eq_const          -> arith.constant + check.expect_eq
check.expect_almost_eq_const   -> arith.constant + check.expect_almost_eq
```

These patterns make the rest of the lowering pipeline smaller. Conversion passes only need to handle `expect_eq` and `expect_almost_eq`, while authors of tests can still write compact constant assertions.

This also teaches an important MLIR design lesson. Some operations exist to make IR easier to write, generate, or read at a particular phase. They are still real operations, but their intended lifetime may be short. The constant check operations are useful near the source or test-authoring level; the primitive check operations are what matter to the HAL and VM conversions.

## HAL Conversion

The tensor-checking operations can be lowered toward HAL buffer views. This conversion applies to:

```text
check.expect_all_true
check.expect_eq
check.expect_almost_eq
```

When an operation uses builtin tensors, the conversion maps those tensors to `!hal.buffer_view` values and adds a device operand. The resulting check operation keeps the same logical meaning, but it is now expressed in terms that the IREE runtime can inspect.

For example, a tensor equality check conceptually becomes a buffer-view equality check:

```text
tensor values + selected device -> HAL buffer views + device-aware check
```

This step is important because IREE is a runtime-oriented compiler. By the time a program is executable, tensors may live in device memory, not as ordinary host-side MLIR tensor values. The check dialect therefore needs a path from value-semantics tensor IR to runtime buffer handles.

The conversion uses the surrounding IREE machinery to find the device for the operation and map tensor values to buffer views. A beginner should read this as "make the check executable by the runtime," not as a change in the assertion's meaning.

## VM Conversion

The VM conversion lowers check operations to imports from the native runtime module named `check`. The imported functions are:

```text
check.expect_true
check.expect_false
check.expect_all_true
check.expect_eq
check.expect_almost_eq
```

The two constant helper operations are expected to have canonicalized away before this point. The VM conversion only needs the five primitive runtime checks.

A distinctive detail is that these imports are optional. The conversion checks whether the import is resolved before calling it. If the runtime module provides the `check` import, the assertion runs. If the runtime module is absent, the compiled program skips the check and continues.

That behavior has a direct implication: `check` operations are for tests and validation, not for required production invariants. A compiled module can contain checks, but those checks only do work when the executable is run with a runtime that registers the check module.

The common tool for this is `iree-check-module`. It creates and registers the native check module alongside the user module, then runs the compiled program. Running the same compiled module with a normal runtime path that does not include the check module can cause the optional imports to be unavailable, so the checks are ignored.

## Runtime Behavior

The native runtime implementation registers five functions matching the VM imports. The scalar functions are direct:

```text
expect_true(i32)
expect_false(i32)
```

The buffer-view functions may need to inspect device memory. If a buffer view is already host-visible and mappable, the runtime can map and read it. If it is not host-visible, the runtime transfers the buffer to host-accessible memory, waits for the transfer, maps the copy, and then checks the contents.

`expect_all_true` walks every element and requires each element to be nonzero. The implementation supports common integer and floating-point element types.

`expect_eq` first checks that the element types match, then checks that the shapes match, then compares the raw contents. If any of those pieces differ, the expectation fails and the runtime prints useful details about the mismatch.

`expect_almost_eq` also checks element type and shape, then performs a fuzzy floating-point comparison. The runtime supports common floating-point formats used by IREE tests, including `f64`, `f32`, `f16`, `bf16`, and several 8-bit floating-point formats. On failure, it reports the first failing element and the tolerance used.

The host-transfer behavior matters when reading tests. A check that looks like a simple equality assertion in MLIR may cause a runtime copy from device memory back to the host. That is acceptable for tests, but it is not something to treat as free in performance-sensitive code.

## When To Use It

Use the `check` dialect when writing IREE tests, examples, or validation modules that should assert facts about compiled execution results. It is especially useful for end-to-end compiler tests because the assertion travels through the compiler with the code under test.

Use `check.expect_true` and `check.expect_false` for scalar predicates. Use `check.expect_all_true` when a tensor of predicates should all pass. Use `check.expect_eq` for exact tensor or buffer-view equality. Use `check.expect_almost_eq` for floating-point numerical comparisons where tolerance is part of the test contract. Use the `_const` forms when the expected tensor is a literal and you want compact, readable test IR.

Do not use the dialect as a substitute for application error handling. The imports are optional, the checks are designed around test reporting, and the runtime implementation may copy buffers back to the host. If a production program needs a required runtime guard, that should be modeled with ordinary program control flow and runtime-supported behavior, not with an optional test assertion module.

## How To Read check IR

Start by asking whether the operation is scalar, tensor, or buffer-view based. Scalar checks are direct truth assertions. Tensor checks will either be lowered to buffer views or already have buffer-view operands. If a device operand is present, the check is already in the runtime-facing shape.

Next, distinguish exact equality from approximate equality. Exact equality checks type, shape, and contents strictly. Approximate equality is for floating-point tests and is controlled by `atol` and `rtol`.

Finally, check where the IR is in the pipeline. Early IR may contain `check.expect_eq_const` or `check.expect_almost_eq_const`. Mid-pipeline IR should usually canonicalize those into constants plus primitive checks. Late IREE IR should contain HAL buffer views and optional VM imports. If the runtime check module is registered, the assertions execute; if it is not registered, the optional imports are skipped.

The main implication is that `check` makes test expectations first-class in MLIR without making them part of normal computation. It is a practical dialect for validating compiler output, but its semantics are intentionally tied to IREE's testing runtime.

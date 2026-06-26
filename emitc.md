# EmitC Dialect

## Beginner Summary

The `emitc` dialect represents C and C++ code inside MLIR. It is used near the end of a lowering pipeline when the goal is to print readable C/C++ instead of lowering to LLVM IR, SPIR-V, or another binary-oriented target.

For a beginner, the important idea is that EmitC is source-oriented. It has operations for C-like arithmetic, function calls, variables, assignments, arrays, pointers, includes, globals, control flow, classes, fields, and raw verbatim text. The final step is usually the `mlir-to-cpp` translation, which prints EmitC IR as C/C++ code.

Use EmitC when your compiler should produce C or C++ as an output artifact. Do not use it as a replacement for `arith`, `scf`, `memref`, or `func` early in a compiler pipeline.

## Why This Dialect Exists

Many compiler pipelines lower to LLVM or a hardware dialect. Some pipelines instead need a portable C or C++ source file. This can be useful for:

- Generating code for platforms where a C/C++ compiler is the final backend.
- Integrating with build systems or embedded toolchains.
- Emitting readable reference code.
- Lowering MLIR programs into C APIs, C++ classes, or C-like kernels.
- Preserving source-level constructs that would be awkward after LLVM lowering.

The `emitc` dialect exists because C/C++ has concepts that generic SSA dialects do not model directly: lvalues, declarations, includes, arrays as source types, pointer syntax, class fields, raw code fragments, and operator precedence.

## When It Matters

EmitC matters when the pipeline target is source code rather than object code. It is especially relevant after the program has already been simplified into structured loops, scalar arithmetic, memref-like access, and function calls.

Typical users include:

- Code generators that want to emit C99 or C++11-compatible code.
- MLIR examples and integrations that need a human-readable output target.
- Embedded or accelerator flows that use vendor C/C++ compilers downstream.
- Frontends that want to lower a restricted MLIR program to portable source.
- Tests that check how MLIR constructs print as C/C++.

## When To Use It

Use `emitc` when:

- You are deliberately targeting C or C++ source emission.
- Your IR is already close to C-like structured control flow.
- You need explicit arrays, pointers, lvalues, assignments, and source-level calls.
- You need to inject target-specific C/C++ names with opaque calls or verbatim text.
- You want to use `mlir-translate -mlir-to-cpp`.

Avoid using EmitC when:

- You still need high-level tensor, vector, affine, or sparse optimizations.
- You need LLVM-level ABI control, instruction selection, or binary codegen.
- You want aggressive target-specific optimization inside MLIR.
- The code depends on C/C++ semantics that the EmitC emitter does not model.

## Core Concepts

### Source-Oriented IR

EmitC is closer to C/C++ source than to machine IR. Operations such as `emitc.add`, `emitc.mul`, `emitc.cmp`, and `emitc.call_opaque` are intended to print as source expressions. Statement-like operations such as `emitc.assign`, `emitc.if`, `emitc.for`, `emitc.switch`, and `emitc.return` print as C/C++ statements.

### Lvalues

C and C++ distinguish values from locations that can be assigned to. EmitC models assignable locations with `!emitc.lvalue<T>`.

Common producers of lvalues include:

- `emitc.variable`
- `emitc.get_global`
- `emitc.subscript`
- `emitc.member`
- `emitc.member_of_ptr`
- `emitc.dereference`

Use `emitc.load` to read an lvalue as an SSA value, and `emitc.assign` to write an SSA value to an lvalue.

### EmitC Types

Important EmitC-specific types include:

- `!emitc.array<...>`: source-level C/C++ array type.
- `!emitc.ptr<T>`: pointer to `T`.
- `!emitc.lvalue<T>`: assignable location containing `T`.
- `!emitc.opaque<"...">`: a C/C++ type printed exactly as text.
- `!emitc.size_t`: unsigned `size_t`.
- `!emitc.ssize_t`: signed size type corresponding to `ssize_t`.
- `!emitc.ptrdiff_t`: pointer difference type.

The dialect also accepts many builtin integer, index, and floating-point types when they have a clear C/C++ spelling.

### Opaque Escape Hatches

EmitC deliberately has escape hatches:

- `#emitc.opaque<"...">` represents an opaque attribute printed as text.
- `!emitc.opaque<"...">` represents an opaque type printed as text.
- `emitc.call_opaque` calls a named external function or intrinsic-like spelling.
- `emitc.literal` creates an expression from literal text.
- `emitc.verbatim` emits raw C/C++ text, optionally substituting operands.

These are useful, but they shift correctness to the producer. The verifier cannot fully understand arbitrary C++ text.

### Expressions

The `emitc.expression` op groups expression operations so the C++ emitter can inline them with correct precedence and avoid unnecessary temporary variables. The `form-expressions` pass can wrap C-operator operations into expression regions and fold single-use expressions into users.

### C++ Emitter

The final printer is registered as `mlir-to-cpp`. It expects the translated operation, and almost all operations in that region, to be EmitC-compatible. It has useful options:

- `-declare-variables-at-top`: declare variables at the beginning of functions.
- `-file-id=<id>`: emit only matching `emitc.file` regions.

## Operations

EmitC operations are best understood by source-code role.

### Modules, Files, Includes, And Raw Text

- `emitc.file` groups code for a named output file id. The C++ translation can filter by `-file-id`.
- `emitc.include` emits an include directive, either quoted or angled.
- `emitc.verbatim` emits raw source text and can substitute operands into placeholders.

### Functions And Calls

- `emitc.func` defines an EmitC function with an SSACFG region.
- `emitc.return` returns from an `emitc.func`.
- `emitc.declare_func` emits a function declaration for a referenced function symbol.
- `emitc.call` calls an `emitc.func` symbol.
- `emitc.call_opaque` emits a call to a named C/C++ function or expression that is not modeled as an MLIR symbol.
- `emitc.member_call_opaque` emits an opaque member call, such as a C++ method call on a receiver.

### Values, Variables, Loads, And Assignments

- `emitc.constant` creates a typed constant from a typed attribute or an opaque attribute.
- `emitc.literal` creates a source expression from literal text.
- `emitc.variable` declares a mutable local variable and returns an lvalue or array value.
- `emitc.load` reads an `!emitc.lvalue<T>` into a value of type `T`.
- `emitc.assign` writes a value into an lvalue.
- `emitc.yield` terminates regions in EmitC control-flow and expression operations.

### Arithmetic, Logical, And Comparison Operators

- `emitc.add` emits `+`.
- `emitc.sub` emits `-`.
- `emitc.mul` emits `*`.
- `emitc.div` emits `/`.
- `emitc.rem` emits `%`.
- `emitc.unary_plus` emits unary `+`.
- `emitc.unary_minus` emits unary `-`.
- `emitc.bitwise_and` emits bitwise `&`.
- `emitc.bitwise_or` emits bitwise `|`.
- `emitc.bitwise_xor` emits bitwise `^`.
- `emitc.bitwise_not` emits bitwise `~`.
- `emitc.bitwise_left_shift` emits `<<`.
- `emitc.bitwise_right_shift` emits `>>`.
- `emitc.logical_and` emits logical `&&`.
- `emitc.logical_or` emits logical `||`.
- `emitc.logical_not` emits logical `!`.
- `emitc.cmp` emits comparison operators such as `==`, `!=`, `<`, `<=`, `>`, `>=`, and C++ three-way comparison.
- `emitc.conditional` emits a ternary conditional expression.
- `emitc.cast` emits a cast between EmitC-supported types.

### Pointers, Arrays, Members, And Globals

- `emitc.address_of` emits address-of on an lvalue.
- `emitc.dereference` emits pointer dereference and returns an lvalue.
- `emitc.subscript` emits array, pointer, or opaque subscription and returns an lvalue.
- `emitc.member` emits member access with `.`.
- `emitc.member_of_ptr` emits member access with `->`.
- `emitc.global` defines a global variable.
- `emitc.get_global` obtains access to a global variable.

### Structured Control Flow

- `emitc.if` emits an if or if-else statement.
- `emitc.for` emits a structured for loop.
- `emitc.do` emits a do-while loop.
- `emitc.switch` emits a switch statement with case regions and an optional default.

### Classes And Fields

- `emitc.class` defines a C++ class, struct, or union-like aggregate.
- `emitc.field` defines a field inside an `emitc.class`.
- `emitc.get_field` obtains access to a field inside a class instance or class wrapper context.

### Expression Grouping

- `emitc.expression` groups expression-producing EmitC operations into one C/C++ expression. It can be marked `noinline` when a producer needs a separate temporary instead of inline expression emission.

## Transformations

EmitC-specific transformations include:

- `form-expressions`: wraps EmitC C-operator operations in `emitc.expression` and folds single-use expressions into users where possible.
- `wrap-emitc-func-in-class`: transforms `emitc.func` operations into `emitc.class` operations. Function arguments become fields, and the function body becomes a member method. The `func-name` option controls the generated member function name, defaulting to `operator()`.

The broader conversion passes that produce EmitC are:

- `convert-to-emitc`: generic conversion to EmitC through dialect interfaces. Options include `filter-dialects` and `lower-to-cpp`.
- `convert-arith-to-emitc`: converts supported `arith` operations to EmitC operators and constants.
- `convert-func-to-emitc`: converts `func.func`, calls, and returns to EmitC functions and calls. It has a `lower-to-cpp` option.
- `convert-math-to-emitc`: converts supported `math` operations to `emitc.call_opaque` calls targeting libc/libm-style functions. The `language-target` option can select `c99` or `cpp11`.
- `convert-memref-to-emitc`: converts supported `memref` operations to EmitC arrays, pointers, subscripts, loads, stores, allocations, and copies. It has a `lower-to-cpp` option.
- `convert-scf-to-emitc`: converts structured `scf` control flow to EmitC structured statements while preserving structured control flow.

## Conversions/Lowering Paths

A common EmitC pipeline is:

1. Lower high-level tensor, linalg, vector, or domain-specific IR to scalar, structured, and memory-like IR.
2. Convert structured control flow with `convert-scf-to-emitc`.
3. Convert scalar arithmetic with `convert-arith-to-emitc`.
4. Convert math calls with `convert-math-to-emitc` if needed.
5. Convert memory operations with `convert-memref-to-emitc`.
6. Convert functions with `convert-func-to-emitc`.
7. Optionally run `form-expressions` to make the output more source-like.
8. Optionally run `wrap-emitc-func-in-class` for C++ class wrappers.
9. Print with `mlir-translate -mlir-to-cpp`.

The generic `convert-to-emitc` pass can use dialect conversion interfaces to delegate conversion pattern population to dialects that know how to lower themselves to EmitC.

EmitC does not usually lower onward to LLVM. It is itself a target-facing representation. The "lowering" after EmitC is source translation to C/C++ text.

## Example IR

This example emits a simple C-like function with arithmetic operators.

```mlir
emitc.include <"math.h">

emitc.func @saxpy(%x: f32, %y: f32, %a: f32) -> f32 {
  %prod = emitc.mul %a, %x : (f32, f32) -> f32
  %sum = emitc.add %prod, %y : (f32, f32) -> f32
  emitc.return %sum : f32
}
```

This example shows source-level array access, lvalues, and assignment.

```mlir
emitc.func @store(%buffer: !emitc.array<4xf32>, %i: !emitc.size_t, %value: f32) {
  %slot = emitc.subscript %buffer[%i] : (!emitc.array<4xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  emitc.assign %value : f32 to %slot : !emitc.lvalue<f32>
  emitc.return
}
```

This example shows a C++ class wrapper shape.

```mlir
emitc.class final @Model {
  emitc.field @weights : !emitc.array<4xf32>
  emitc.func @execute() {
    %w = emitc.get_field @weights : !emitc.array<4xf32>
    emitc.return
  }
}
```

This example shows file grouping and a forward declaration.

```mlir
emitc.file "header" {
  emitc.include <"stdint.h">
  emitc.declare_func @run
  emitc.func @run() {
    emitc.return
  }
}
```

## Mental Model

Think of EmitC as "C/C++ before printing." It is still MLIR, so it has SSA values, verifiers, regions, symbols, and passes. But the meaning of each operation is tied to the source text it will emit.

The central question is not "what machine instruction does this become?" The question is "what C/C++ source fragment does this represent?"

That explains many design choices:

- `!emitc.lvalue<T>` exists because assignment and address-of need locations.
- `emitc.expression` exists because source code has nested expressions and precedence.
- `emitc.opaque` exists because C/C++ ecosystems often need names and types MLIR does not model.
- `emitc.verbatim` exists because no source dialect can model every target header, macro, pragma, or vendor API.
- `emitc.file` exists because source generation may produce more than one output file.

## Gotchas

- EmitC is a target dialect. Running high-level optimizations after converting to EmitC is usually too late.
- Opaque types, opaque attributes, `emitc.literal`, and `emitc.verbatim` are powerful but weakly checked. Use them for target integration, not as a way to avoid modeling important semantics.
- `emitc.call_opaque` is not a symbol call. It prints a named call-like expression and assumes the generated source environment provides it.
- `emitc.declare_func` must reference a valid `emitc.func` symbol.
- `!emitc.array` models source arrays, which have different behavior than memrefs or LLVM pointers.
- `!emitc.lvalue<T>` is not an SSA value of type `T`; use `emitc.load` to read it.
- `emitc.assign` writes to an lvalue and returns no SSA result.
- `emitc.expression` affects printing. It is not a general-purpose region container.
- The `mlir-to-cpp` translation expects almost all relevant IR to be EmitC-compatible. Leftover high-level dialect operations usually cause translation failures.
- `emitc.verbatim` can easily generate invalid C/C++ if the format string and operands do not match the intended output.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/EmitC/IR/EmitCBase.td`
- `mlir/include/mlir/Dialect/EmitC/IR/EmitC.td`
- `mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td`
- `mlir/include/mlir/Dialect/EmitC/IR/EmitCAttributes.td`
- `mlir/include/mlir/Dialect/EmitC/IR/EmitCInterfaces.td`
- `mlir/include/mlir/Dialect/EmitC/Transforms/Passes.td`
- `mlir/lib/Dialect/EmitC/IR/EmitC.cpp`
- `mlir/lib/Dialect/EmitC/Transforms/FormExpressions.cpp`
- `mlir/lib/Dialect/EmitC/Transforms/WrapFuncInClass.cpp`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/ConvertToEmitC/`
- `mlir/lib/Conversion/ArithToEmitC/`
- `mlir/lib/Conversion/FuncToEmitC/`
- `mlir/lib/Conversion/MathToEmitC/`
- `mlir/lib/Conversion/MemRefToEmitC/`
- `mlir/lib/Conversion/SCFToEmitC/`
- `mlir/include/mlir/Target/Cpp/CppEmitter.h`
- `mlir/lib/Target/Cpp/TranslateToCpp.cpp`
- `mlir/lib/Target/Cpp/TranslateRegistration.cpp`

Useful tests:

- `mlir/test/Dialect/EmitC/ops.mlir`
- `mlir/test/Dialect/EmitC/types.mlir`
- `mlir/test/Dialect/EmitC/attrs.mlir`
- `mlir/test/Dialect/EmitC/form-expressions.mlir`
- `mlir/test/Dialect/EmitC/wrap-func-in-class.mlir`
- `mlir/test/Conversion/MemRefToEmitC/`
- `mlir/test/Conversion/ConvertToEmitC/`
- `mlir/test/Target/Cpp/`

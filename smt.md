# SMT Dialect

## Beginner Summary

The `smt` dialect models satisfiability modulo theories inside MLIR.

SMT means "satisfiability modulo theories." In practice, it is a way to build
logical formulas and ask a solver whether those formulas are satisfiable,
unsatisfiable, or unknown.

The `smt` dialect gives MLIR operations for:

- Solver scopes.
- Symbolic values.
- Assertions.
- Satisfiability checks.
- Boolean formulas.
- Quantifiers.
- Infinite-precision integers.
- Fixed-width bit-vectors.
- SMT arrays.
- Uninterpreted sorts and functions.
- Export to SMT-LIB.

Think of `smt` as MLIR's vocabulary for talking to an SMT solver.

It is not a normal executable-program dialect. It does not describe machine
instructions, tensor computation, or runtime control flow. It describes logical
queries that a solver can answer.

## Why This Dialect Exists

Compilers often need to prove facts.

Examples:

- Is this set of constraints satisfiable?
- Can this index ever be out of bounds?
- Are two expressions equivalent?
- Can these parameters be chosen so a transform is legal?
- Does this rewrite preserve a required relationship?
- Is a symbolic integer within a requested range?
- Can a bit-vector expression overflow, alias, or match a target pattern?

Those questions are naturally expressed as formulas.

SMT solvers already understand common mathematical theories such as booleans,
integers, arrays, bit-vectors, and uninterpreted functions. The `smt` dialect
lets MLIR build those formulas directly in IR instead of hand-assembling
SMT-LIB strings.

This gives compiler code a structured representation that can be verified,
printed, transformed, inspected, and exported.

## When It Matters

The `smt` dialect matters when an MLIR pipeline needs symbolic reasoning.

It is useful for:

- Constraint solving.
- Transform-parameter constraints.
- Verification-oriented IR.
- Lowering proof obligations to external SMT solvers.
- Modeling integer formulas.
- Modeling fixed-width machine arithmetic with bit-vectors.
- Modeling abstract memory with SMT arrays.
- Modeling uninterpreted domains and functions.
- Emitting SMT-LIB scripts for tools such as Z3.

A typical flow looks like this:

```text
compiler analysis or transform problem
  -> build smt.solver with symbolic values and assertions
  -> export-smtlib
  -> external SMT solver
  -> sat / unsat / unknown result
```

The dialect is also used by the Transform dialect SMT extension:

```text
transform params
  -> transform.smt.constrain_params
  -> SMT formulas over symbolic param values
```

In this checkout, that Transform extension records the intended constraint
shape, but its operational implementation is still a TODO.

## When To Use It

Use the `smt` dialect when the thing you need to represent is a logical query.

Use it for:

- Declaring symbolic values with SMT sorts.
- Asserting constraints.
- Checking satisfiability.
- Representing formulas over SMT booleans and integers.
- Representing exact fixed-width bit-vector operations.
- Representing SMT arrays as functional arrays.
- Representing quantified formulas.
- Exporting a solver query to SMT-LIB.

Do not use `smt` as a replacement for ordinary MLIR computation.

Use other dialects instead when you need:

- Executable arithmetic: use `arith`, `math`, or target dialects.
- Program control flow: use `scf` or `cf`.
- Memory operations: use `memref`, `tensor`, or `llvm`.
- Machine integer arithmetic in a program: use `arith` or target operations.
- Runtime calls into a solver API: use an implementation dialect or runtime
  binding, not the logical formula dialect itself.

The `smt` dialect is for asking questions about values. It is not for
executing the program that produced those values.

## Core Concepts

### SMT Values Are Logical Values

SMT values are not ordinary MLIR runtime values.

For example:

```mlir
%x = smt.declare_fun "x" : !smt.int
```

This does not compute an integer at runtime. It declares a symbolic integer
named with the prefix `x`. A solver may assign any integer value to it as long
as all asserted constraints are satisfied.

### Solver Scope

`smt.solver` creates a solver instance and a lifetime for SMT values.

```mlir
smt.solver () : () -> () {
  %x = smt.declare_fun "x" : !smt.int
  ...
}
```

SMT operations are intended to run within the lifespan of a solver region.
SMT values must not escape from one solver context to another.

`smt.solver` can syntactically take non-SMT inputs and return non-SMT results,
which allows MLIR control-flow-style integration. However, the current
SMT-LIB exporter only supports solver scopes with no inputs and no results.

### Assertions And Checks

An assertion adds a boolean formula to the solver state:

```mlir
smt.assert %condition
```

`smt.check` asks whether the current assertions are satisfiable:

```mlir
smt.check sat {} unknown {} unsat {}
```

The operation has three regions because a solver can answer:

- `sat`: the constraints are satisfiable.
- `unsat`: the constraints are contradictory.
- `unknown`: the solver cannot decide.

For SMT-LIB export, the check operation must not produce values and the three
regions must be empty.

### Sorts And Types

SMT calls its types "sorts." MLIR represents them as dialect types:

- `!smt.bool`: boolean sort.
- `!smt.int`: infinite-precision integer sort.
- `!smt.bv<8>`: fixed-size bit-vector sort with width 8.
- `!smt.array<[!smt.int -> !smt.bool]>`: array from integer indices to boolean
  values.
- `!smt.func<(!smt.int, !smt.bool) !smt.bool>`: first-order SMT function.
- `!smt.sort<"name">`: uninterpreted sort.
- `!smt.sort<"name"[!smt.int]>`: parameterized uninterpreted sort use.

SMT function types can have any number of domain sorts but exactly one range
sort. Function types cannot be nested because SMT is first-order here.

### Declare Versus Compute

`smt.declare_fun` creates a fresh symbolic value each time.

This is true even when two declarations have the same name prefix:

```mlir
%a = smt.declare_fun "x" : !smt.int
%b = smt.declare_fun "x" : !smt.int
```

`%a` and `%b` are different symbolic values. The string is a naming prefix for
readable export, not an identity key.

This is why `smt.declare_fun` is not marked `Pure`. Generic CSE must not merge
two declarations just because they look textually similar.

### SMT Arrays Are Functional

SMT arrays are immutable logical values.

`smt.array.store` does not mutate an array in place. It returns a new array
that is equal to the old array except at one index.

```mlir
%updated = smt.array.store %array[%index], %value
  : !smt.array<[!smt.int -> !smt.bool]>
```

`smt.array.select` reads the value associated with an index:

```mlir
%value = smt.array.select %updated[%index]
  : !smt.array<[!smt.int -> !smt.bool]>
```

This is a logical memory model, not a mutable buffer model.

### Bit-Vectors Versus Integers

`!smt.int` is an infinite-precision mathematical integer.

`!smt.bv<N>` is a fixed-width bit-vector.

Use `!smt.int` for mathematical constraints such as:

```text
0 <= x && x < 10
```

Use `!smt.bv<N>` when width and wraparound matter, such as modeling machine
integer arithmetic.

The dialect has conversion operations `smt.int2bv` and `smt.bv2int`, but the
current SMT-LIB exporter in this checkout reports these operations as
unsupported for emission.

### Export Is The Main Lowering Path

The SMT dialect does not lower to LLVM or executable code in the usual way.

Its main target path is:

```text
smt dialect
  -> mlir-translate --export-smtlib
  -> SMT-LIB text
  -> external solver
```

The exporter also has:

```text
--smtlibexport-inline-single-use-values
```

which prints single-use expressions inline instead of introducing SMT-LIB
`let` bindings.

## Operations

### Solver And Script Operations

`smt.solver`
: Creates a solver scope and lifetime for SMT values. It contains a region
  that acts like a structured SMT script.

`smt.set_logic`
: Sets the SMT logic string, such as `"QF_LIA"`, `"QF_BV"`, `"AUFLIA"`, or
  `"HORN"`.

`smt.declare_fun`
: Declares a fresh symbolic value or function. If the result type is a
  function sort, export uses SMT-LIB `declare-fun`; otherwise it uses
  `declare-const`.

`smt.apply_func`
: Applies an SMT function value to arguments. The operand and result types
  must match the `!smt.func` domain and range.

`smt.assert`
: Adds a boolean formula to the current solver assertions.

`smt.check`
: Checks satisfiability and branches to `sat`, `unknown`, or `unsat` regions.
  Export supports only the simple no-result form with empty regions.

`smt.push`
: Pushes levels on the assertion stack.

`smt.pop`
: Pops levels from the assertion stack.

`smt.reset`
: Resets the solver.

`smt.yield`
: Terminates SMT regions, including `smt.solver`, `smt.check`,
  `smt.forall`, and `smt.exists`.

### Core Boolean And Formula Operations

`smt.constant`
: Produces `true` or `false`.

`smt.not`
: Boolean negation.

`smt.and`
: Variadic boolean conjunction. Requires at least two operands.

`smt.or`
: Variadic boolean disjunction. Requires at least two operands.

`smt.xor`
: Variadic boolean exclusive OR. Requires at least two operands.

`smt.implies`
: Boolean implication.

`smt.eq`
: Equality over any non-function SMT type. It is variadic and requires at
  least two operands.

`smt.distinct`
: Pairwise distinctness over any non-function SMT type. It is variadic and
  requires at least two operands.

`smt.ite`
: SMT if-then-else expression. The then value, else value, and result have the
  same SMT type.

`smt.forall`
: Universal quantifier. The body must yield exactly one `!smt.bool`.

`smt.exists`
: Existential quantifier. The body must yield exactly one `!smt.bool`.

Quantifiers can carry bound variable name prefixes, a weight, optional pattern
regions, or `no_pattern`. The exporter supports quantifiers and patterns, but
currently rejects `no_pattern`.

### Integer Theory Operations

`smt.int.constant`
: Produces an infinite-precision integer constant.

`smt.int.add`
: Variadic integer addition.

`smt.int.mul`
: Variadic integer multiplication.

`smt.int.sub`
: Binary integer subtraction.

`smt.int.div`
: Binary integer division.

`smt.int.mod`
: Binary integer modulo.

`smt.int.abs`
: Integer absolute value.

`smt.int.cmp`
: Integer comparison. Predicates are:

- `lt`: less than.
- `le`: less than or equal.
- `gt`: greater than.
- `ge`: greater than or equal.

`smt.int2bv`
: Converts an SMT integer to an inferred-width bit-vector result. This is
  modeled for solver APIs such as Z3, but the current SMT-LIB exporter does not
  support emitting it.

### Bit-Vector Operations

`smt.bv.constant`
: Produces a fixed-width bit-vector constant using `#smt.bv<...>`.

`smt.bv.neg`
: Two's-complement unary minus.

`smt.bv.not`
: Bitwise negation.

`smt.bv.add`
: Bit-vector addition.

`smt.bv.mul`
: Bit-vector multiplication.

`smt.bv.udiv`
: Unsigned bit-vector division.

`smt.bv.sdiv`
: Signed bit-vector division.

`smt.bv.urem`
: Unsigned bit-vector remainder.

`smt.bv.srem`
: Signed remainder where the sign follows the dividend.

`smt.bv.smod`
: Signed modulo where the sign follows the divisor.

`smt.bv.and`
: Bitwise AND.

`smt.bv.or`
: Bitwise OR.

`smt.bv.xor`
: Bitwise exclusive OR.

`smt.bv.shl`
: Shift left.

`smt.bv.lshr`
: Logical shift right.

`smt.bv.ashr`
: Arithmetic shift right.

`smt.bv.cmp`
: Signed or unsigned bit-vector comparison. Predicates are:

- `slt`, `sle`, `sgt`, `sge`
- `ult`, `ule`, `ugt`, `uge`

`smt.bv.concat`
: Concatenates two bit-vectors.

`smt.bv.extract`
: Extracts a slice of bits starting at a low bit. The result width determines
  the high bit.

`smt.bv.repeat`
: Repeats one bit-vector value multiple times by repeated concatenation.

`smt.bv2int`
: Converts a bit-vector to an SMT integer. It can treat the bit-vector as
  unsigned or as signed two's-complement when the `signed` marker is present.
  The current SMT-LIB exporter does not support emitting it.

### Array Operations

`smt.array.broadcast`
: Constructs a constant array where every index maps to the same value.

`smt.array.select`
: Reads the value associated with an index.

`smt.array.store`
: Returns a new array with one index updated to a new value.

These correspond to the SMT array theory, not MLIR memory mutation.

## Transformations

### Generic MLIR Transformations

The SMT dialect does not define a large set of ordinary optimization passes.

Most expression operations are pure, so generic MLIR infrastructure such as
verification, round-trip parsing, and CSE can still be useful.

The important exception is `smt.declare_fun`. It is not pure because each
operation creates a fresh symbolic value. A CSE pass must not merge two
declarations, even if they use the same name prefix and type.

Constants such as `smt.constant`, `smt.int.constant`, and `smt.bv.constant`
have constant-like behavior and folders.

### Transform Dialect SMT Extension

The Transform dialect has an SMT extension operation:

```mlir
transform.smt.constrain_params
```

It lets a transform script express constraints over transform parameters by
mapping those parameters to SMT block arguments.

Conceptually:

```text
transform params
  -> symbolic SMT values
  -> SMT assertions
  -> satisfying assignment becomes new transform params
```

In the current implementation, the operation records and verifies the shape of
the constraint region, but its interpreter semantics are still documented as a
TODO and always fail. The chapter treats it as an important design direction,
not as a fully usable solver-backed transform yet.

## Conversions And Export Paths

### SMT-LIB Export

The main conversion path is SMT-LIB export:

```bash
mlir-translate --export-smtlib input.mlir
```

The exporter maps SMT dialect constructs to SMT-LIB text:

- `smt.solver` becomes a solver script section.
- `!smt.bool` becomes `Bool`.
- `!smt.int` becomes `Int`.
- `!smt.bv<N>` becomes `(_ BitVec N)`.
- `!smt.array<[A -> B]>` becomes `(Array A B)`.
- `!smt.sort<"name">` becomes an uninterpreted sort declaration.
- `smt.declare_fun` becomes `declare-const` or `declare-fun`.
- `smt.assert` becomes `assert`.
- `smt.check` becomes `check-sat`.
- Boolean, integer, bit-vector, array, quantifier, and function-application
  expressions are printed as SMT-LIB terms.

The option:

```bash
mlir-translate --export-smtlib --smtlibexport-inline-single-use-values input.mlir
```

prints single-use expressions inline instead of introducing `let` bindings.

### Export Limitations

The current exporter is intentionally stricter than the dialect.

For export:

- `smt.solver` must have no inputs and no results.
- An exported solver body must contain only SMT dialect operations.
- `smt.check` must have no results.
- The `sat`, `unknown`, and `unsat` regions of exported `smt.check` must be
  empty.
- `smt.int2bv` and `smt.bv2int` are currently rejected by the exporter.
- Quantifier `no_pattern` is currently rejected by the exporter.

These limitations do not mean the operations are invalid MLIR. They mean that
this particular SMT-LIB emission path does not yet support them.

### No Executable Lowering

The SMT dialect is not normally lowered to LLVM, SPIR-V, or machine code.

It represents solver queries. The external solver is the consumer.

If a compiler needs runtime solver calls, that is a separate runtime API
problem. The `smt` dialect itself is the logical formula representation.

## Example IR

### Integer Range Query

This example asks whether there exists an integer `x` such that:

```text
0 <= x && x < 10
```

```mlir
smt.solver () : () -> () {
  smt.set_logic "QF_LIA"
  %x = smt.declare_fun "x" : !smt.int
  %zero = smt.int.constant 0
  %ten = smt.int.constant 10
  %ge_zero = smt.int.cmp ge %x, %zero
  %lt_ten = smt.int.cmp lt %x, %ten
  %in_range = smt.and %ge_zero, %lt_ten
  smt.assert %in_range
  smt.check sat {} unknown {} unsat {}
}
```

This should be satisfiable because many integers, such as `0`, satisfy the
constraints.

### Bit-Vector Query

This example uses fixed-width arithmetic.

```mlir
smt.solver () : () -> () {
  smt.set_logic "QF_BV"
  %x = smt.declare_fun "x" : !smt.bv<8>
  %one = smt.bv.constant #smt.bv<1> : !smt.bv<8>
  %sum = smt.bv.add %x, %one : !smt.bv<8>
  %same = smt.eq %sum, %x : !smt.bv<8>
  %changed = smt.not %same
  smt.assert %changed
  smt.check sat {} unknown {} unsat {}
}
```

Because `!smt.bv<8>` is an 8-bit bit-vector, `%sum` uses fixed-width
bit-vector arithmetic, not infinite-precision integer arithmetic.

### Array Query

This example constructs a constant array, stores `true` at index `0`, selects
that index, and asserts the selected value.

```mlir
smt.solver () : () -> () {
  smt.set_logic "QF_AUFLIA"
  %zero = smt.int.constant 0
  %false = smt.constant false
  %true = smt.constant true
  %base = smt.array.broadcast %false : !smt.array<[!smt.int -> !smt.bool]>
  %updated = smt.array.store %base[%zero], %true
    : !smt.array<[!smt.int -> !smt.bool]>
  %selected = smt.array.select %updated[%zero]
    : !smt.array<[!smt.int -> !smt.bool]>
  smt.assert %selected
  smt.check sat {} unknown {} unsat {}
}
```

The array value is immutable. `%updated` is a new array value.

### Quantified Formula

This example asserts a trivial universally quantified formula.

```mlir
smt.solver () : () -> () {
  smt.set_logic "AUFLIA"
  %formula = smt.forall ["i"] {
  ^bb0(%i: !smt.int):
    %zero = smt.int.constant 0
    %non_negative = smt.int.cmp ge %i, %zero
    %trivial = smt.implies %non_negative, %non_negative
    smt.yield %trivial : !smt.bool
  }
  smt.assert %formula
  smt.check sat {} unknown {} unsat {}
}
```

The quantifier body binds `%i` as a symbolic integer. The body must yield one
boolean value.

## Mental Model

Most MLIR dialects model computations a compiler will eventually execute.

The `smt` dialect models a question.

When you write:

```mlir
%x = smt.declare_fun "x" : !smt.int
%c = smt.int.constant 10
%p = smt.int.cmp lt %x, %c
smt.assert %p
```

you are not computing `%x < 10` for a runtime value of `%x`. You are building a
logical formula that says:

```text
find an integer x such that x < 10
```

Then `smt.check` asks the solver whether such an assignment exists.

Good beginner rule:

```text
Use smt when you need proof or satisfiability.
Use arith/scf/memref when you need executable computation.
```

## Gotchas

- `smt.declare_fun` creates a fresh symbolic value every time. The name string
  is only a prefix for readability.
- `!smt.int` is infinite precision. It is not the same as `i32` or `i64`.
- `!smt.bv<N>` is fixed-width. Use it when bit width and wraparound matter.
- SMT arrays are immutable logical arrays, not mutable memrefs.
- `smt.array.store` returns a new array value.
- `smt.check` has three possible outcomes: `sat`, `unsat`, and `unknown`.
- The dialect can represent solver-control regions with results, but the
  SMT-LIB exporter only supports simple no-result checks.
- The SMT-LIB exporter only supports solver scopes with no inputs and no
  results.
- The exporter rejects non-SMT operations inside exported solver scopes.
- `smt.int2bv`, `smt.bv2int`, and quantifier `no_pattern` are present in the
  dialect but currently unsupported by the SMT-LIB exporter.
- Transform dialect SMT constraints are present as IR, but the current
  interpreter semantics are still a TODO.

## Source Map

Primary definitions:

- `mlir/include/mlir/Dialect/SMT/IR/SMT.td`
- `mlir/include/mlir/Dialect/SMT/IR/SMTDialect.td`
- `mlir/include/mlir/Dialect/SMT/IR/SMTTypes.td`
- `mlir/include/mlir/Dialect/SMT/IR/SMTAttributes.td`
- `mlir/include/mlir/Dialect/SMT/IR/SMTOps.td`
- `mlir/include/mlir/Dialect/SMT/IR/SMTIntOps.td`
- `mlir/include/mlir/Dialect/SMT/IR/SMTBitVectorOps.td`
- `mlir/include/mlir/Dialect/SMT/IR/SMTArrayOps.td`
- `mlir/lib/Dialect/SMT/IR/`
- `mlir/include/mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.td`
- `mlir/lib/Dialect/Transform/SMTExtension/`
- `mlir/include/mlir/Target/SMTLIB/ExportSMTLIB.h`
- `mlir/lib/Target/SMTLIB/ExportSMTLIB.cpp`
- `mlir/test/Dialect/SMT/`
- `mlir/test/Target/SMTLIB/`
- `mlir/test/Target/ExportSMTLIB/`

All SMT dialect operations covered in this chapter:

- `smt.and`
- `smt.apply_func`
- `smt.array.broadcast`
- `smt.array.select`
- `smt.array.store`
- `smt.assert`
- `smt.bv.add`
- `smt.bv.and`
- `smt.bv.ashr`
- `smt.bv.cmp`
- `smt.bv.concat`
- `smt.bv.constant`
- `smt.bv.extract`
- `smt.bv.lshr`
- `smt.bv.mul`
- `smt.bv.neg`
- `smt.bv.not`
- `smt.bv.or`
- `smt.bv.repeat`
- `smt.bv.sdiv`
- `smt.bv.shl`
- `smt.bv.smod`
- `smt.bv.srem`
- `smt.bv.udiv`
- `smt.bv.urem`
- `smt.bv.xor`
- `smt.bv2int`
- `smt.check`
- `smt.constant`
- `smt.declare_fun`
- `smt.distinct`
- `smt.eq`
- `smt.exists`
- `smt.forall`
- `smt.implies`
- `smt.int.abs`
- `smt.int.add`
- `smt.int.cmp`
- `smt.int.constant`
- `smt.int.div`
- `smt.int.mod`
- `smt.int.mul`
- `smt.int.sub`
- `smt.int2bv`
- `smt.ite`
- `smt.not`
- `smt.or`
- `smt.pop`
- `smt.push`
- `smt.reset`
- `smt.set_logic`
- `smt.solver`
- `smt.xor`
- `smt.yield`

SMT-related transform/export operations and commands covered:

- `transform.smt.constrain_params`
- `mlir-translate --export-smtlib`
- `mlir-translate --export-smtlib --smtlibexport-inline-single-use-values`

# CIRCT Verif Dialect

## Beginner Summary

The CIRCT `verif` dialect represents verification intent inside hardware IR.
It gives the compiler explicit operations for assertions, assumptions, coverage
points, formal tests, simulation tests, symbolic inputs, bounded model checking,
equivalence checking, refinement checking, and local formal contracts.

For a beginner, the important idea is that `verif` is not a normal hardware
implementation dialect. Most `verif` operations are not "gates" that should
exist in the final design. They are statements about the design: things that
must always be true, conditions that a formal engine may assume, situations that
should be reachable, and test harnesses that explain how the design should be
checked. Later passes either lower these statements into SystemVerilog
verification constructs, lower them into SMT problems, or erase/apply them after
they have served their verification purpose.

## Why This Dialect Exists

Hardware compilers need a way to carry verification information alongside the
design being transformed. A source language may contain assertions and coverage
checks. A compiler pass may generate a formal contract around a block. A test
author may want to create symbolic inputs and ask a model checker to find a
failure. If this information is represented only as comments or as backend text,
it cannot reliably survive optimization, lowering, and analysis.

The `verif` dialect solves that by making verification constructs first-class
MLIR operations. A pass can inspect them, combine them, move them through
lowering, convert them to another verification backend, or reject invalid forms
before the compiler emits final output.

## When It Matters

Use `verif` when the IR needs to say something about correctness, not just
implementation.

It matters when a design contains assertions that should be emitted to
SystemVerilog, assumptions that constrain formal analysis, coverage points that
describe useful reachable behavior, or symbolic values that stand in for
arbitrary inputs. It also matters for compiler-generated verification tasks such
as bounded model checking, logical equivalence checking, refinement checking, and
contract checking.

It is especially important in CIRCT flows that target formal tools. The dialect
connects hardware IR to LTL properties, SMT lowering, SystemVerilog assertion
emission, and test extraction. Without it, the compiler would have to encode
verification intent indirectly in general-purpose hardware operations, which
would make the IR harder to analyze and easier to miscompile.

## When To Use It

Use `verif.assert` when a property must hold. Use `verif.assume` when a formal
engine is allowed to treat a property as a precondition. Use `verif.cover` when
you want to ask whether a situation is reachable. Use the clocked forms when the
property should be sampled on a specific clock edge.

Use `verif.formal` to package a formal unit test. Use `verif.simulation` to
package a simulation unit test. Use `verif.symbolic_value` to create an
unconstrained value for formal analysis. Use `verif.contract` when you want a
local interface-style specification with preconditions and postconditions.

Use the relation-checking operations when the IR itself represents a checking
problem. `verif.lec` asks whether two circuits are logically equivalent.
`verif.refines` asks whether one circuit refines another. `verif.bmc` encodes a
bounded model checking problem with explicit initialization, loop, and circuit
regions.

Do not use `verif` as a substitute for normal hardware structure. If you want to
build modules, wires, instances, registers, combinational logic, or memories,
those belong in CIRCT hardware dialects such as `hw`, `comb`, `seq`, `sv`, or
other implementation dialects. `verif` describes how those designs should be
checked.

## Core Concepts

### Assertions, Assumptions, And Coverage

The immediate assertion-like operations are `verif.assert`, `verif.assume`, and
`verif.cover`.

An assertion says "this property must hold." If the property can be false in a
formal run or simulation, that is a verification failure.

An assumption says "the checker may restrict the search to states where this
property holds." Assumptions are powerful because they prune the verification
space, but they also change what is being proven. A design can appear correct
under an unrealistic assumption, so assumptions should describe real environment
constraints.

A cover point says "try to find a trace where this property holds." Covers are
not failures. They are useful for proving that a testbench can reach interesting
states or that a feature is not dead.

All three operations accept an LTL property operand. They may also carry an
optional `i1` enable and an optional string label. The enable acts like a guard:
the property only matters when the enable is true. The label gives generated
output and diagnostics a stable name.

### Clocked Properties

The clocked forms are `verif.clocked_assert`, `verif.clocked_assume`, and
`verif.clocked_cover`. They attach the property to an explicit clock edge. The
edge is represented by the `ClockEdge` attribute, which can be `posedge`,
`negedge`, or `edge`.

These operations are useful when the property should be sampled on a particular
clock rather than interpreted as an unclocked property. The
`verify-clocked-assert-like` pass checks that the property tree does not contain
nested `ltl.clock` or `ltl.disable` operations, because the clocked Verif op
already supplies the single clocking context.

### Messages And Reset Guards

`verif.format_verilog_string` creates an `!hw.string` value using Verilog-style
formatting. `verif.print` consumes that string and is lowered by the
SystemVerilog conversion into display-style output. This is useful for
diagnostics in simulation-oriented flows.

`verif.has_been_reset` creates an `i1` signal that becomes true only after a
proper reset has been observed. It takes a clock, a reset signal, and an `async`
boolean attribute. This operation is useful for disabling assertions during
startup, before the design has reached a meaningful initialized state.

### Formal And Simulation Entry Points

`verif.formal` defines a formal unit test. Its body contains the hardware under
test, symbolic values, assumptions, assertions, and covers. The op has a symbol
name and a dictionary of parameters. Some parameters have conventional meaning
for test runners, such as `ignore`, `require_runners`, and `exclude_runners`.

`verif.simulation` defines a simulation unit test. Its region receives a
`!seq.clock` block argument and an `i1` init signal. It must end with
`verif.yield` carrying two `i1` values: `done` and `success`. The done signal
tells the simulator when the test is complete; the success signal tells whether
the test passed.

The `verif-lower-tests` pass converts these test operations into top-level
`hw.module` operations so downstream tools can run them.

### Symbolic Values

`verif.symbolic_value` creates an unconstrained value. Formal tools may assign
any concrete value to it, and that value may vary between timesteps. In a
formal test, symbolic values are how you say "let the checker choose the input
that exposes a failure."

The symbolic value op is valid inside `verif.formal` and `hw.module`. The
`verif-lower-symbolic-values` pass lowers it into a backend-friendly form, such
as an external-module boundary or a SystemVerilog wire marked as symbolic for
Yosys-style flows.

### Contracts

`verif.contract` describes a local specification for a piece of IR. Its body
contains `verif.require` and `verif.ensure` operations.

`verif.require` is a precondition. It describes what must be true about the
inputs or surrounding context for the implementation to be expected to work.

`verif.ensure` is a postcondition. It describes what the implementation promises
when the requirements hold.

Contracts have two related uses. When checking a contract, requirements become
assumptions and ensures become assertions. When applying a contract as a
simplification in another verification task, requirements become assertions and
ensures become assumptions. This distinction matters because a contract is both
a thing to prove and a thing that can help prove larger properties.

### Explicit Checking Problems

`verif.bmc` represents a bounded model checking problem. It has a `bound`, a
number of externalized registers, an array of initial register values, and three
regions: `init`, `loop`, and `circuit`. The conversion to SMT unrolls this
structure into an MLIR program that asks whether the checked property can fail
within the bound.

`verif.lec` represents logical equivalence checking between two circuit
regions. The two regions must have matching input and output signatures. It can
optionally produce an `i1` result saying whether equivalence has been proven.

`verif.refines` represents refinement checking. The second circuit refines the
first when, for each input, the possible outputs of the second are a subset of
the possible outputs of the first. For deterministic combinational circuits,
this is equivalent to logical equivalence. For nondeterministic circuits,
refinement is stricter and more informative.

`verif.yield` terminates the regions used by `verif.bmc`, `verif.lec`,
`verif.refines`, and `verif.simulation`.

## Operation Inventory

| Operation | Meaning |
| --- | --- |
| `verif.assert` | Requires an LTL or boolean-like property to hold. |
| `verif.assume` | Adds a property that formal checking may assume. |
| `verif.cover` | Asks whether a property can be reached. |
| `verif.clocked_assert` | Checks a property on a specified clock edge. |
| `verif.clocked_assume` | Assumes a property on a specified clock edge. |
| `verif.clocked_cover` | Covers a property on a specified clock edge. |
| `verif.format_verilog_string` | Builds an `!hw.string` using Verilog-style formatting. |
| `verif.print` | Prints a formatted verification or simulation message. |
| `verif.has_been_reset` | Produces an enable-like signal after a proper reset has occurred. |
| `verif.bmc` | Encodes a bounded model checking problem with init, loop, and circuit regions. |
| `verif.lec` | Encodes a logical equivalence checking problem between two circuits. |
| `verif.refines` | Encodes a refinement checking problem between two circuits. |
| `verif.yield` | Yields values from Verif checking or test regions. |
| `verif.formal` | Defines a formal unit test. |
| `verif.simulation` | Defines a simulation unit test. |
| `verif.symbolic_value` | Creates an unconstrained value for formal verification. |
| `verif.contract` | Defines a local formal contract. |
| `verif.require` | Defines a contract precondition. |
| `verif.ensure` | Defines a contract postcondition. |

## Transformation And Conversion Inventory

| Pass | Role |
| --- | --- |
| `combine-assert-like` | Conjoins multiple asserts or assumptions in a block into one assert and one assumption. |
| `fold-assume` | Folds the block assumption into assertion enables, after `combine-assert-like`. |
| `prepare-for-formal` | Prepares a top-level `hw.module` for formal verification. |
| `verify-clocked-assert-like` | Validates clocked assert, assume, and cover operations. |
| `verif-lower-tests` | Converts `verif.formal` and `verif.simulation` into top-level `hw.module` tests. |
| `lower-contracts` | Applies contracts in modules and emits formal tests that check them. |
| `verif-lower-symbolic-values` | Lowers `verif.symbolic_value` into blackbox instances or symbolic wires. |
| `simplify-assume-eq` | Replaces symbolic values that are assumed equal to other values. |
| `strip-contracts` | Removes `verif.contract` ops by replacing them with their operands. |
| `lower-ltl-to-core` | Converts LTL and Verif operations to Core for supported overlapping properties. |
| `lower-verif-to-sv` | Converts Verif constructs to SystemVerilog-oriented operations. |
| `convert-verif-to-smt` | Converts Verif operations to SMT operations for solver-backed checking. |

## Lowering Paths

The SystemVerilog path is used when verification intent should become emitted
hardware verification code. `lower-verif-to-sv` converts `verif.assert`,
`verif.assume`, `verif.cover`, and their clocked variants into SV property
operations. It also lowers `verif.print` and `verif.has_been_reset` into
SystemVerilog-oriented constructs. This path is natural when the output is a
SystemVerilog design or testbench that another simulator or formal tool will
consume.

The SMT path is used when the compiler wants to build a solver problem inside
MLIR. `convert-verif-to-smt` lowers assertions, assumptions, logical
equivalence checks, refinement checks, and bounded model checks into SMT,
`scf`, `arith`, and `func` operations. The pass has a `rising-clocks-only`
option that affects how `verif.bmc` is lowered.

The LTL-to-Core path is narrower. `lower-ltl-to-core` can convert LTL and Verif
constructs to Core when the properties fit the supported form, such as
overlapping properties without delays.

## How To Read Verif IR

A small immediate assertion looks like this:

```mlir
verif.assert %ok label "output_matches" : i1
```

Read this as: when the checker reaches this operation, `%ok` is expected to be
true. The label gives the property a useful diagnostic name.

A guarded assertion looks like this:

```mlir
verif.assert %ok if %enabled label "after_reset" : i1
```

Read this as: `%ok` only needs to hold when `%enabled` is true. A common source
of `%enabled` is `verif.has_been_reset`.

A formal test often creates symbolic inputs and checks a design instance:

```mlir
verif.formal @AdderTest {} {
  %a = verif.symbolic_value : i8
  %b = verif.symbolic_value : i8
  %sum = hw.instance "dut" @Adder(a: %a: i8, b: %b: i8) -> (sum: i8)
  %expected = comb.add %a, %b : i8
  %ok = comb.icmp eq %sum, %expected : i8
  verif.assert %ok label "adder_matches_comb_add" : i1
}
```

The formal engine is free to choose `%a` and `%b`. If any choice makes `%ok`
false, the assertion fails and the tool can report a counterexample.

## What It Implies

Seeing `verif` in IR means the compiler is preserving a proof, test, or
checking problem, not merely optimizing implementation logic. Transformations
must be careful with these operations because changing an assumption or
assertion changes what is being verified.

An assertion increases the obligations of the design. An assumption restricts
the environment. A cover checks reachability. A symbolic value opens the search
space. A contract states a local interface between requirements and guarantees.
A conversion pass chooses the backend vocabulary, such as SystemVerilog
properties or SMT formulas.

For beginners, the safest mental model is this: implementation dialects say
what hardware does; the `verif` dialect says what must be true about that
hardware and how tools should check it.

## Common Pitfalls

Do not treat assumptions as harmless comments. An invalid assumption can hide a
real bug by removing the failing behavior from the formal search space.

Do not forget reset behavior. Many hardware assertions are only meaningful after
initialization. `verif.has_been_reset` exists because simply disabling during
reset is often not enough to cover startup and power-cycle behavior.

Do not mix clocking responsibilities. If you use `verif.clocked_assert`,
`verif.clocked_assume`, or `verif.clocked_cover`, the property should not also
embed its own nested clock or disable structure.

Do not expect every `verif` op to survive to the final output. Many are
intermediate verification constructs. Some lower to SV, some lower to SMT, some
lower to test modules, and some are stripped after they have been applied.


# CIRCT LTL Dialect

The `ltl` dialect is CIRCT's dialect for Linear Temporal Logic, temporal
sequences, and temporal properties. It is mainly used for hardware verification:
assertions, assumptions, coverage, and formal checks that talk about behavior
over time instead of only values in one cycle.

For a beginner, the most important idea is that normal hardware IR usually
describes what a circuit computes now. LTL describes what must happen across
cycles. A property might say that if a request is seen now, an acknowledge must
eventually appear, or that a signal must remain true until another signal fires.
The `ltl` dialect gives CIRCT a structured way to represent those statements
inside MLIR.

## Why This Dialect Exists

SystemVerilog Assertions are the common hardware-verification language for
temporal behavior, but SVA has a lot of syntax, special cases, and legacy
surface details. CIRCT's LTL dialect is not intended to be a direct SVA syntax
tree. Its purpose is to capture the core temporal formalism underneath SVA in a
clean IR.

This matters because hardware tools need to do more than print assertions back
out. They may canonicalize temporal formulas, lower simple pieces to gates,
preserve rich properties for Verilog emission, or feed formal tooling. A
dedicated LTL dialect gives these tools typed operations for sequences,
properties, delays, repetitions, implications, clocks, and temporal
quantifiers.

The dialect depends on CIRCT's `hw` and `comb` dialects because temporal
properties are usually built over hardware values. It also works closely with
the `verif` dialect, whose assertion and assumption operations can carry
`!ltl.sequence` and `!ltl.property` values.

## Types

The dialect defines two explicit types.

| Type | Meaning |
| --- | --- |
| `!ltl.sequence` | A sequence of observations over time, similar to a regular expression over clock cycles. |
| `!ltl.property` | A verifiable temporal claim built from sequences and property operators. |

An `i1` value is also accepted where a sequence is expected: a boolean is a
sequence that holds in the current cycle. An `i1` or `!ltl.sequence` value is
also accepted where a property is expected. This makes simple boolean
assertions, sequence assertions, and full temporal properties fit into the same
verification flow.

The difference is conceptual. A sequence matches behavior. A property states
that behavior must satisfy a temporal claim.

## Operation Inventory

The current `ltl` dialect defines 17 operations.

| Operation | Role |
| --- | --- |
| `ltl.and` | Conjunction over booleans, sequences, or properties. |
| `ltl.or` | Disjunction over booleans, sequences, or properties. |
| `ltl.intersect` | Intersection that requires operands to hold with the same start and end times. |
| `ltl.clock` | Associates a sequence or property tree with a clock edge. |
| `ltl.delay` | Delays a sequence by a fixed, bounded, or unbounded finite number of cycles. |
| `ltl.clocked_delay` | Delays a sequence using an explicit clock and edge. |
| `ltl.past` | Observes a hardware value a fixed number of clock cycles earlier. |
| `ltl.sampled` | Models SystemVerilog sampled-value semantics for assertion action contexts. |
| `ltl.concat` | Concatenates sequences into a longer sequence. |
| `ltl.repeat` | Repeats a sequence consecutively a fixed, bounded, or unbounded finite number of times. |
| `ltl.goto_repeat` | Non-consecutive go-to style repetition where the final counted evaluation must match. |
| `ltl.non_consecutive_repeat` | Non-consecutive repetition where the final evaluation does not have to match. |
| `ltl.boolean_constant` | Produces a constant boolean `!ltl.property`. |
| `ltl.not` | Negates a property. |
| `ltl.implication` | Checks a consequent property when an antecedent sequence matches. |
| `ltl.until` | Weak until: one property holds until another property holds. |
| `ltl.eventually` | Strong eventually: a property must hold after a finite number of cycles. |

The dialect also defines a clock-edge attribute with the spelling `posedge`,
`negedge`, or `edge`.

## Sequences

Sequences are the part of the dialect that model temporal matching. A plain
boolean value is a one-cycle sequence. `ltl.delay` moves a sequence into the
future. For example, a fixed delay means the input sequence must match exactly
that many cycles later. A delay with a length gives a bounded window. A delay
without a length is unbounded but still finite, meaning the match must happen
eventually after the base delay.

`ltl.concat` combines sequences. It is important that concatenation itself does
not add an implicit clock cycle. It aligns the end of one sequence with the
start of the next. If the intended meaning is "the next sequence starts one
cycle later," that delay must be represented explicitly with `ltl.delay`.

This is one of the dialect's major design choices. SVA writes delays inside
concatenation syntax, such as `a ##1 b`. The LTL dialect separates that into
two concepts: `ltl.delay` for the cycle distance and `ltl.concat` for joining
sequences. That makes zero-length and repeated sequences easier to reason about
in IR.

`ltl.repeat` models consecutive repetition. It can represent exactly N
repetitions, a bounded N-to-M range, or an unbounded finite range. The two
non-consecutive forms, `ltl.goto_repeat` and
`ltl.non_consecutive_repeat`, model SVA-style repetitions where matches may be
separated by cycles that do not match. The go-to form requires the final counted
evaluation to match; the non-consecutive form does not.

## Properties

Properties are temporal claims. `ltl.implication` is the common bridge from a
sequence into a property: if the antecedent sequence matches, then the
consequent property is checked starting at the end point of that match. If the
antecedent does not match, the implication holds vacuously.

The dialect models the overlapping implication form directly. A non-overlapping
SVA implication can be represented by explicitly concatenating a one-cycle delay
before the consequent is checked. This keeps clock-related behavior explicit
instead of hidden inside a separate operator.

`ltl.until` is weak until. It says that the input property holds until the
condition property holds once. Because it is weak, the result can still hold if
the condition never arrives as long as the input keeps holding.

`ltl.eventually` is strong. It requires that the input property hold after a
finite number of cycles. If the input can never hold in the future, the
eventuality does not hold.

`ltl.not`, `ltl.and`, and `ltl.or` provide boolean-style composition at the
property level. `ltl.boolean_constant` gives canonicalization patterns a
property-typed constant.

## Clocking And Sampling

Temporal operators need a notion of cycle. `ltl.clock` associates a sequence or
property expression with a clock and edge. The edge can be `posedge`,
`negedge`, or `edge`. The clock applies to the expression tree below it, except
where another nested clock overrides that context.

`ltl.clocked_delay` is more explicit. It carries the clock and edge directly on
the delay operation. This is useful when a delay needs to be tied to a specific
clock without relying on an enclosing `ltl.clock`.

`ltl.past` observes a hardware integer value a fixed number of cycles earlier
on a clock. In the `lower-ltl-to-core` conversion, this operation becomes a
sequential shift register.

`ltl.sampled` models SystemVerilog sampled-value semantics. This is useful in
concurrent assertion action contexts, where the value read in an action block
must be the same sampled value that triggered the assertion.

## Transformations

Several LTL operations have folders or canonicalizers. These are local
simplifications, not full temporal proof engines.

The declarative fold patterns include:

- merging nested `ltl.delay` operations by adding their delays and combining
  bounded lengths,
- pushing `ltl.delay` into the first element of an `ltl.concat`,
- merging nested `ltl.clocked_delay` operations when the clock and edge match,
- pushing `ltl.clocked_delay` into concatenations,
- flattening nested `ltl.concat` operations.

The operation definitions also mark folders or canonicalizers on operations
such as `ltl.delay`, `ltl.clocked_delay`, `ltl.concat`, `ltl.repeat`,
`ltl.goto_repeat`, `ltl.non_consecutive_repeat`, and
`ltl.boolean_constant`. `ltl.and`, `ltl.or`, and `ltl.intersect` have
canonicalization hooks and infer their result type from their operands: if any
operand is a property the result is a property; otherwise, if any operand is a
sequence the result is a sequence; otherwise the result is `i1`.

These transformations make the IR cleaner and remove obvious structural
redundancy. They do not replace model checking or assertion solving.

## Conversion And Emission

The dedicated conversion pass for this dialect is `lower-ltl-to-core`.

| Pass | What it does |
| --- | --- |
| `lower-ltl-to-core` | Converts selected LTL and Verif operations inside `hw.module` operations to core HW, Comb, SV, and Seq dialect operations. |

This pass is intentionally partial. It can lower boolean-shaped `ltl.and`,
`ltl.or`, `ltl.not`, and `ltl.implication` to `comb` operations when doing so is
legal. For example, boolean implication becomes "not antecedent or consequent."
It lowers `ltl.past` to a `seq.shiftreg` on the given clock. It also handles a
related `verif.has_been_reset` operation. The pass type-converts
`!ltl.sequence` and `!ltl.property` to `i1` where needed for simple assertion
uses, but it does not try to compile every rich temporal property into gates.

The other major path is Verilog emission. CIRCT's ExportVerilog support knows
how to emit LTL expressions as SystemVerilog assertion syntax in appropriate
contexts. For example, `ltl.implication` can emit as an SVA implication, and
repetition operators map to SVA repetition forms. This is why LTL IR may remain
present late in the pipeline: the desired output may be an assertion, not a
hardware circuit implementing the assertion.

## How To Read LTL IR

Start by asking whether a value is a sequence or a property. If it is a
sequence, read it as a temporal pattern to be matched. If it is a property, read
it as a claim about what must hold.

Then look for clocking. A delay without an obvious clock gets its cycle meaning
from surrounding clock context, usually an `ltl.clock` or an assertion context.
An `ltl.clocked_delay` is explicit and can be read locally.

For sequence expressions, read delays and concatenations carefully. `ltl.concat`
does not mean "next cycle" by itself. The next-cycle behavior must appear as a
delay. For property expressions, implication, until, and eventually are the
high-level temporal operators that describe the verification intent.

Finally, look at how the value is used. If it feeds `verif.assert`,
`verif.assume`, or `verif.cover`, it is part of a verification statement. If it
is being lowered by `lower-ltl-to-core`, only the boolean-like parts may become
core gates. If it is being exported, it may turn into SystemVerilog assertion
syntax.

## What It Implies

The presence of `ltl` implies that the IR is carrying verification intent, not
just implementation logic. That intent may need to survive until formal tooling
or Verilog emission. Rewriting it as ordinary boolean logic too early can lose
temporal meaning.

It also implies that type matters. An `i1`, a `!ltl.sequence`, and a
`!ltl.property` may all look like "truth-like" values, but they live at
different levels. A boolean is about one cycle. A sequence is about a match over
time. A property is about a claim that can be asserted, assumed, covered, or
combined with temporal quantifiers.

The main pitfall is assuming that SVA syntax maps one-to-one to one LTL op.
CIRCT deliberately factors SVA constructs into smaller core temporal concepts.
For example, a non-overlapping implication is modeled using explicit delay and
concatenation plus `ltl.implication`, not a separate non-overlap operation.

## Summary

The `ltl` dialect is CIRCT's clean IR for temporal hardware verification. It
models sequences, properties, delays, repetitions, implications, clocking,
past-value sampling, and eventuality. It is important when assertions need to
survive as structured temporal formulas, when SVA-like constructs need a
compiler-friendly representation, and when parts of those formulas must either
lower to core hardware operations or emit as SystemVerilog assertions.

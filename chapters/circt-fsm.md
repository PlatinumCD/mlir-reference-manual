# CIRCT `fsm` Dialect

## Beginner Summary

The CIRCT `fsm` dialect represents finite-state machines explicitly. It gives
states, transitions, transition guards, transition actions, machine inputs,
machine outputs, and internal variables first-class operations.

For a beginner, the central idea is:

```text
fsm.machine
  contains fsm.variable and fsm.state
  each fsm.state has output logic and transition logic
  each fsm.transition can have a guard and an action
```

This is useful because finite-state-machine structure is often obvious to a
designer but hard to recover from low-level RTL. The FSM dialect preserves that
structure so CIRCT can analyze it, print it, transform it, lower it to hardware,
lower it to software-like forms, or translate it into SMT for checking.

## Why This Dialect Exists

FSMs are common in controllers, protocols, firmware models, verification
models, hardware IP blocks, and software drivers. A compiler can represent the
same behavior as arbitrary registers and muxes, but then the state-machine
meaning is hidden. The `fsm` dialect exists so the compiler can keep the
structure explicit.

The dialect is target-agnostic. It can be instantiated in a hardware-style
context with clock and reset, or in a software-style context where an instance
is triggered sequentially. This lets the same abstract state-machine model sit
between source descriptions and several downstream targets.

The rationale document describes three goals: explicit representation of
states, transitions, and variables; target-agnostic instantiation; and lowering
paths to hardware, software-style core IR, and verification-oriented forms.

## When It Matters

You will see `fsm` when a flow wants to preserve or recover state-machine
structure. It matters when:

- a controller is easier to understand as states and transitions than as muxes;
- a Calyx control schedule is being represented as a machine;
- an RTL netlist is being analyzed to extract FSM structure;
- a machine should be emitted as SystemVerilog hardware;
- a machine should be translated into SMT for safety checking;
- a graph view of state transitions is useful for debugging.

The dialect is not meant to replace all sequential hardware IR. Use `seq` for
registers and sequential primitives, and use `fsm` when the state-machine
structure itself is important.

## Core Model

`fsm.machine` defines a machine. It has a symbol name, an initial state name,
a function type for inputs and outputs, optional port names, and a body region.
The body contains `fsm.state` operations and may contain `fsm.variable`
operations.

`fsm.state` defines one named state. It may have an `output` region that
computes the machine outputs for that state and terminates with `fsm.output`.
It may also have a `transitions` region containing `fsm.transition` operations.

`fsm.transition` names the next state. It may have a `guard` region terminated
by `fsm.return` with an i1 condition. If the guard is absent or empty, the
transition is always taken. When multiple transitions exist, they are
prioritized in the order they appear. A transition may also have an `action`
region, where side-effecting transition work can happen.

`fsm.variable` declares an internal variable associated with each FSM instance.
`fsm.update` updates such a variable in a transition action region. This avoids
state explosion: not every piece of extended state has to become a distinct FSM
state.

The dialect defines one type, `!fsm.instance`, used by software-style
instantiation.

## Operation Inventory

The current `fsm` dialect defines ten operations:

```text
fsm.hw_instance
fsm.instance
fsm.machine
fsm.output
fsm.return
fsm.state
fsm.transition
fsm.trigger
fsm.update
fsm.variable
```

`fsm.machine` is the top-level state-machine definition.

`fsm.state` defines a named state inside a machine.

`fsm.output` terminates a state's output region and returns the values produced
by the machine while it is in that state.

`fsm.transition` defines an outgoing transition from a state to another state.
Its guard chooses whether it is taken, and its action performs updates.

`fsm.return` terminates a transition guard region and returns the guard
condition.

`fsm.variable` declares internal extended state with an initialization value.

`fsm.update` writes a new value to a variable from a transition action region.

`fsm.instance` creates a software-style instance and produces a `!fsm.instance`
value.

`fsm.trigger` triggers a software-style instance once, potentially advancing
the machine by one transition and producing outputs.

`fsm.hw_instance` creates a hardware-style instance with inputs, outputs, clock,
and reset.

## Instantiation Modes

The dialect has two ways to instantiate a machine because hardware and software
contexts have different execution models.

In a software-style context, `fsm.instance` creates an instance value and
`fsm.trigger` explicitly advances it. Each trigger represents one opportunity
for the machine to take a transition.

In a hardware-style context, `fsm.hw_instance` consumes inputs and produces
outputs continuously, with clock and reset operands. It is intended for lowering
into HW, Comb, Seq, and SV-style hardware.

This distinction is important. A state machine used as software control flow
needs explicit trigger points. A state machine used as hardware is naturally
clocked and continuously connected.

## Guards, Actions, And Variables

Transition guards describe whether a transition is enabled. Guards are
side-effect-free regions that return an i1. If several transitions are present
in a state's transition region, the first enabled transition wins.

Transition actions describe what happens when a transition is taken. Actions may
update variables using `fsm.update`, and they may contain other operations that
the relevant lowering path supports.

Variables provide extended state. Instead of creating a separate FSM state for
every counter value or data value, the machine can keep a variable such as
`cnt`, update it on transitions, and use it in later guards or output logic.

## Transformations And Conversions

The FSM dialect's own pass is:

```text
fsm-print-graph
```

`fsm-print-graph` builds an FSM graph for each `fsm.machine` and prints a DOT
graph. This is a debugging and visualization pass, not a lowering pass.

The main conversion passes involving FSM are:

```text
lower-calyx-to-fsm
materialize-calyx-to-fsm
calyx-remove-groups-fsm
convert-core-to-fsm
convert-fsm-to-core
convert-fsm-to-sv
convert-fsm-to-smt
```

`lower-calyx-to-fsm` lowers a Calyx control schedule into an intermediate
`fsm.machine` inside a Calyx component. `materialize-calyx-to-fsm` gives that
machine top-level group go/done I/O. `calyx-remove-groups-fsm` outlines the
machine and removes the original Calyx groups.

`convert-core-to-fsm` extracts FSM structure from an RTL netlist made of core
hardware dialects. It can use explicitly listed state registers or infer state
registers from names containing `state`.

`convert-fsm-to-core` lowers FSMs to HW, Comb, and Seq-style core hardware IR.
`convert-fsm-to-sv` lowers FSMs to SystemVerilog-oriented hardware, including
SV constructs and emitted enum typedef fragments. `convert-fsm-to-smt` lowers
FSMs into SMT models that can be used for safety-property checking; it can also
include an explicit time parameter.

## What The Dialect Implies

Seeing `fsm` means state-machine structure is meant to remain visible. The
states are named, transitions are ordered, guards are explicit, and actions are
separate from output logic. This makes the IR easier to reason about than a
flat pile of state registers and muxes.

It also implies that downstream lowering choices are still open. The same
machine may be used as a hardware instance, triggered in a software-like
context, lowered to core hardware, emitted through SV, or converted to SMT for
verification.

The tradeoff is that not every arbitrary sequential circuit is naturally an FSM
at this level. If the structure cannot be extracted cleanly, or if reset and
state-register assumptions do not fit the conversion pass, a lower-level
sequential representation may be more appropriate.

## How To Use It In Practice

When reading FSM IR, start with `fsm.machine` and note its `initialState`
attribute, inputs, outputs, and variables. Then read each `fsm.state`: first its
output region, then its transitions. For each transition, check the next-state
symbol, guard condition, and action updates.

For debugging behavior, remember that transition order is priority order. An
earlier true guard prevents later transitions from being considered.

For lowering questions, identify the intended target. Hardware emission usually
uses `fsm.hw_instance` and `convert-fsm-to-sv` or `convert-fsm-to-core`.
Software-style triggering uses `fsm.instance` and `fsm.trigger`. Verification
work may use `convert-fsm-to-smt`.

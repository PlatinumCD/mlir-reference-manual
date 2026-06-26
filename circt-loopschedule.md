# CIRCT LoopSchedule Dialect

The `loopschedule` dialect is a CIRCT dialect for representing loops after a
scheduler has assigned operations to cycles. It sits between software-like loop
IR, such as MLIR's `affine` dialect, and hardware-oriented IR, especially the
Calyx dialect. Its job is to preserve the structure of a loop while also making
the schedule visible.

For a beginner, the key idea is this: ordinary loop IR says what repeats, but
scheduled loop IR also says when each piece of the loop happens. In hardware
compilation, that timing matters. A loop may be pipelined so that several
iterations are active at the same time. One iteration might be in an early
stage, the previous iteration might be in a later stage, and another iteration
might be starting. The `loopschedule` dialect gives CIRCT a way to represent
that shape before lowering it into explicit hardware control.

## Why This Dialect Exists

Hardware synthesis needs more information than a normal software loop provides.
Consider a simple loop with a load, an arithmetic operation, and a store. In
software IR, these operations are mostly ordered by the loop body. In hardware
IR, the compiler needs to decide which cycle each operation starts in, how long
values must be held, which values move through registers, and how quickly new
loop iterations can start.

The `loopschedule` dialect exists to capture those scheduling decisions without
immediately flattening the loop into low-level hardware. It keeps the loop
recognizable. A `loopschedule.pipeline` still looks like a structured loop. Its
stages still contain the operations for that loop. Its terminator still says
which values become next-iteration loop-carried values and which values are
returned by the whole pipeline.

This is useful because many hardware compilers need an intermediate point
between "unscheduled program" and "fully expanded control circuit." At that
point, it is helpful to inspect, verify, and lower a schedule as a schedule.

## When It Is Important

The dialect is important when CIRCT is doing high-level synthesis style lowering
from loop programs into hardware. It is especially relevant when:

- an `affine.for` loop has been analyzed and scheduled,
- a loop is going to become a pipelined hardware structure,
- multiple loop iterations may be in flight at the same time,
- loop-carried values must be routed through stage registers,
- the next target is Calyx rather than direct Verilog-like IR.

You should not think of `loopschedule` as a frontend authoring dialect for
general MLIR users. It is mainly a compiler-internal representation. You will
see it when debugging CIRCT scheduling and Calyx lowering, or when studying how
CIRCT represents pipelined loops before it builds hardware control.

## Core Model

The dialect's current implementation is centered on pipelined loops. CIRCT's
rationale document discusses pipelined and sequential scheduled loops as design
concepts, but the current operation inventory is the pipeline representation.

A pipelined loop has an initiation interval, written as `II`. The initiation
interval is the number of cycles between the start of successive loop
iterations. If `II = 1`, the pipeline can start a new iteration every cycle. If
`II = 3`, it starts a new iteration every three cycles.

Inside the pipeline, operations are grouped into stages. Each
`loopschedule.pipeline.stage` has a `start` attribute that gives the stage's
start cycle. Values computed in a stage are passed out through
`loopschedule.register`, which models the fact that hardware needs registers to
carry values forward across cycles. The pipeline as a whole ends with
`loopschedule.terminator`, which says which registered values become the next
iteration arguments and which become the pipeline results.

This gives the dialect a two-level structure:

- the pipeline describes the loop-level schedule,
- each stage describes the operations assigned to one start time.

## Operation Inventory

The current `loopschedule` dialect defines four operations.

| Operation | Role |
| --- | --- |
| `loopschedule.pipeline` | Top-level scheduled loop operation with an initiation interval, optional trip count, condition region, stages region, optional iteration arguments, and optional results. |
| `loopschedule.pipeline.stage` | A scheduled stage inside a pipeline, marked with a non-negative `start` cycle and an optional `when` predicate. |
| `loopschedule.register` | Terminator for the pipeline condition region or for a pipeline stage; registers values so they become stage results or the pipeline condition value. |
| `loopschedule.terminator` | Terminator for the pipeline stages region; supplies next-iteration `iter_args` and final pipeline results. |

These four operations are enough to describe the scheduled loop skeleton. The
actual computation inside a stage usually comes from other dialects, such as
`arith` and `memref`.

## `loopschedule.pipeline`

`loopschedule.pipeline` is the root operation of the dialect. It represents a
statically scheduled pipeline loop. It has an `II` attribute, an optional
`trip_count` attribute, optional iteration arguments, two regions, and optional
results.

The first region is the condition region. It is a single-block region that
computes whether the pipeline should keep starting new iterations. The verifier
restricts this region to a combinational body. In the current implementation,
that means a limited set of arithmetic, comparison, cast, shift, select, and
similar operations. The region must terminate with `loopschedule.register`, and
that register must carry exactly one `i1` value.

The second region is the stages region. This region contains the pipeline
stages and ends with `loopschedule.terminator`. The verifier requires the stages
region to contain at least one stage, and it only permits
`loopschedule.pipeline.stage` and `loopschedule.terminator` operations directly
inside the stages block.

The pipeline may also have `iter_args`. These are like loop-carried values in
`scf` or `affine`, but they are interpreted through the schedule. Their next
values are supplied by the pipeline terminator. This is how the dialect models
induction variables, reductions, and other values that flow from one iteration
to the next.

## `loopschedule.pipeline.stage`

`loopschedule.pipeline.stage` represents work that starts at a particular cycle
relative to the start of a pipeline iteration. Its `start` attribute must be
non-negative, and stages inside a pipeline must appear in monotonically
increasing start-time order.

The stage body is a single-block region. Operations inside the region are the
computations assigned to that stage. The body ends with `loopschedule.register`.
The values passed to that register become the stage results, and the verifier
requires those operand types to match the stage result types.

A stage may also have an optional `when` predicate. This predicate allows
conditional stage execution. If the predicate is false, the stage represents a
bubble moving through the pipeline rather than valid work. This is separate from
the pipeline condition, which controls whether new iterations are initiated.

Stages are intentionally constrained. They may use values that dominate the
pipeline, pipeline iteration arguments, or values produced by earlier stages.
That restriction preserves a coarse-grained schedule: data flows from earlier
scheduled times to later scheduled times, and lowering passes are responsible
for preserving values across the needed cycles.

## `loopschedule.register`

`loopschedule.register` is a terminator used in two places. In the condition
region of a pipeline, it registers the condition value that controls whether new
iterations start. In a pipeline stage, it registers the values computed by that
stage so those values can be consumed by later stages, the pipeline terminator,
or future iterations.

The name is intentionally hardware-flavored. In software SSA, a value can simply
be used later if it dominates the use. In hardware, a value that must survive
from one cycle to another usually needs storage. `loopschedule.register` marks
the boundary where values leave one scheduled stage and become available as
registered stage results.

The operation does not by itself choose a concrete register implementation.
That happens during lowering. At the `loopschedule` level, the important fact is
that the value crosses a scheduled boundary.

## `loopschedule.terminator`

`loopschedule.terminator` ends the stages region of a pipeline. It has two
lists: `iter_args` and `results`.

The `iter_args` list gives the values for the next loop iteration. Their types
must match the pipeline's iteration argument types, and each value must be
defined by a `loopschedule.pipeline.stage`. The `results` list gives the values
returned by the pipeline. Their types must match the pipeline result types, and
they must also come from stages.

This means the terminator is not just a syntactic end marker. It describes the
feedback and output behavior of the pipeline. Even though it appears at the end
of the stages region, the values it names may need to be available earlier than
the final stage in the next iteration. Lowering passes must preserve those
scheduled relationships.

## Transformations And Conversions

There is no separate dialect-local pass file for `loopschedule` in this tree.
The relevant passes are CIRCT conversion passes.

| Pass | Direction | What it does |
| --- | --- | --- |
| `convert-affine-to-loopschedule` | Into `loopschedule` | Converts suitable `affine` loops into scheduled `loopschedule.pipeline` operations. |
| `lower-loopschedule-to-calyx` | Out of `loopschedule` | Lowers `loopschedule` pipelines into Calyx hardware components and control. |

`convert-affine-to-loopschedule` runs on `func.func`. It analyzes affine loops
and control flow, uses dependence and cyclic scheduling analyses, constructs a
modulo scheduling problem, solves it, and creates a
`loopschedule.pipeline`. In the current implementation, it works on perfectly
nested loops but only converts single-loop nests. It lowers affine load and
store indexing to explicit memref operations, creates an induction-variable
iteration argument, computes a loop condition, groups operations by scheduled
start time, creates stages, and wires registered values into later stages and
the terminator.

`lower-loopschedule-to-calyx` runs on a module. It finds or requires a top-level
function, labels the entry point, creates Calyx components, converts index types
to fixed-width integers, builds registers for block arguments, return values,
while iteration values, and pipeline stages, builds Calyx groups for operations
and pipeline stages, generates control, performs late SSA replacement, rewrites
memory accesses, removes converted functions, and cleans up Calyx control
groups. In other words, it turns the scheduled loop structure into Calyx's
hardware-oriented component and control representation.

## How To Read LoopSchedule IR

When reading `loopschedule` IR, start with the pipeline header. The `II` tells
you how frequently new iterations can start. A `trip_count`, when present,
gives a constant loop trip count that lowering can use. The `iter_args` tell you
which values are carried from one iteration to another.

Next, read the condition region. It should compute a single boolean. In a loop
converted from `affine`, this is often an induction-variable comparison against
an upper bound.

Then read the stages in order. Each `loopschedule.pipeline.stage` has a `start`
time. Values registered by a stage appear as that stage's results. Later stages
use those results rather than directly reaching back into earlier stage bodies.
This is the visible data path of the schedule.

Finally, read `loopschedule.terminator`. It tells you which stage results feed
back into the next iteration and which are returned by the pipeline. If you are
debugging incorrect hardware behavior, this terminator is often one of the most
important places to inspect because it records the loop-carried and output
edges.

## What It Implies

The presence of `loopschedule` implies that CIRCT has already made scheduling
decisions. The IR is no longer just describing a loop algorithm. It is
describing an algorithm plus a timing plan.

It also implies that many ordinary loop rewrites are no longer safe. The
operation definition explicitly notes that a pipeline captures the result of
scheduling and is not generally safe to transform except for lowering to
hardware dialects. Moving operations between stages, changing stage order, or
rewiring registered values would change the schedule and may change the
hardware.

Another implication is that operation latency matters, even if it is not always
explicitly represented in the IR. The conversion from Affine to LoopSchedule
uses an operator library and scheduling problem to decide where operations
belong. The rationale document notes that the pipeline representation is still
tightly coupled to lowering assumptions about operator latency. That is a useful
warning for readers: a start time is meaningful only in the context of the
latency model used by the scheduler and lowering pipeline.

## Summary

The `loopschedule` dialect is CIRCT's structured representation for scheduled
loops, especially pipelined loops. It uses `loopschedule.pipeline` to hold the
loop, `loopschedule.pipeline.stage` to group operations by cycle,
`loopschedule.register` to carry values across scheduled boundaries, and
`loopschedule.terminator` to describe feedback and results. Its main conversion
path is from Affine into `loopschedule`, then from `loopschedule` into Calyx.
For beginners, it is best understood as the point where a loop stops being only
a repeated program and starts being a hardware pipeline schedule.

# CIRCT `synth` Dialect

## Beginner Summary

The CIRCT `synth` dialect represents logic during synthesis.

In hardware compilation, "synthesis" means turning a circuit into a better,
lower-level logic network that can be mapped to gates, lookup tables, or other
implementation resources. The `synth` dialect is CIRCT's place for representing
that synthesis-oriented logic directly.

The most important operation is `synth.aig.and_inv`, an and-inverter graph
node. It represents an AND whose inputs may be individually inverted. This is a
classic form used by logic synthesis tools because many Boolean functions can be
normalized, optimized, and mapped efficiently as AIGs.

The dialect also includes richer three-input gates such as `synth.majority`,
`synth.dot`, `synth.mux_inv`, `synth.onehot`, and `synth.gamble`, plus meta
operations used by synthesis algorithms such as `synth.choice` and
`synth.cut_rewrite_pattern`.

## Why This Dialect Exists

CIRCT already has the `comb` dialect for combinational hardware operations such
as add, and, or, xor, compare, concat, and extract. That is a good structural
hardware IR, but it is not always the best representation for synthesis
algorithms.

Logic synthesis often wants a graph of very small Boolean primitives with
domain-specific properties:

- inputs can be inverted without adding separate NOT nodes;
- equivalent choices can be represented explicitly;
- Boolean nodes can report area and delay costs;
- cut enumeration can rewrite a subgraph using a cheaper implementation;
- external tools such as ABC can consume and return AIG-like networks;
- timing and resource analyses can reason about the synthesis graph.

The `synth` dialect exists to hold those ideas. It is lower level than `comb`
in some ways, but higher level than a final technology netlist because it can
still contain variadic and word-level forms used for scalable optimization.

## When It Matters

You will usually see `synth` in the middle of a CIRCT logic synthesis flow.

A typical path looks like this:

```text
hw + comb
  -> comb and datapath cleanup
  -> convert-comb-to-synth
  -> synth AIG / three-input / choice operations
  -> synth optimization passes
  -> optional ABC or AIGER round trip
  -> convert-synth-to-comb for verification or further lowering
```

The `circt-synth` tool uses this style of flow. Its documentation describes
lowering combinational HW and Comb IR to an AIG represented by the `synth`
dialect, with an option to optimize for area or timing.

## When To Use It

Use `synth` when writing or debugging CIRCT logic synthesis passes.

Use it for:

- AIG-style and-inverter graphs;
- majority-inverter and other three-input logic representations;
- optimizing area and timing with cut rewriting;
- representing proven equivalent choices;
- lowering word-level logic to bit-level logic;
- structural hashing of synthesis graphs;
- mapping logic to LUTs or technology cells;
- invoking external AIGER or ABC-based flows;
- running longest-path and resource-usage analyses.

Do not use `synth` as a frontend hardware dialect. Frontends normally emit
`hw`, `comb`, `seq`, `sv`, or other higher-level CIRCT dialects.

Do not use it as a complete final physical netlist. It is an optimization and
mapping dialect, not a full placement, routing, or cell-library netlist format.

## Core Concepts

### Invertible Operands

Most important Synth logic operations use invertible operands. Instead of
building a separate NOT operation before a gate, the operation stores an
inversion flag for each input.

For example:

```mlir
%0 = synth.aig.and_inv not %a, %b : i1
```

This means "AND the inverse of `%a` with `%b`." The inversion is part of the
gate representation. This is useful because AIGs are defined around AND nodes
with optional input inversions.

### AIG And MIG

An AIG, or and-inverter graph, represents Boolean logic with AND nodes and
input inversions. CIRCT's `synth.aig.and_inv` is the AIG node.

A MIG, or majority-inverter graph, represents Boolean logic with majority nodes
and inversions. CIRCT's `synth.majority` is the main MIG-style node.

The dialect also includes experimental or specialized three-input gates that
can help synthesis algorithms search for smaller or faster logic networks.

### Variadic And Word-Level Forms

The final AIG representation may need binary single-bit nodes, but it is often
more efficient to keep larger forms temporarily. `synth.aig.and_inv` and
`synth.xor_inv` can be variadic and can operate on integer values wider than
one bit.

Later passes such as `synth-lower-variadic` and `synth-lower-word-to-bits`
lower these scalable forms toward smaller pieces.

### Choices

`synth.choice` means the operands are functionally equivalent alternatives.
A later pass may select any one of them. This is useful when an optimization
discovers multiple implementations of the same logic and wants to preserve that
freedom for mapping or timing decisions.

### Cut Rewriting

A cut is a small subgraph boundary. A cut-rewriting pass enumerates possible
cuts, evaluates replacement patterns, and chooses a better implementation based
on area or timing.

`synth.cut_rewrite_pattern` and `synth.yield` provide a declarative way to
describe rewrite patterns and their costs.

## Types And Attributes

The inspected Synth source does not define custom Synth types. The operations
work mostly over integer-like hardware values, including `i1` and wider integer
types.

Synth does define several attributes used by timing and mapping:

| Attribute | Meaning |
| --- | --- |
| `#synth.nldm_table` | Resolved 2D nonlinear delay model lookup table. |
| `#synth.nldm_timing_arc` | NLDM timing arc with delay and transition lookup tables. |
| `#synth.polarity` | Timing arc polarity, either positive or negative. |
| `#synth.linear_timing_arc` | Simplified linear timing arc. |
| `#synth.mapping_cost` | Area and timing cost used by technology mapping and cut rewriting. |

## Operation Inventory

The Synth dialect currently defines 10 operations.

| Operation | Purpose |
| --- | --- |
| `synth.aig.and_inv` | AIG AND node with optional per-input inversion; may be variadic and word-level before later lowering. |
| `synth.xor_inv` | XOR/parity node with optional per-input inversion. |
| `synth.dot` | Ordered three-input Boolean gate for area-oriented synthesis research. |
| `synth.choice` | Marks multiple functionally equivalent values as legal alternatives. |
| `synth.majority` | Three-input majority gate with optional per-input inversion. |
| `synth.onehot` | Three-input one-hot gate with optional per-input inversion. |
| `synth.mux_inv` | Three-input multiplexer gate with optional per-input inversion. |
| `synth.gamble` | Three-input gate true when all inputs agree, with optional per-input inversion. |
| `synth.cut_rewrite_pattern` | Declarative rewrite pattern for cut-based mapping. |
| `synth.yield` | Terminator for Synth pattern regions. |

## Transformations And Conversions

Synth has a small operation set but a large pass surface.

### Conversion Passes

| Pass | What it does |
| --- | --- |
| `convert-comb-to-synth` | Lowers Comb operations to Synth operations, usually toward AIG form. |
| `convert-synth-to-comb` | Lowers Synth operations back to Comb, mainly for verification of post-synthesis results. |

`convert-comb-to-synth` has options for additional legal Comb operations,
forcing AIG-only lowering, and controlling how many unknown bits to emulate in
table lookup lowering.

### Synth Transform And Analysis Passes

| Pass | What it does |
| --- | --- |
| `synth-test-priority-cuts` | Tests priority-cut enumeration for synthesis. |
| `synth-tech-mapper` | Performs technology mapping with cut rewriting, NPN forms, and area/timing strategy. |
| `synth-generic-lut-mapper` | Maps logic to generic K-input LUTs represented with `comb.truth_table`. |
| `synth-lower-variadic` | Lowers variadic operations to binary operations, optionally using timing-aware balancing. |
| `synth-lower-word-to-bits` | Lowers multi-bit and-inverter operations to single-bit operations. |
| `synth-print-longest-path-analysis` | Reports longest-path timing information through Synth/AIG logic. |
| `synth-print-resource-usage-analysis` | Reports local and hierarchical resource usage. |
| `synth-aiger-runner` | Exports to AIGER, runs an external solver, and imports the result. |
| `synth-abc-runner` | Runs ABC over AIGER files as a wrapper around the AIGER runner. |
| `synth-structural-hash` | Performs structural hashing and synthesis-aware CSE. |
| `synth-maximum-and-cover` | Collapses single-fanout and-inverter nodes into users where profitable. |
| `synth-sop-balancing` | Uses sum-of-products balancing for delay optimization. |
| `synth-functional-reduction` | Uses simulation and SAT solving to prove equivalent values and materialize `synth.choice`. |

### Registered Pipelines

The Synth source registers two named pipelines:

| Pipeline | Purpose |
| --- | --- |
| `synth-comb-lowering-pipeline` | Default pipeline for lowering Comb-oriented logic toward Synth. |
| `synth-optimization-pipeline` | Default AIG optimization pipeline. |

The optimization pipeline can lower word-level logic to bits, lower variadic
ops, run structural hashing, run maximum-and-cover, optionally run SOP
balancing, optionally run functional reduction, and optionally run ABC.

## How To Read Synth IR

Start by looking for inversion markers.

```mlir
%0 = synth.aig.and_inv not %a, %b : i1
```

This is a single AIG node. It is not a separate NOT followed by an AND. The
input inversion is part of the node.

Next, check whether the operation is structural logic or a synthesis decision.
`synth.aig.and_inv`, `synth.xor_inv`, and `synth.majority` compute Boolean
functions. `synth.choice` does not compute a new function in the usual sense;
it records equivalent options. `synth.cut_rewrite_pattern` is not design logic;
it is pattern data for mapping.

Finally, check the bit width. A `synth.aig.and_inv` over `i8` is a word-level
batch representation. A later pass may split it into eight `i1` nodes.

## What It Implies

Using Synth implies that the design is in a synthesis-oriented phase.

It also implies that later passes may legally restructure the logic as long as
they preserve Boolean behavior. That is different from a source-level hardware
dialect where preserving readable structure may still matter.

Important consequences:

- operation shape is chosen for optimization, not source readability;
- inversion flags are semantically meaningful and should not be ignored;
- choices may be replaced by any equivalent operand;
- wide or variadic nodes may later become bit-level binary nodes;
- mapping costs and timing arcs can guide replacement decisions;
- external tools such as ABC may rewrite the network.

Synth is therefore the dialect to study when you want to understand how CIRCT
turns hardware logic into a synthesis graph and improves it before final
lowering or verification.

## Source Files Inspected

This chapter was written from the local CIRCT source:

- `circt/include/circt/Dialect/Synth/Synth.td`
- `circt/include/circt/Dialect/Synth/SynthOps.td`
- `circt/include/circt/Dialect/Synth/SynthAttributes.td`
- `circt/include/circt/Dialect/Synth/SynthEnums.td`
- `circt/include/circt/Dialect/Synth/Transforms/SynthPasses.td`
- `circt/include/circt/Dialect/Synth/Transforms/SynthesisPipeline.h`
- `circt/lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp`
- `circt/include/circt/Conversion/Passes.td`
- `circt/docs/Dialects/Synth/RationaleSynth.md`
- `circt/docs/Tools/circt-synth.md`

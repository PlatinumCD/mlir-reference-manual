# The External MLIR Ecosystem

MLIR is not only the dialects that live in the upstream LLVM repository. One of
its main design goals is extensibility: projects can define their own dialects,
passes, conversions, and tools while still using the same core IR
infrastructure.

That is why the ecosystem around MLIR is so broad. Framework compilers,
hardware compilers, runtime-oriented compilers, and domain-specific research
systems can all build their own MLIR layers. These layers often look different
from upstream dialects because they are tied to a specific product, runtime,
frontend, or hardware family.

## How To Read These Chapters

Read third-party dialect chapters as examples of MLIR specialization.

An upstream dialect such as `arith`, `tensor`, `memref`, or `scf` is generally
available to MLIR users and appears in many pipelines. A third-party dialect
may be just as important, but usually inside its own project. For example, an
IREE dialect matters inside the IREE compiler. A CIRCT dialect matters inside a
hardware design flow. A torch-mlir dialect matters while lowering PyTorch
programs. StableHLO matters as a portable operation set for ML framework and
accelerator workflows.

The beginner question is not "is this dialect universal?" The better question
is:

- what domain does this project care about?
- what information does this dialect preserve?
- what does it lower from?
- what does it lower to?
- what runtime, backend, or tool expects it?

## Common Ecosystem Patterns

Many ecosystem dialects follow one of a few patterns.

Frontend dialects preserve source-framework meaning. The `torch` dialect in
torch-mlir and the `onnx` dialect in ONNX-MLIR keep framework concepts visible
before they are lowered into more general tensor, linear algebra, or backend
IR.

Runtime compiler dialects describe execution systems. IREE dialects such as
`flow`, `stream`, `hal`, and `vm` model dispatch formation, asynchronous
execution, hardware abstraction, and virtual-machine-level runtime behavior.

Hardware dialects model circuits, signals, modules, protocols, and generated
hardware artifacts. CIRCT dialects are the main example in this book.

Compatibility and portability dialects stabilize boundaries between tools.
StableHLO and VHLO are examples: they help machine-learning systems exchange
programs without depending on one frontend or backend implementation.

Domain-specific extension dialects represent concepts that upstream MLIR does
not directly own. IMEX and DaCe show this pattern for array programming,
distributed data, GPU runtime extensions, and data-centric graph execution.

## How These Dialects Connect To Upstream MLIR

Third-party dialects still use the same MLIR building blocks: operations,
values, attributes, types, regions, blocks, verification, passes, conversion
targets, pattern rewrites, and pass pipelines. That shared substrate is what
lets project-specific IR coexist with upstream dialects.

A typical ecosystem pipeline may start in a custom frontend dialect, lower
through upstream tensor or structured-computation dialects, use project-specific
runtime dialects in the middle, and eventually lower toward target dialects or
a custom runtime format.

For example:

```text
PyTorch or ONNX frontend dialect
  -> tensor, linalg, StableHLO, or project-specific tensor IR
  -> runtime/compiler dialects such as IREE flow, stream, HAL, or VM
  -> target-specific code generation and runtime artifacts
```

For hardware:

```text
hardware-oriented source dialect
  -> CIRCT structural and behavioral dialects
  -> SystemVerilog, simulation, verification, or hardware-specific output
```

For data-centric execution:

```text
array or graph-oriented dialect
  -> structured computation, memory, GPU, or distributed runtime dialects
  -> target-specific execution code
```

The exact path changes by project. The recurring idea is the same: custom
dialects preserve the information that a project needs, then passes and
conversions gradually translate that information into a form another layer can
consume.

## Reading Order

If you are new to MLIR, read the upstream chapters first. They teach the common
IR mechanics that every ecosystem project reuses. Then use these ecosystem
sections based on your goal:

- read frontend and model dialects if you care about ML framework import,
  portability, and model-level semantics;
- read IREE if you care about runtime-oriented ML compilation;
- read CIRCT if you care about hardware, circuits, simulation, or verification;
- read IMEX and DaCe if you care about array programming, distributed data, or
  data-centric compilation.

The purpose of this part of the book is not to make every reader an expert in
every external project. It is to show how MLIR scales beyond upstream LLVM and
how real projects use dialects to organize their compiler stacks.

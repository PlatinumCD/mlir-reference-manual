# `omp` Dialect

## Beginner Summary

The `omp` dialect represents OpenMP directives in MLIR.

It models:

- Parallel regions.
- Teams and target offload.
- Worksharing loops.
- SIMD and loop wrappers.
- Sections and single/master/masked regions.
- Tasks and task loops.
- Reductions and private variables.
- Atomics and synchronization.
- Data mapping for target regions.
- OpenMP loop transformation handles.

For beginners, `omp` is the dialect that keeps OpenMP semantics visible inside
MLIR before lowering to LLVM-compatible IR and, eventually, OpenMP runtime calls
or LLVM IR constructs.

## Why This Dialect Exists

OpenMP is not just "spawn some threads." It has a large directive language:

```text
parallel
for / do
simd
teams
target
map
private / firstprivate
reduction
task
atomic
barrier
critical
ordered
```

Those directives carry semantic information that should not be lost too early.
The `omp` dialect preserves that information in MLIR form.

This gives MLIR a place to:

- Import OpenMP from Clang or Flang-style frontends.
- Convert `scf.parallel` into OpenMP worksharing constructs.
- Analyze and transform OpenMP target offload regions.
- Represent privatization and reductions explicitly.
- Keep OpenMP directives separate from the eventual LLVM IR lowering.

## When It Matters

The `omp` dialect matters in compiler pipelines that target OpenMP execution.

You see it when:

- A frontend lowers OpenMP source constructs into MLIR.
- `scf.parallel` is converted to OpenMP.
- A CPU pipeline wants OpenMP parallel worksharing.
- A GPU or accelerator offload pipeline uses OpenMP `target` regions.
- Privatization, mapping, reduction, or task semantics must remain visible.
- OpenMP ops are being converted to LLVM-compatible operand and region types.

The dialect usually remains present until late lowering. `convert-openmp-to-llvm`
does not erase OpenMP semantics; it converts operand types and nested IR so the
OpenMP dialect can later be translated to LLVM IR using OpenMP lowering support.

## When To Use It

Use `omp` when the IR should preserve OpenMP directive semantics.

Good uses include:

- Representing OpenMP source directives in MLIR.
- Lowering `scf.parallel` to OpenMP.
- Modeling OpenMP target offload.
- Expressing OpenMP data mapping with `omp.map.info` and `omp.map.bounds`.
- Representing reductions with `omp.declare_reduction`.
- Representing private and firstprivate behavior with `omp.private`.
- Keeping tasking and synchronization semantics explicit.

Avoid using `omp` as a generic threading dialect. If the program only needs
structured loops, use `scf`. If it needs GPU execution without OpenMP source
semantics, use `gpu` or target-specific dialects. Use `omp` when OpenMP itself
is the semantic contract.

## Core Concepts

### OpenMP Directives As Operations

Most OpenMP directives become operations with regions and clauses.

For example:

```mlir
omp.parallel {
  omp.barrier
  omp.terminator
}
```

The operation is not just a generic region. It means OpenMP parallel execution,
with OpenMP rules for teams of threads, data sharing, reductions, and barriers.

### Clauses Become Operands And Attributes

OpenMP clauses such as `if`, `num_threads`, `private`, `reduction`, `schedule`,
`map`, and `nowait` are represented through operation operands and attributes.

This makes clauses visible to MLIR verification and transformation passes.

### Loop Wrappers And Loop Nests

The dialect separates wrapper directives from the rectangular loop nest:

- `omp.wsloop` models a worksharing loop directive.
- `omp.simd` models SIMD execution.
- `omp.distribute` models distribution across teams.
- `omp.loop_nest` holds the rectangular loop bounds, steps, and induction
  variables.

This separation lets one loop nest be wrapped by several OpenMP semantics.

### OpenMP Target Offload

OpenMP target offload uses operations such as:

- `omp.target`
- `omp.target_data`
- `omp.target_enter_data`
- `omp.target_exit_data`
- `omp.target_update`
- `omp.map.info`
- `omp.map.bounds`

These describe how host values are mapped to device execution.

### OpenMP Conversion To LLVM Is Not Erasure

`convert-openmp-to-llvm` converts the IR around OpenMP to LLVM-compatible types
and operations, but OpenMP operations can remain as OpenMP operations. This is
different from a dialect that lowers completely to `llvm.*` ops in one step.

## Operations

The current OpenMP dialect in this LLVM checkout defines 65 generated
operations.

### Parallel, Teams, And Region Structure

Core region operations:

- `omp.parallel`
- `omp.teams`
- `omp.scope`
- `omp.terminator`
- `omp.yield`

Single-thread or selected-thread regions:

- `omp.single`
- `omp.master`
- `omp.masked`

Sections:

- `omp.sections`
- `omp.section`

Workshare wrappers:

- `omp.workshare`
- `omp.workshare.loop_wrapper`
- `omp.workdistribute`

### Loops, Worksharing, SIMD, And Loop Transform Handles

Loop representation and wrappers:

- `omp.loop_nest`
- `omp.loop`
- `omp.wsloop`
- `omp.simd`
- `omp.distribute`

Canonical-loop and transformation support:

- `omp.new_cli`
- `omp.canonical_loop`
- `omp.tile`
- `omp.fuse`
- `omp.unroll_heuristic`

These model OpenMP-compatible canonical loop information and transformations.

### Tasks

Tasking operations include:

- `omp.task`
- `omp.taskgroup`
- `omp.taskwait`
- `omp.taskyield`
- `omp.taskloop.context`
- `omp.taskloop.wrapper`

Use these for OpenMP task semantics, including task loop lowering support.

### Synchronization And Atomics

Synchronization operations:

- `omp.barrier`
- `omp.flush`
- `omp.critical`
- `omp.critical.declare`
- `omp.ordered`
- `omp.ordered.region`
- `omp.cancel`
- `omp.cancellation_point`
- `omp.scan`

Atomic operations:

- `omp.atomic.read`
- `omp.atomic.write`
- `omp.atomic.update`
- `omp.atomic.capture`
- `omp.atomic.compare`

These carry OpenMP memory and synchronization semantics.

### Data Sharing, Reductions, And Declarations

Data sharing and reductions:

- `omp.private`
- `omp.declare_reduction`
- `omp.declare_simd`

Mapping and mapper declarations:

- `omp.declare_mapper`
- `omp.declare_mapper.info`

Allocation directives:

- `omp.allocate_dir`
- `omp.allocate_free`

Iterator and affinity helper operations:

- `omp.iterator`
- `omp.affinity_entry`

### Target Offload And Memory

Target and data movement operations:

- `omp.target`
- `omp.target_data`
- `omp.target_enter_data`
- `omp.target_exit_data`
- `omp.target_update`
- `omp.map.info`
- `omp.map.bounds`

Target/shared allocation operations:

- `omp.target_allocmem`
- `omp.target_freemem`
- `omp.alloc_shared_mem`
- `omp.free_shared_mem`

Thread and group storage:

- `omp.threadprivate`
- `omp.groupprivate`

## Transformations

The OpenMP dialect has a small set of native transformation passes focused on
offload metadata and device memory preparation.

### Native OpenMP Passes

`omp-mark-declare-target`
: Marks functions called by an OpenMP declare-target function or `omp.target`
  region as declare target.

`omp-offload-privatization-prepare`
: Prepares OpenMP maps for privatized variables used by deferred target tasks,
  especially `nowait` target regions where stack lifetime is not enough.

`omp-stack-to-shared`
: Replaces selected `llvm.alloca` operations on target devices with
  `omp.alloc_shared_mem` and `omp.free_shared_mem` so values intended to be
  shared across threads use target shared memory.

### Built-In Rewrite And Verification Surface

OpenMP operations have extensive verifier logic because clause combinations,
region bodies, wrapper nesting, reductions, and mapping operands must satisfy
OpenMP rules.

Important transformation-related concepts include:

- Privatizer recipes through `omp.private`.
- Reduction recipes through `omp.declare_reduction`.
- Mapper recipes through `omp.declare_mapper`.
- Loop wrapper composition around `omp.loop_nest`.
- Canonical loop information through `omp.new_cli` and `omp.canonical_loop`.
- Target mapping through `omp.map.info` and `omp.map.bounds`.

## Conversions And Lowering Paths

### From SCF

`convert-scf-to-openmp` converts `scf.parallel` to OpenMP parallel and
worksharing constructs.

The conversion builds:

- `omp.parallel`
- `omp.wsloop`
- `omp.loop_nest`
- `omp.yield`
- `omp.terminator`
- reduction declarations when needed

The pass has a `num-threads` option.

Use this when a structured parallel loop should become an OpenMP worksharing
loop.

### To LLVM-Compatible OpenMP

`convert-openmp-to-llvm` converts OpenMP operations so their operands, nested
operations, and related memory/function/control-flow pieces are compatible with
the LLVM dialect.

It populates conversion patterns for:

- OpenMP operations.
- `arith` operations used inside OpenMP regions.
- `cf` operations and assertions.
- MemRef operations.
- Func operations.

Some OpenMP operations such as `omp.barrier`, `omp.flush`, `omp.taskwait`,
`omp.taskyield`, and `omp.terminator` remain legal OpenMP operations after this
conversion. The generic `convert-to-llvm` driver can also use the OpenMP
conversion interface.

The final translation to LLVM IR is where OpenMP runtime and LLVM OpenMP
lowering support become relevant.

## Example IR

### Parallel Region

```mlir
func.func @parallel_region() {
  omp.parallel {
    omp.barrier
    omp.terminator
  }
  func.return
}
```

This creates an OpenMP parallel region with an explicit barrier inside it.

### Worksharing Loop

```mlir
func.func @workshare_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : index = (%c0) to (%c10) step (%c1) {
        omp.yield
      }
    }
    omp.terminator
  }
  func.return
}
```

`omp.wsloop` gives the loop OpenMP worksharing semantics. `omp.loop_nest`
contains the rectangular loop bounds and induction variable.

### SCF To OpenMP Input

```mlir
func.func @scf_parallel() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.parallel (%i) = (%c0) to (%c10) step (%c1) {
    scf.reduce
  }
  func.return
}
```

Running `convert-scf-to-openmp` turns this into OpenMP parallel and worksharing
loop constructs.

## Mental Model

Think of the `omp` dialect as "OpenMP directives in MLIR form."

It is higher level than LLVM OpenMP runtime calls. It is also more specific than
generic parallel loops because it preserves the OpenMP standard's semantics.

The main question is:

```text
Is the compiler still reasoning about OpenMP directives,
or is it ready to lower to LLVM-compatible runtime/codegen form?
```

If the answer is OpenMP directives, keep `omp`. If the answer is target codegen,
run the OpenMP-to-LLVM conversion path and later LLVM IR translation.

## Gotchas

OpenMP regions have OpenMP-specific terminators.

Many OpenMP regions end with `omp.terminator` or `omp.yield`, not `func.return`
or `scf.yield`.

Loop wrappers are not the loop body itself.

`omp.wsloop`, `omp.simd`, and `omp.distribute` wrap loop semantics. The actual
rectangular loop information is commonly in `omp.loop_nest`.

OpenMP-to-LLVM does not mean every `omp` op disappears.

The conversion prepares OpenMP operations for LLVM-compatible lowering. Some
OpenMP operations remain as OpenMP dialect operations until final translation.

Clauses are semantic.

Changing or dropping `map`, `private`, `reduction`, `nowait`, or `schedule`
information changes program meaning.

Target offload needs lifetime care.

The `omp-offload-privatization-prepare` pass exists because deferred target
tasks can outlive stack allocations from the generating task.

## Source Map

Primary source files:

- `mlir/include/mlir/Dialect/OpenMP/OpenMPDialect.td`
- `mlir/include/mlir/Dialect/OpenMP/OpenMPOpBase.td`
- `mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td`
- `mlir/include/mlir/Dialect/OpenMP/OpenMPClauses.td`
- `mlir/include/mlir/Dialect/OpenMP/OpenMPEnums.td`
- `mlir/include/mlir/Dialect/OpenMP/OpenMPOpsInterfaces.td`
- `mlir/include/mlir/Dialect/OpenMP/Transforms/Passes.td`
- `mlir/lib/Dialect/OpenMP/IR/OpenMPDialect.cpp`
- `mlir/lib/Dialect/OpenMP/Transforms/`
- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/OpenMPToLLVM/OpenMPToLLVM.cpp`
- `mlir/lib/Conversion/SCFToOpenMP/SCFToOpenMP.cpp`

Generated op documentation source:

```text
mlir-tblgen --gen-op-doc -dialect=omp \
  -I llvm-project-build/tools/mlir/include \
  -I llvm-project/mlir/include \
  llvm-project/mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td
```

# Intel IMEX `region` Dialect

## Beginner Summary

The Intel IMEX `region` dialect is a small annotation dialect. It wraps a group
of operations and attaches an environment to that group. The most important
environment in IMEX is a GPU environment, written as `#region.gpu_env<...>`.

For a beginner, the useful mental model is:

```text
ordinary MLIR operations
  inside region.env_region
  with an environment attribute
  later passes interpret that environment
```

The dialect does not define arithmetic, tensor operations, memory operations, or
device kernels. Instead, it preserves meaning across lowering. For example, an
NDArray value may say that its data lives on a GPU. As the program lowers into
Linalg, MemRef, SCF, and GPU dialects, the original high-level NDArray operation
may disappear. A `region.env_region` can keep the placement fact visible long
enough for later passes to allocate memory differently, convert parallel loops
to GPU launches, insert GPU copies, or erase the wrapper once it has served its
purpose.

## Why This Dialect Exists

MLIR already has attributes and SSA values, but neither is always convenient for
marking a whole group of operations with information that must survive unrelated
passes. Plain attributes can be dropped when operations are rewritten. Encoding
annotations only through use-def chains can be persistent, but it can also make
IR harder to build and harder to read.

The `region` dialect gives IMEX a simple middle ground: wrap the operations in a
single-block region and attach an environment attribute plus optional SSA
arguments. The operations inside the region execute exactly once. They are not
isolated from above, so they can use values that dominate the region operation.
Values needed after the region are returned with `region.env_region_yield`.

This is especially useful for the compute-follows-data model. If a tensor is
known to live on a GPU, computations over that tensor should usually happen in
the same device environment. The `region` dialect lets IMEX carry that
placement intent through intermediate rewrites until device-specific lowering
can act on it.

## When It Matters

You will see `region` when IMEX needs to preserve execution-environment
information while other dialects are being lowered. It matters when:

- NDArray operations over GPU-resident arrays are being wrapped by
  `add-gpu-regions`;
- Linalg, tensor, or memref operations still need to remember that they belong
  to a GPU environment;
- `scf.parallel` operations should become `gpu.launch` only inside GPU-marked
  regions;
- memref allocations or copies should be replaced by GPU-aware operations only
  for work inside a GPU environment;
- the placement annotation is no longer needed and the region can be erased.

Use the dialect when a group of operations needs a durable environment marker.
Do not use it as a general structured-control-flow dialect. For loops, branches,
and execution regions without environment meaning, use the standard MLIR
dialects such as SCF, CF, Func, or GPU directly.

## Core Model

`region.env_region` is the wrapper operation. It takes one required
`environment` attribute and optional SSA arguments. It may produce any number of
results. Its body is a single region, and that region is terminated by
`region.env_region_yield`.

The operation is intentionally not isolated from above. Code inside the body can
use values from the surrounding lexical scope. The body has its own lexical
scope, so values defined inside the region are not visible after the region
unless they are yielded.

The environment attribute is deliberately general. IMEX defines
`#region.gpu_env<device = "...">`, but `region.env_region` can also carry other
attributes, including string attributes used in tests. The dialect itself does
not decide what an environment means. Later passes decide whether to interpret,
preserve, or ignore a particular environment.

Nested regions are allowed. A generic region may contain a GPU region, or GPU
regions may appear next to each other. Interpretation of nesting is left to the
passes that consume the regions.

## Attribute Inventory

The dialect defines one concrete attribute:

```text
#region.gpu_env<device = "...">
```

`#region.gpu_env` is represented by `GPUEnvAttr`. Its current parameter is a
string attribute named `device`. The string identifies the target device or
runtime filter. Examples in the IMEX sources use values such as `"XeGPU"`,
`"gpu"`, `"test"`, and `"opencl:gpu:0"`.

The attribute is important because helper predicates such as `isGpuRegion`
recognize GPU regions by checking whether the `environment` attribute is a
`GPUEnvAttr`. That means a region with a plain string environment may still be a
valid `region.env_region`, but GPU-specific passes will not treat it as a GPU
placement marker unless the environment is a `#region.gpu_env`.

## Operation Inventory

The current `region` dialect defines two operations:

```text
region.env_region
region.env_region_yield
```

`region.env_region` groups operations under an environment. It has:

- an `environment` attribute;
- optional SSA operands called `args`;
- optional variadic results;
- one single-block body region;
- an implicit `region.env_region_yield` terminator.

Its assembly shape is:

```mlir
%result = region.env_region #region.gpu_env<device = "XeGPU">
    %arg0 : index -> tensor<16xf32> {
  ...
  region.env_region_yield %value : tensor<16xf32>
}
```

`region.env_region_yield` terminates the body of `region.env_region`. Its
operands become the parent region operation's results. If the parent has no
results, the yield may have no operands and the custom printer can omit it in
simple cases.

## Canonicalization

`region.env_region` has canonicalization patterns. These run through MLIR's
normal `canonicalize` infrastructure.

The first pattern merges adjacent `region.env_region` operations when they have
the same environment and the same operands. This turns several consecutive
regions with identical placement into one larger region.

The second pattern cleans up yield values. It removes unused yielded results and
deduplicates repeated yielded values. This matters because regions are often
introduced mechanically by passes. Without cleanup, they can accumulate extra
results that make the IR harder to read and harder for later rewrites to
process.

## Bufferization Behavior

IMEX registers external bufferization interfaces for both region operations.
This lets the region wrapper survive tensor-to-buffer lowering.

For `region.env_region`, tensor operands are converted to buffers before being
passed into the new region operation. Result types are inferred by converting
the operands of the terminator to buffers. The operation is recreated with
bufferized operands and result types, then the original region body is moved
into the new operation.

For `region.env_region_yield`, yielded tensor values are converted to buffers.
Yield operands are treated as equivalent to the corresponding parent operation
results and must bufferize in place. In beginner terms, bufferization keeps the
environment wrapper but changes the values crossing its boundary from tensors
to memrefs when needed.

## Transformations And Conversions

The main passes directly related to this dialect are:

```text
drop-regions
convert-region-parallel-loops-to-gpu
add-gpu-regions
insert-gpu-allocs
insert-gpu-copy
convert-ndarray-to-linalg
```

`drop-regions` removes `region.env_region` operations. It inlines the body into
the parent block, erases the `region.env_region_yield`, and replaces the region
operation's results with the yielded values. Use this when later lowering no
longer needs the environment annotation.

`convert-region-parallel-loops-to-gpu` applies MLIR's SCF parallel-loop-to-GPU
conversion only inside GPU regions. It walks the IR, collects
`region.env_region` operations whose environment is `#region.gpu_env`, and runs
the conversion on those collected regions. This means a mapped `scf.parallel`
outside a GPU region can remain an `scf.parallel`, while the same shape inside a
GPU environment can become `gpu.launch`.

`add-gpu-regions` is an NDArray transformation, but it is one of the most
important producers of `region` IR. When an NDArray operation works on types
with a `GPUEnvAttr`, the pass creates a `region.env_region` with that GPU
environment, moves the operation inside, and yields the result.

`insert-gpu-allocs` and `insert-gpu-copy` are general IMEX transformations that
use region placement. `insert-gpu-allocs` can restrict GPU allocation rewriting
to allocations inside GPU regions through its `in-regions` option.
`insert-gpu-copy` converts `memref.copy` to `gpu.memcpy` when the copy appears
inside an environment region.

`convert-ndarray-to-linalg` depends on the Region dialect because NDArray
lowering may need to preserve environment placement while replacing high-level
array operations with Linalg, Tensor, MemRef, and bufferization constructs.

## What The Dialect Implies

Seeing `region.env_region` means the compiler is carrying information that is
not represented by the inner operations alone. The inner operations may look
ordinary, but the enclosing environment changes how later passes should treat
them.

Seeing `#region.gpu_env` specifically implies device placement. It does not, by
itself, allocate GPU memory or launch a GPU kernel. It says that the enclosed
work belongs to a GPU environment, so later passes are allowed to make
GPU-specific decisions.

A region also implies a lifetime boundary for local SSA values. Values produced
inside the body do not escape unless they are yielded. This gives the wrapper a
clear interface even though it can still read values from the surrounding scope.

## How To Use It In Practice

When reading IMEX IR, first ask what the environment attribute is. A
`#region.gpu_env` region is a GPU placement marker. A string or other attribute
may simply be a generic annotation or a temporary protection marker used by a
specific pass.

Then inspect the yielded values. Results of the `region.env_region` are exactly
the operands of `region.env_region_yield`. If a value from inside the region is
used afterward, it must appear in the yield.

For lowering questions, find the next consumer. If the goal is GPU launch
creation, look for `convert-region-parallel-loops-to-gpu`. If the goal is GPU
memory behavior, look for `insert-gpu-allocs` or `insert-gpu-copy`. If the
environment marker has already done its job, look for `drop-regions`.

## Common Mistakes

Do not assume that every `region.env_region` is a GPU region. GPU-specific logic
checks for `#region.gpu_env`; a plain string environment is not enough.

Do not assume the region launches work by itself. It only marks and scopes work.
The launch conversion happens later and only for supported patterns such as
mapped `scf.parallel` inside a GPU region.

Do not forget that the region is single-block and terminator-based. Values that
must escape need to be yielded, and yielded operands must match the parent
operation's result types.

## Source Pointers

The core definitions are in
`include/imex/Dialect/Region/IR/RegionOps.td` and
`lib/Dialect/Region/IR/RegionOps.cpp`. Region helper predicates are in
`include/imex/Dialect/Region/RegionUtils.h`. Bufferization support is in
`lib/Dialect/Region/Transforms/BufferizableOpInterfaceImpl.cpp`. The main
conversion implementations are in `lib/Conversion/DropRegions` and
`lib/Conversion/RegionParallelLoopToGpu`.

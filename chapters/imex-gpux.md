# Intel IMEX gpux Dialect

The Intel IMEX `gpux` dialect extends MLIR's upstream `gpu` dialect with explicit device, context, and stream handles. It exists because the upstream `gpu` dialect can describe GPU allocation, copies, memset, waits, and kernel launches, but it does not expose a first-class stream argument on those operations. GPUX makes the stream visible in IR.

For a beginner, the important idea is that GPUX is not a full replacement for the GPU dialect. It is a narrow extension for runtime control. A normal `gpu.launch_func` says "launch this kernel." A `gpux.launch_func` says "launch this kernel on this particular stream." The same pattern applies to `alloc`, `dealloc`, `memcpy`, `memset`, and `wait`: the operation remains GPU-like, but the stream is explicit.

This is important for runtimes such as Level Zero and SYCL, where queues, streams, devices, and contexts are real runtime objects. If a compiler pipeline needs to control or preserve those runtime objects, hiding the stream until late LLVM lowering is too late. GPUX introduces a middle-level IR stage where stream choice is still visible and optimizable.

## Operation Inventory

The `gpux` dialect defines these operations:

```text
gpux.create_device
gpux.create_context
gpux.create_stream
gpux.destroy_device
gpux.destroy_context
gpux.destroy_stream
gpux.launch_func
gpux.alloc
gpux.dealloc
gpux.wait
gpux.memcpy
gpux.memset
```

It also defines these runtime-handle types:

```text
!gpux.OpaqueType
!gpux.DeviceType
!gpux.ContextType
!gpux.StreamType
```

During lowering to LLVM, these GPUX handle types are converted to opaque LLVM pointers.

## Device, Context, And Stream Ops

`gpux.create_device` creates a GPU device handle from an input device value and returns `!gpux.DeviceType`. The RFC describes this as creating a SYCL or Level Zero device handle.

`gpux.create_context` creates a context from a `!gpux.DeviceType` and returns `!gpux.ContextType`. Contexts are the runtime scope in which resources and commands are associated with a device.

`gpux.create_stream` creates a stream or queue. It takes optional device and context operands and returns `!gpux.StreamType`. In the conversion implementation, if a function does not already contain a `gpux.create_stream`, the `convert-gpu-to-gpux` pass inserts one at the start of the single-block function and inserts `gpux.destroy_stream` before the terminator.

`gpux.destroy_device`, `gpux.destroy_context`, and `gpux.destroy_stream` release the corresponding runtime handles. In the current GPU-to-GPUX conversion, only stream creation and destruction are automatically inserted. Device and context operations are part of the dialect surface but are not the main path used by the conversion patterns in this checkout.

## Stream-Aware GPU Ops

`gpux.alloc` is the GPUX version of `gpu.alloc`. It allocates a memref on a GPU stream. It keeps upstream GPU allocation concepts such as async dependencies, dynamic sizes, symbol operands, and `hostShared`, but adds the `!gpux.StreamType` operand. Its result is the allocated memref, with an optional async token.

`gpux.dealloc` is the stream-aware version of `gpu.dealloc`. It takes async dependencies, a stream, and the memref to free. The stream says where the deallocation is queued.

`gpux.launch_func` is the stream-aware version of `gpu.launch_func`. It has async dependencies, a stream, a kernel symbol reference, grid sizes, block sizes, optional dynamic shared memory size, and kernel operands. The op exposes helpers to retrieve the kernel module name and kernel function name. Its result is an optional async token.

`gpux.wait` waits for operations associated with a stream. It extends `gpu.wait` by adding the stream operand.

`gpux.memcpy` copies between two memrefs on a stream. It keeps the memory effects of reading the source and writing the destination. `gpux.memset` writes a scalar value into a destination memref on a stream.

These six stream-aware operations are the core reason GPUX exists. They preserve the upstream GPU programming model while making stream scheduling explicit.

## When To Use It

Use `gpux` after a program has already been represented with MLIR GPU operations and before lowering to runtime calls. It is useful when a backend or integration layer needs explicit control over which stream or queue receives allocation, copy, memset, wait, and launch work.

You should inspect GPUX IR when debugging IMEX GPU lowering, SYCL or Level Zero runtime integration, stream creation, or runtime call generation. It is also useful when checking whether a pass inserted only one stream per function or accidentally created multiple independent streams.

Do not use `gpux` for GPU kernel bodies. Kernel bodies still use normal MLIR dialects inside `gpu.module` and `gpu.func`, such as `arith`, `memref`, `scf`, vector dialects, SPIR-V-oriented dialects, or target-specific dialects. GPUX describes host-side runtime orchestration around those kernels.

## Conversion Into GPUX

The pass `convert-gpu-to-gpux` converts selected upstream GPU operations into GPUX operations. It is implemented as greedy rewrite patterns, not a full dialect conversion target.

The helper `getGpuStream` looks for an existing `gpux.create_stream` in the single block of the surrounding `func.func`. If it finds one, all converted operations use that stream. If it does not find one, it inserts `gpux.create_stream` at the start of the block and inserts `gpux.destroy_stream` before the terminator.

The pass rewrites:

```text
gpu.alloc       -> gpux.alloc
gpu.dealloc     -> gpux.dealloc
gpu.launch_func -> gpux.launch_func
gpu.wait        -> gpux.wait
gpu.memcpy      -> gpux.memcpy
gpu.memset      -> gpux.memset
```

For `gpu.launch_func`, the converter looks up the referenced `gpu.module` and `gpu.func`, preserves grid and block sizes, preserves dynamic shared memory, passes through kernel operands, and adds the stream operand. Async tokens and async dependencies are preserved where present.

This conversion implies that all converted host-side GPU activity in the function is queued on one visible stream unless the IR already introduced a stream.

## Conversion Out Of GPUX

The pass `convert-gpux-to-llvm` lowers GPUX operations to LLVM dialect plus GPU runtime wrapper calls. It also calls MLIR's existing `populateGpuToLLVMConversionPatterns`, then adds GPUX-specific type conversions and patterns through `populateGpuxToLLVMPatternsAndLegality`.

The runtime wrapper calls used by the lowering include:

```text
gpuCreateStream
gpuStreamDestroy
gpuMemAlloc
gpuMemFree
gpuMemCopy
gpuModuleLoad
gpuModuleUnload
gpuKernelGet
gpuLaunchKernel
gpuWait
```

The type converter maps `!gpux.OpaqueType`, `!gpux.StreamType`, `!gpux.DeviceType`, and `!gpux.ContextType` to LLVM pointer type.

`gpux.create_stream` lowers to `gpuCreateStream`. In the current implementation, the conversion passes null device and context pointers for the workflow where no explicit device and context are provided.

`gpux.destroy_stream` lowers to `gpuStreamDestroy`.

`gpux.alloc` lowers to `gpuMemAlloc`, computes the memref byte size, uses a fixed 64-byte alignment, passes the `hostShared` flag, and constructs an LLVM memref descriptor with the allocated pointer, aligned pointer, offset, sizes, and strides.

`gpux.dealloc` extracts the allocated pointer from the converted memref descriptor and lowers to `gpuMemFree`.

`gpux.memcpy` computes the total byte size from the source memref shape and element size, computes source and destination element pointers from their memref descriptors, and lowers to `gpuMemCopy`.

`gpux.launch_func` lowers to a sequence of runtime calls. It extracts the serialized GPU binary from an attribute on the `gpu.module`, materializes it as an LLVM global string, calls `gpuModuleLoad`, creates a global C string for the kernel name, calls `gpuKernelGet`, builds the kernel parameter array and event dependency array on the stack, and calls `gpuLaunchKernel`. If the launch is not async, it emits `gpuWait` on the stream and erases the launch. If the launch is async, it returns the event as the async token.

The conversion also removes `gpu.module` operations after their binary contents have been consumed.

## Coverage Notes

The declared GPUX dialect surface is larger than the lowering patterns visible in this checkout. The LLVM lowering explicitly adds patterns for stream create and destroy, allocation, deallocation, memcpy, launch, and GPU module removal. It uses `gpuWait` internally for non-async launches. I do not see standalone lowering patterns here for `gpux.wait`, `gpux.memset`, `gpux.create_device`, `gpux.create_context`, `gpux.destroy_device`, or `gpux.destroy_context`. Since the conversion target marks the GPUX dialect illegal, those operations would need patterns or be absent for a successful full pipeline.

That detail matters for readers: a dialect can declare operations that are planned or useful at the IR level, while a particular lowering path may only support the subset used by current pipelines.

## How To Read gpux IR

Start by finding `gpux.create_stream`. Then trace the stream value through `gpux.alloc`, `gpux.memcpy`, `gpux.memset`, `gpux.launch_func`, `gpux.wait`, and `gpux.dealloc`. That tells you the host-side ordering and queue association for GPU work.

Next, inspect async tokens. GPUX keeps the GPU async interface, so dependencies can still express ordering separate from plain operation order. The stream operand and async dependencies answer different questions: the stream tells where work is queued; async dependencies tell what work must complete first.

Finally, look at the lowering target. If the next stage is `convert-gpux-to-llvm`, the GPUX operations are about to become calls to the runtime wrapper API. At that point, stream visibility is no longer just an MLIR modeling convenience; it becomes an actual pointer passed to allocation, copy, launch, wait, and stream destruction calls.

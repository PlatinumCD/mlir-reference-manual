# IREE hal Dialect

The IREE `hal` dialect models the Hardware Abstraction Layer used by IREE to talk to devices, queues, buffers, command buffers, executable binaries, synchronization objects, and runtime configuration. The dialect description calls it a Vulkan-like model with the graphics parts removed. That is a good beginner mental model: HAL is the compiler IR for "how this program will use a device" rather than "what tensor algebra does this model compute."

`hal` is important after IREE has moved beyond pure tensor computation. Earlier dialects describe programs, tensor operations, stream scheduling, and executable code. HAL connects those compiler-level decisions to runtime resources: which device is selected, which allocator produces a buffer, how dispatch commands are recorded, what executable variant is loaded, and which fence signals completion.

You normally do not hand-write large HAL programs. You inspect HAL when debugging IREE's runtime boundary, device selection, buffer allocation, dispatch launch metadata, executable translation, or synchronization. It is also one of the best examples of an MLIR dialect that models a real runtime API while still keeping enough structure for compiler passes.

## Operation Inventory

The core `hal` dialect defines these operations:

```text
hal.allocator.allocate
hal.allocator.import
hal.allocator.resolve_memory_properties
hal.allocator.select
hal.buffer.allocation.discard
hal.buffer.allocation.is_terminal
hal.buffer.allocation.preserve
hal.buffer.assert
hal.buffer.length
hal.buffer.load
hal.buffer.store
hal.buffer.subspan
hal.buffer_usage
hal.buffer_view.assert
hal.buffer_view.buffer
hal.buffer_view.create
hal.buffer_view.dim
hal.buffer_view.element_type
hal.buffer_view.encoding_type
hal.buffer_view.rank
hal.buffer_view.trace
hal.channel.create
hal.channel.rank_and_count
hal.channel.split
hal.command_buffer.begin_debug_group
hal.command_buffer.collective
hal.command_buffer.copy_buffer
hal.command_buffer.create
hal.command_buffer.device
hal.command_buffer.dispatch
hal.command_buffer.dispatch.indirect
hal.command_buffer.end_debug_group
hal.command_buffer.execution_barrier
hal.command_buffer.fill_buffer
hal.command_buffer.finalize
hal.command_buffer.update_buffer
hal.device.allocator
hal.device.memoize
hal.device.query
hal.device.queue.alloca
hal.device.queue.barrier
hal.device.queue.copy
hal.device.queue.dealloca
hal.device.queue.execute
hal.device.queue.execute.indirect
hal.device.queue.fill
hal.device.queue.flush
hal.device.queue.read
hal.device.queue.update
hal.device.queue.write
hal.device.resolve
hal.devices.count
hal.devices.get
hal.dispatch.extern
hal.element_type
hal.encoding_type
hal.ex.file.from_memory
hal.executable
hal.executable.binary
hal.executable.calculate_workgroups
hal.executable.condition
hal.executable.constant.block
hal.executable.constant.load
hal.executable.create
hal.executable.export
hal.executable.export.ordinal
hal.executable.lookup
hal.executable.lookup.function
hal.executable.source
hal.executable.source_end
hal.executable.variant
hal.executable.variant_end
hal.executable_end
hal.fence.await
hal.fence.create
hal.fence.fail
hal.fence.join
hal.fence.query
hal.fence.signal
hal.instrument.memory.load
hal.instrument.memory.store
hal.instrument.print
hal.instrument.value
hal.instrument.workgroup
hal.interface.binding.subspan
hal.interface.constant.load
hal.interface.workgroup.count
hal.interface.workgroup.id
hal.interface.workgroup.size
hal.memory_type
hal.return
hal.tensor.alias
hal.tensor.barrier
hal.tensor.export
hal.tensor.import
hal.tensor.transients
```

## Core Types And Attributes

The main HAL object types are `!hal.allocator`, `!hal.buffer`, `!hal.buffer_view`, `!hal.collective.channel`, `!hal.command_buffer`, `!hal.device`, `!hal.event`, `!hal.executable`, `!hal.fence`, and `!hal.file`. Each is a compiler-visible handle to a runtime concept.

`!hal.buffer` is raw device-visible memory. `!hal.buffer_view` wraps a buffer with shape, element type, and encoding metadata so values can cross ABI boundaries. `!hal.command_buffer` records commands for later queue submission. `!hal.fence` represents a set of semaphore timepoints. `!hal.executable` is a loaded device executable, and `!hal.device` is the logical device that owns queues and allocators.

The important attributes are mostly runtime enums and bitfields. Memory and access behavior uses `MemoryTypeBitfield`, `BufferUsageBitfield`, `MemoryAccessBitfield`, `AccessScopeBitfield`, and `MemoryModel`. Queue commands use flags such as `CommandBufferModeBitfield`, `CommandCategoryBitfield`, `ExecutionStageBitfield`, `ExecutionBarrierFlagBitfield`, `DispatchFlags`, `ExecuteFlagBitfield`, `CopyFlagBitfield`, `FillFlagBitfield`, `ReadFlagBitfield`, `WriteFlagBitfield`, `UpdateFlagBitfield`, and `WaitFlagBitfield`. Executables use `PipelineLayout`, `PipelineBinding`, `ExecutableTarget`, and executable object attributes. Device selection uses `DeviceAlias`, `DeviceTarget`, `DeviceSelect`, `DeviceAffinity`, `DevicePromise`, `DeviceOptimal`, `DeviceTopology`, and `DeviceLink`.

These attributes are not decoration. They are how HAL preserves the runtime contract in IR: where memory lives, how it can be used, what synchronization is required, which backend owns an executable variant, and which device topology the module expects.

## Tensor Bridge And Pseudo Ops

`hal.tensor.import`, `hal.tensor.export`, `hal.tensor.alias`, `hal.tensor.transients`, and `hal.tensor.barrier` are bridge operations between tensor-level IR and HAL buffer views. `hal.tensor.import` turns a HAL buffer view into a tensor value. `hal.tensor.export` packages a tensor into a buffer view. `hal.tensor.alias` hints that tensor storage should alias a given buffer view. `hal.tensor.transients` marks tensors that should use transient storage from a provided buffer. `hal.tensor.barrier` signals a fence when tensors are available.

These ops are important because IREE has to cross between value-like tensor semantics and explicit runtime resources. A tensor is a compiler value; a buffer view is a runtime handle with shape and type metadata.

`hal.dispatch.extern` represents a dispatch of workgroups across a 3D grid for external code. It is a pseudo-style boundary operation used when dispatch behavior is not represented as a normal HAL executable variant in the current IR.

## Allocators, Buffers, And Buffer Views

Allocator operations are the first runtime resource layer. `hal.allocator.select` chooses a device or queue for allocation from a set. `hal.allocator.allocate` creates an empty buffer. `hal.allocator.import` imports a host buffer when the allocator supports it. `hal.allocator.resolve_memory_properties` turns resource lifetime and affinity information into concrete memory type and buffer usage properties.

Buffer operations work on `!hal.buffer`. `hal.buffer.assert` checks compatibility. `hal.buffer.length` returns byte length. `hal.buffer.subspan` creates an aliasing slice. `hal.buffer.load` and `hal.buffer.store` read and write scalar elements. The allocation ownership operations, `hal.buffer.allocation.preserve`, `hal.buffer.allocation.discard`, and `hal.buffer.allocation.is_terminal`, help passes reason about who owns the underlying allocation and whether it can be safely moved or released.

`hal.buffer_usage` and `hal.memory_type` materialize integer runtime bitmasks from symbolic buffer usage and memory type attributes. These are small operations, but they are essential because runtime calls often require integer flags.

Buffer view operations work on shaped buffer references. `hal.buffer_view.create` builds a view from a buffer, shape, element type, and encoding. `hal.buffer_view.buffer` extracts the underlying buffer. `hal.buffer_view.rank`, `hal.buffer_view.dim`, `hal.buffer_view.element_type`, and `hal.buffer_view.encoding_type` query metadata. `hal.buffer_view.assert` validates a view, and `hal.buffer_view.trace` emits trace information about values.

`hal.element_type` and `hal.encoding_type` are helper ops that translate MLIR element or encoding information into the integer forms used by the runtime ABI.

## Channels And Collectives

HAL channels model collective communication groups. `hal.channel.create` creates a collective channel. `hal.channel.split` derives a sub-channel. `hal.channel.rank_and_count` returns the local rank and the participant count.

Collective work can be recorded into command buffers with `hal.command_buffer.collective`. The associated attributes describe the collective kind, reduction operation, element type, and communication metadata. This matters for distributed or multi-device execution where "run this dispatch" is not enough; participants must agree on group identity and communication semantics.

## Command Buffers And Queue Commands

Command buffer operations describe recorded device work. `hal.command_buffer.create` allocates a command buffer, `hal.command_buffer.finalize` ends recording, and `hal.command_buffer.device` queries its device. Debug grouping uses `hal.command_buffer.begin_debug_group` and `hal.command_buffer.end_debug_group`.

Transfer and synchronization commands include `hal.command_buffer.execution_barrier`, `hal.command_buffer.fill_buffer`, `hal.command_buffer.update_buffer`, and `hal.command_buffer.copy_buffer`. Dispatch commands include `hal.command_buffer.dispatch` and `hal.command_buffer.dispatch.indirect`. These operations record work into a command buffer; they do not necessarily execute it immediately.

Queue operations submit or perform work on a device queue. `hal.device.queue.alloca` and `hal.device.queue.dealloca` manage queue-ordered transient buffers. `hal.device.queue.fill`, `hal.device.queue.update`, `hal.device.queue.copy`, `hal.device.queue.read`, and `hal.device.queue.write` enqueue transfer work. `hal.device.queue.barrier` enqueues a queue barrier. `hal.device.queue.execute` submits command buffers, while `hal.device.queue.execute.indirect` submits indirectly specified command buffers. `hal.device.queue.flush` flushes locally pending queue submissions.

This distinction is central: command buffer ops describe recorded command streams; queue ops submit or enqueue work with wait and signal fences.

## Devices And Device Selection

`hal.devices.count` and `hal.devices.get` query available devices. `hal.device.resolve` resolves device handles from affinity. `hal.device.allocator` gets a device allocator. `hal.device.query` reads runtime configuration from a device, often with a default path when a query is not available.

`hal.device.memoize` wraps regions that memoize resources for a particular device and queue affinity. This is a compiler-visible way to say that resource creation should be cached instead of repeated every time control reaches the same code.

The device-target passes make this group work. `iree-hal-assign-target-devices`, `iree-hal-assign-legacy-target-devices`, `iree-hal-materialize-target-devices`, `iree-hal-resolve-device-promises`, `iree-hal-resolve-device-aliases`, and `iree-hal-verify-devices` turn target specifications and aliases into concrete device globals and verify that compiler target plugins can handle them.

## Executables

HAL executable operations describe device code and the path from source to runtime binary. `hal.executable` is a target-specific executable module container, terminated by `hal.executable_end`. `hal.executable.variant` is a backend-specific variant of that executable, terminated by `hal.executable.variant_end`. `hal.executable.source` and `hal.executable.source_end` preserve source contents. `hal.executable.export` declares an entry point. `hal.executable.condition` holds host code that decides whether an executable is enabled. `hal.executable.calculate_workgroups` computes dispatch workgroup counts.

Constants and binary forms use `hal.executable.constant.block`, `hal.executable.constant.load`, and `hal.executable.binary`. Runtime lookup and creation use `hal.executable.create`, `hal.executable.lookup`, `hal.executable.lookup.function`, and `hal.executable.export.ordinal`.

The executable pass sequence is large because this is where IREE target backends do real work. `iree-hal-materialize-interfaces` defines executable variants for stream executables. `iree-hal-configure-executables` and `iree-hal-configure-target-executable-variants` configure variants for target backends. `iree-hal-translate-all-executables` and `iree-hal-translate-target-executable-variants` translate variants. `iree-hal-link-all-executables` and `iree-hal-link-target-executables` combine executables. `iree-hal-resolve-export-ordinals` assigns numeric export ordinals. `iree-hal-serialize-all-executables` and `iree-hal-serialize-target-executables` turn variants into binary blobs. `iree-hal-prune-executables`, `iree-hal-strip-executable-contents`, `iree-hal-substitute-executables`, and `iree-hal-hoist-executable-objects` clean, replace, or reorganize executable contents.

For beginners, the implication is that `hal.executable` is not just a function. It is a compilation product with sources, variants, target metadata, entry points, workgroup count logic, constants, and serialized binary data.

## Interface And Instrumentation Ops

`hal.interface.binding.subspan` returns a subspan of interface binding data, usually a buffer slice passed to a dispatch. `hal.interface.constant.load` loads constants from the interface constant block. `hal.interface.workgroup.id`, `hal.interface.workgroup.count`, and `hal.interface.workgroup.size` query the current dispatch's workgroup coordinates, grid size, and workgroup size.

Instrumentation ops model debug and profiling events. `hal.instrument.workgroup` emits a dispatch workgroup event. `hal.instrument.print` emits formatted text. `hal.instrument.value` records a scalar. `hal.instrument.memory.load` and `hal.instrument.memory.store` record memory events. `iree-hal-materialize-dispatch-instrumentation` materializes related host and device resources.

## Fences And Synchronization

Fence operations model asynchronous completion. `hal.fence.create` creates an unsignaled fence. `hal.fence.join` creates a fence from timepoints. `hal.fence.query` checks state. `hal.fence.signal` signals success, `hal.fence.fail` signals failure, and `hal.fence.await` waits asynchronously with a timeout or wait flag behavior.

Fences connect queue submissions, tensor barriers, and host-visible completion. In a HAL program, correctness is not only "the computation is right"; it is also "the right work waits for the right earlier work and signals the right later work."

## Transform And Conversion Surface

The main conversion pass is `iree-hal-conversion`. Its implementation builds a `HALTypeConverter` and populates conversion patterns from `populateHALToHALPatterns`, `populateUtilToHALPatterns`, `populateStandardToHALPatterns`, and `populateStreamToHALPatterns`. Registered dialect conversion interfaces can also contribute tensor-to-buffer mappings.

After HAL exists, it is eventually lowered toward IREE's VM and runtime modules. The HAL-to-VM population function is `populateHALToVMPatterns`, which delegates to `populateHALAllocatorToVMPatterns`, `populateHALBufferToVMPatterns`, `populateHALBufferViewToVMPatterns`, `populateHALChannelToVMPatterns`, `populateHALCommandBufferToVMPatterns`, `populateHALDeviceToVMPatterns`, `populateHALDevicesToVMPatterns`, `populateHALExecutableToVMPatterns`, `populateHALExperimentalToVMPatterns`, and `populateHALFenceToVMPatterns`.

The named HAL transform passes include:

```text
iree-hal-annotate-target-devices
iree-hal-assign-legacy-target-devices
iree-hal-assign-target-devices
iree-hal-capture-executable-sources
iree-hal-configure-executables
iree-hal-configure-target-executable-variants
iree-hal-conversion
iree-hal-dump-executable-benchmarks
iree-hal-dump-executable-sources
iree-hal-elide-redundant-commands
iree-hal-hoist-executable-objects
iree-hal-initialize-devices
iree-hal-inline-memoize-regions
iree-hal-link-all-executables
iree-hal-link-target-executables
iree-hal-materialize-dispatch-instrumentation
iree-hal-materialize-interfaces
iree-hal-materialize-resource-caches
iree-hal-materialize-target-devices
iree-hal-memoize-device-queries
iree-hal-memoize-device-selection
iree-hal-outline-memoize-regions
iree-hal-preprocess-executables-with-pipeline
iree-hal-preprocess-executables-with-tool
iree-hal-prune-executables
iree-hal-repeat-dispatches
iree-hal-resolve-device-aliases
iree-hal-resolve-device-promises
iree-hal-resolve-export-ordinals
iree-hal-resolve-topology-queries
iree-hal-serialize-all-executables
iree-hal-serialize-target-executables
iree-hal-strip-executable-contents
iree-hal-substitute-executables
iree-hal-translate-all-executables
iree-hal-translate-target-executable-variants
iree-hal-verify-devices
```

Several of these are best read as stages. Device passes choose and verify targets. Executable passes configure, translate, link, and serialize backend code. Memoization and resource cache passes avoid rebuilding device resources. Cleanup and debugging passes elide redundant commands, repeat dispatches for testing, dump sources or benchmarks, capture sources into IR, and strip contents when the full executable payload is too large.

## How To Read HAL IR

When reading HAL, start with the resource handles. Find `!hal.device`, `!hal.allocator`, `!hal.buffer`, `!hal.buffer_view`, `!hal.command_buffer`, `!hal.executable`, and `!hal.fence` values. These are the nouns of the program.

Next, separate recording from submission. `hal.command_buffer.*` operations record commands. `hal.device.queue.*` operations submit or enqueue work on a queue. If you mix those up, HAL will feel much harder than it is.

Then look for the executable path. `hal.executable`, `hal.executable.variant`, `hal.executable.export`, `hal.executable.binary`, and lookup/create ops explain what code is going to run. Interface ops explain how dispatch code reads bindings, constants, workgroup IDs, and workgroup sizes.

Finally, inspect fences. Fences describe ordering and completion. A HAL program with the right buffers and executables can still be wrong if waits and signals are wrong.

The broader lesson is that HAL is where IREE's compiler becomes an executable runtime plan. It preserves enough structure for optimization and target compilation, but it is already speaking in the language of devices, memory, queues, binaries, and synchronization.

# CIRCT `esi` Dialect

The CIRCT `esi` dialect is the Elastic Silicon Interconnect dialect. It is about communication in hardware systems: typed channels, channel bundles, services, host connectivity, metadata manifests, and lowering those abstractions to concrete HW and SystemVerilog structures.

For a beginner, the key idea is that hardware modules should not have to agree manually on every low-level valid, ready, FIFO, or interface wire. ESI lets a design talk in terms of typed messages and services first. Later passes decide how to lower those messages into ports, interfaces, FIFOs, pipeline stages, co-simulation endpoints, and host-visible metadata.

ESI is not a compute dialect like `comb`, `seq`, or `firrtl`. It is a system-integration dialect. It describes how blocks communicate and how software or board-support infrastructure can find and use those communication paths.

## When ESI Is Important

Use ESI when a hardware design needs typed, latency-insensitive communication across module boundaries, between IP blocks, or between hardware and software. It is especially relevant for accelerator systems where kernels, services, board support packages, host memory, telemetry, MMIO, and co-simulation need a common communication model.

ESI is important when you need to answer questions such as:

- What message type flows across this connection?
- Is the connection valid-ready, FIFO-like, or valid-only?
- Does this channel need buffering or pipelining?
- Is this connection part of a service request, such as host memory or telemetry?
- How will software discover this accelerator interface?
- Which AppID path identifies this request or service in the design hierarchy?
- Has the abstract channel model already been lowered to concrete HW/SV ports?

You do not use ESI to describe arithmetic or registers. You use it to describe communication contracts around hardware blocks.

## Why It Is Needed

Traditional RTL interfaces often encode protocol details directly in port lists. One block has `data`, `valid`, and `ready`; another has FIFO `empty` and `rden`; a third exposes a SystemVerilog interface; a host bridge uses a different transport. Without a common abstraction, integration becomes a pile of handwritten adapters.

ESI separates the message layer from the physical signaling layer. A channel can carry a strongly typed payload, while the signaling choice can be lowered later. A service request can name a high-level resource, while a pass bubbles the request through the hierarchy and connects it to an implementation point. A manifest can describe the resulting design to host software.

The implication is that ESI shifts communication integration from informal wiring conventions into typed IR that compiler passes can verify, transform, and lower.

## Type Inventory

The ESI type system extends hardware types with communication and message-shaping types.

- `!esi.channel<T>` is a typed unidirectional message channel carrying `T`. It has a signaling mode: `ValidReady`, `FIFO`, or `ValidOnly`, and can carry a data-delay parameter.
- `!esi.bundle<[...]>` groups named channels with directions `to` and `from`. A common use is a request-response pair where one channel goes to the receiver and the other comes back.
- `!esi.list<T>` represents runtime-variable-sized data. In hardware, this is naturally sent over multiple cycles.
- `!esi.any` is a placeholder used in service declarations when the concrete type is selected later.
- `!esi.window<...>` is a data window. It describes how a large logical message is broken into frames.
- `!esi.window.frame<...>` declares a frame inside a data window.
- `!esi.window.field<...>` names a field included in a frame, optionally with an item count or bulk-transfer count width.

The most common type you will see is `!esi.channel<T>`. The others become important when a design has bidirectional service ports, variable-length messages, or wide payloads that should be moved over several cycles.

## Attribute Inventory

The main ESI attributes are:

- `#esi.appid<...>` identifies an application-relevant instance. AppIDs let later passes and software-facing metadata find meaningful components through hierarchy.
- `#esi.appid_path<...>` records a hierarchical path of AppID components rooted at a top symbol.
- `#esi.blob<...>` stores a binary blob, used for compressed manifest data.
- `#esi.null<...>` marks an unconnected channel value.

The channel type also uses enum-like values for signaling and direction. Channel signaling can be `ValidReady`, `FIFO`, or `ValidOnly`. Bundle channel direction can be `to` or `from`, where the direction is described from the perspective of the bundle sender or service.

## Operation Inventory

The local ESI dialect defines 43 operations. The easiest way to learn them is by role.

### Channel Observation and Wrapping

`esi.wrap.vr` wraps raw data plus a valid bit into a valid-ready channel and returns the downstream ready signal.

`esi.unwrap.vr` unwraps a valid-ready channel with a supplied ready signal and returns raw data plus valid.

`esi.wrap.fifo` wraps raw data and an empty signal into a FIFO-signaled channel and returns a read-enable signal.

`esi.unwrap.fifo` unwraps a FIFO-signaled channel with a read-enable signal and returns raw data plus empty.

`esi.wrap.vo` wraps raw data and valid into a valid-only channel. Valid-only has no backpressure.

`esi.unwrap.vo` unwraps a valid-only channel and returns raw data plus valid.

`esi.snoop.vr` observes a channel's valid, ready, and data signals without becoming a channel consumer.

`esi.snoop.xact` observes transaction occurrence and data without becoming a channel consumer.

These operations are the boundary between abstract channel values and signal-level logic. Wraps and unwraps are also canonicalized when they meet directly.

### Interfaces, Buffers, FIFOs, and Endpoints

`esi.wrap.iface` wraps a SystemVerilog interface modport into an ESI channel.

`esi.unwrap.iface` unwraps an ESI channel into a SystemVerilog interface modport.

`esi.buffer` is an abstract channel buffer. It says a connection should receive buffering or pipeline stages, optionally with a stage count and name.

`esi.stage` is a physical elastic pipeline stage. `lower-esi-to-physical` lowers `esi.buffer` into one or more `esi.stage` operations.

`esi.fifo` is a physical FIFO connecting ESI channels with FIFO-compatible behavior. The physical lowering turns it into a `seq.fifo` and wrapping logic.

`esi.cosim.to_host` and `esi.cosim.from_host` are co-simulation endpoints. They connect simulated hardware channels to an outside software process.

`esi.null` creates a channel that never produces messages.

### Bundles and Data Windows

`esi.bundle.pack` packs named channels into an `!esi.bundle`.

`esi.bundle.unpack` unpacks an `!esi.bundle` into its constituent channels.

`esi.window.wrap` wraps a lowered frame value into an ESI data window.

`esi.window.unwrap` unwraps a data window into the lowered frame representation.

Bundles are used for grouped interfaces such as request-response services. Windows are used when a logical payload should be transferred as multiple frames rather than one wide data value.

### Services

`esi.service.decl` declares a custom service interface at module scope.

`esi.service.port` declares a named service port inside `esi.service.decl`.

`esi.service.req` requests a connection to a service port. The result is a channel bundle for the client.

`esi.service.instance` instantiates a service implementation point. It may target a specific service symbol or act as a default implementation point.

`esi.service.impl_req` is the canonical implementation request produced by `esi-connect-services`. It gathers the surfaced client requests that must be implemented at a service instance.

`esi.service.impl_req.req` records one surfaced client request inside a service implementation request.

The beginner model is that clients emit `esi.service.req`, implementations emit `esi.service.instance`, and `esi-connect-services` rewrites the design so requests are gathered and routed to implementation points.

### Standard Services

`esi.mem.ram` declares a random-access memory service with read and write ports.

`esi.service.std.channel` declares a simple host-to-device or device-to-host channel service.

`esi.service.std.func` declares a function-call-style service exposed by a hardware client to software.

`esi.service.std.call` declares a service against which hardware can call into software.

`esi.service.std.mmio` declares an MMIO-backed service expected to be implemented by a board support package.

`esi.service.std.hostmem` declares host-memory read/write service ports, useful for DMA-like access.

`esi.service.std.telemetry` declares a telemetry reporting service.

These are built-in service declarations. They do not make the hardware implementation by themselves; they define the typed service contract that later code or BSP-specific generators can implement.

### Pure Modules

`esi.pure_module` describes an ESI-only module with no ordinary top-level ports. Its non-local interaction is expected to happen through services.

`esi.pure_module.input` creates an input that becomes an HW module input when the pure module is lowered.

`esi.pure_module.output` creates an output that becomes an HW module output when the pure module is lowered.

`esi.pure_module.param` creates a parameter that becomes an HW module parameter during lowering.

Pure modules are useful for service-generated modules and simulation-oriented designs where the external top-level wiring is supplied later.

### Manifest and Metadata

`esi.manifest.req` records an original service request for software-visible metadata.

`esi.manifest.service_impl` records a service implementation and details needed to connect to it.

`esi.manifest.impl_conn` records per-client connection details under a service implementation record.

`esi.manifest.hier_root` is the root of an AppID instance hierarchy.

`esi.manifest.hier_node` is a node in that AppID hierarchy.

`esi.manifest.constants` stores constant values associated with a symbol.

`esi.manifest.sym` stores optional symbol metadata such as name, repository, commit, version, and summary.

`esi.manifest.compressed` stores the zlib-compressed JSON manifest as a blob.

The manifest operations are how the hardware-side design communicates its shape and access points to software tools.

## Transformations and Conversions

ESI lowering is staged because communication abstractions need different decisions at different times.

`verify-esi-connections` checks that channel and bundle values obey the single-consumer rules. Bundles must have exactly one use; channels are verified through channel-type rules.

`esi-connect-services` connects service requests to service providers. It converts `esi.service.req` into implementation connection requests, surfaces those requests through hierarchy, records `esi.manifest.req`, and replaces service instances with `esi.service.impl_req`.

`esi-appid-hier` builds an AppID hierarchy rooted at a chosen top module. It uses AppID paths and clones manifest data into `esi.manifest.hier_root` and `esi.manifest.hier_node` structure.

`esi-build-manifest` builds a JSON-style ESI system manifest. It serializes service declarations, service implementations, type information, channel bundles, AppIDs, symbols, windows, and other manifest data. The result can be stored as `esi.manifest.compressed`.

`esi-clean-metadata` erases service declarations and manifest metadata once that information is no longer needed in the hardware IR.

`lower-esi-bundles` lowers channel bundle ports into individual channel ports. It inserts `esi.bundle.pack` and `esi.bundle.unpack` during conversion, then canonicalizes them away.

`lower-esi-to-physical` lowers abstract ESI ops to physical ESI ops. In the local implementation, it lowers `esi.buffer` into `esi.stage`, lowers `esi.fifo` toward `seq.fifo` plus wrappers, and lowers `esi.pure_module` into `hw.module`.

`lower-esi-ports` lowers ESI channel ports on HW modules and extern modules. Depending on the selected channel signaling, it explodes channel ports into valid-ready, FIFO, valid-only, or SystemVerilog interface forms.

`lower-esi-types` lowers high-level ESI data types, especially data windows. It inserts `esi.window.wrap` and `esi.window.unwrap` materializations and rewrites types nested in structs, arrays, aliases, and ports.

`lower-esi-to-hw` lowers remaining physical ESI operations to HW/SV where possible. It handles pipeline stages, null sources, snoops, SystemVerilog interface wrapping, co-simulation endpoints, and manifest ROM-style lowering. It also verifies that channel-typed values are gone when they should be.

A typical flow verifies channels, connects services, builds metadata, lowers bundles and ports, lowers abstract physical choices, lowers ESI-specific types, and finally lowers to HW/SV.

## What ESI Implies

When you see `!esi.channel<T>`, the design is still working at the message level. The compiler has not yet fully committed to individual wires.

When you see `esi.wrap.vr` and `esi.unwrap.vr`, you are near the boundary where message-level channels become valid-ready signals.

When you see `esi.service.req`, some module wants access to a shared service without manually wiring that service through every parent module. After service connection, look for `esi.service.impl_req` and manifest records.

When you see `esi.manifest.*`, the design is carrying software-facing metadata. This is not hardware datapath logic; it is information for host APIs, runtime discovery, or system description.

When you see `esi.buffer`, ESI is still asking for a buffering policy. When you see `esi.stage`, that policy has become concrete pipeline structure.

## How To Read ESI IR

Start with channel and bundle types. They tell you what messages are flowing and whether a connection is unidirectional or request-response-like.

Then inspect service declarations and service requests. A service declaration tells you the contract. A service request tells you which client asks for that contract. A service instance or implementation request tells you where implementation is expected.

Next look at lowering state. If bundles, windows, services, and channels are still present, you are before final HW/SV lowering. If only valid-ready wires, SV interfaces, FIFOs, and HW modules remain, ESI has mostly done its job.

Finally inspect AppID and manifest records. These records connect hardware hierarchy to software-facing names and paths.

## Minimal Example

This small example shows the channel idea:

```mlir
hw.module @PassThrough(%in: !esi.channel<i32>)
    -> (out: !esi.channel<i32>) {
  %ready = hw.constant true
  %data, %valid = esi.unwrap.vr %in, %ready : !esi.channel<i32>
  %chan, %downstream_ready = esi.wrap.vr %data, %valid : i32
  hw.output %chan : !esi.channel<i32>
}
```

The channel type says the module communicates with typed messages. The wrap and unwrap operations expose a valid-ready implementation at the edge of the computation. Later ESI passes can lower the channel port into concrete RTL signals or an interface.

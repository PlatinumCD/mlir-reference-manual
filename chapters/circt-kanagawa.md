# CIRCT `kanagawa` Dialect

The CIRCT `kanagawa` dialect supports porting and eventually open-sourcing an internal hardware development language. It models a class-like hardware language with methods, member variables, static schedulable blocks, containers, ports, hierarchy paths, and a lowering flow toward CIRCT hardware IR.

For a beginner, the most useful mental model is this: Kanagawa starts as an object- and method-shaped hardware language, then progressively becomes a container-and-port hardware representation, and finally lowers to `hw.module` operations. It is not just a netlist dialect. It has high-level concepts such as classes, methods, calls, member variables, and static blocks, and low-level concepts such as port references, containers, and instance hierarchy tunneling.

The local checkout defines 26 `kanagawa` operations, two custom types, one path-step attribute, two enum attributes, and 15 Kanagawa-owned passes.

## When Kanagawa Is Important

`kanagawa` is important when CIRCT is compiling Kanagawa-style hardware source or IR that still contains class structure, method calls, relative port references, or static scheduling regions.

Use this dialect when you need to answer questions like:

- Which class, method, variable, or instance does this operation refer to?
- Is this still high-level method/control-flow IR, or has it become dataflow IR?
- Has a `kanagawa.sblock` been isolated and made ready for scheduling?
- Does a `kanagawa.path` still describe a relative hierarchy reference?
- Are `!kanagawa.portref` values still being tunneled through the hierarchy?
- Has the design been containerized and lowered to `hw.module`?

You usually encounter this dialect inside Kanagawa-specific flows rather than in generic CIRCT examples. The dialect is useful because it records source-language structure that plain `hw`, `comb`, `seq`, or `sv` would not preserve.

## Why It Is Needed

Many hardware languages have both software-like and hardware-like structure. They may have classes, methods, local variables, member variables, static blocks of operations that should be scheduled together, and references to ports across an instance hierarchy.

Lowering all of that directly to `hw.module` would erase too much too early. Kanagawa keeps the source model explicit while the compiler performs transformations such as inlining static blocks, maximizing SSA form, reconstructing schedulable blocks, converting methods to dataflow, tunneling port references, and finally creating hardware modules.

The implication is that `kanagawa` is a staged lowering dialect. Seeing Kanagawa IR means the compiler is still preserving higher-level structure or working through port/hierarchy resolution.

## Types And Attributes

`!kanagawa.scoperef` is a reference to a Kanagawa scope. It may be opaque or may carry an inner reference to a specific scope. Instances and paths produce scope references, and operations such as `kanagawa.get_port` and `kanagawa.get_var` use them to access members of a referenced scope.

`!kanagawa.portref<in T>` and `!kanagawa.portref<out T>` are references to Kanagawa ports. The direction says how the reference is intended to be used. A port reference is not the same thing as the payload value `T`; it is a handle that can later be read, written, tunneled, or lowered.

The `Direction` enum has `in` and `out` cases for port references and port definitions.

The `PathDirection` enum has `parent` and `child` cases for walking relative hierarchy paths.

`#kanagawa.step<...>` is the path-step attribute used by `kanagawa.path`. A path step records whether to move to a parent or child scope, the step type, and optionally the child symbol.

The dialect also defines several interfaces for named inner symbols, ports, scopes, static blocks, and method-like operations. These interfaces matter because many Kanagawa references are inner-symbol references rather than ordinary MLIR symbols.

## Operation Inventory

The local `kanagawa` dialect defines 26 operations.

### Design, Classes, Instances, And Methods

`kanagawa.design` is the top-level Kanagawa design container. It is isolated, has a symbol name, contains an inner symbol table, and holds Kanagawa classes and low-level containers.

`kanagawa.class` models a Kanagawa class. A class can contain methods, member variables, ports, containers, and logic for member variables. It is a graph region and participates in CIRCT's instance graph as a module-like operation.

`kanagawa.instance` instantiates a Kanagawa class and returns a `!kanagawa.scoperef` to the instance. It carries an inner symbol for the instance name and an inner reference to the target class.

`kanagawa.method` is a function-like Kanagawa method with named arguments, unnamed results, and imperative control flow.

`kanagawa.method.df` is the dataflow form of a method. It has the same method-like interface as `kanagawa.method`, but its body is a graph region and control flow is expected to be represented with dataflow operations, especially during Handshake/DC lowering.

`kanagawa.return` terminates `kanagawa.method` and `kanagawa.method.df` bodies.

`kanagawa.call` dispatches a call to a Kanagawa method through an inner reference. It implements the MLIR call interface.

`kanagawa.var` defines a class member variable. Its type is a memref type, and it can represent a singleton or one-dimensional array of values.

`kanagawa.get_var` dereferences a member variable through a `!kanagawa.scoperef`.

### Static Blocks And Scheduling Markers

`kanagawa.sblock` defines a group of operations expected to be statically schedulable. It is not initially isolated from above, which makes construction easier.

`kanagawa.sblock.isolated` is the isolated form of `kanagawa.sblock`. Values used inside the block are passed as block arguments, and values produced by the block are passed as results.

`kanagawa.sblock.dc` is a DC-interfaced static block. Its inputs and outputs use `dc.value`-typed wrappers, and the block is isolated from above.

`kanagawa.sblock.return` terminates `kanagawa.sblock`, `kanagawa.sblock.isolated`, and `kanagawa.sblock.dc`.

`kanagawa.sblock.inline.begin` and `kanagawa.sblock.inline.end` are marker operations used while static blocks are temporarily inlined into method CFG form. They preserve the boundaries and attributes of a source `kanagawa.sblock` while passes such as mem2reg and SSA maximization operate on the method.

`kanagawa.pipeline.header` produces the hardware-like control values used to drive a scheduled pipeline: clock, reset, go, and stall. It is an intermediate operation used while static blocks are prepared for pipeline scheduling.

### Paths, Containers, Ports, And Wires

`kanagawa.path` describes a relative path through the instance hierarchy. It uses `#kanagawa.step` attributes to move to parent or child scopes and returns a `!kanagawa.scoperef` for the final scope.

`kanagawa.container` is the low-level Kanagawa module-like container. It describes logic nested within a class or design and is the operation that eventually converts to `hw.module`.

`kanagawa.container.instance` instantiates a `kanagawa.container` and returns a `!kanagawa.scoperef`.

`kanagawa.get_port` takes a scope reference and a port symbol and returns a `!kanagawa.portref`. The requested portref direction records how the caller intends to use the port: read from it or write to it. That requested direction is independent of the target port's declared direction, which is why tunneling and portref lowering need to inspect uses.

`kanagawa.port.input` declares an input port in a class or container and returns a port reference.

`kanagawa.port.output` declares an output port in a class or container and returns a port reference.

`kanagawa.wire.input` defines an input-style wire. It returns both a `!kanagawa.portref<in T>` and a value of type `T` representing the value to be written to the wire.

`kanagawa.wire.output` defines an output-style wire. It takes a value of type `T` and returns a `!kanagawa.portref<out T>` that can be read.

`kanagawa.port.read` reads the payload value from a port reference.

`kanagawa.port.write` writes a payload value to a port reference.

## Transformations And Conversions

Kanagawa has 15 named passes in this checkout. They form two related stories: high-level method/block cleanup and low-level container/port lowering.

`kanagawa-inline-sblocks` inlines `kanagawa.sblock` operations into MLIR blocks and `cf` operations. It preserves static-block metadata on the parent operation under `kanagawa.blockinfo`.

`kanagawa-reblock` reconstructs `kanagawa.sblock` operations from CFG blocks after the IR has been put into maximal SSA form. It expects calls to be isolated into their own basic blocks and turns statically schedulable regions back into explicit Kanagawa blocks.

`kanagawa-argify-blocks` converts values captured from outside a `kanagawa.sblock` into explicit block arguments, producing `kanagawa.sblock.isolated` operations.

`kanagawa-prepare-scheduling` prepares isolated static blocks for scheduling by creating `kanagawa.pipeline.header` and moving operations into a pipeline unscheduled region connected to that header.

`kanagawa-convert-cf-to-handshake` converts a `kanagawa.method` using `cf` blocks into `kanagawa.method.df` using Handshake fine-grained dataflow operations.

`kanagawa-convert-handshake-to-dc` converts `kanagawa.method.df` from Handshake operations to DC operations and legalizes blocks into `kanagawa.sblock.dc` where needed.

`kanagawa-convert-methods-to-containers` converts dataflow methods into `kanagawa.container` operations.

`kanagawa-call-prep` prepares method calls by converting them to use `dc.value` values.

`kanagawa-add-operator-library` injects the Kanagawa operator library into the module. The implementation adds an `ssp`-based operator library symbol used during scheduling.

`kanagawa-containerize` converts high-level classes to containers. It outlines containers nested inside classes, converts `kanagawa.class` to `kanagawa.container`, and converts `kanagawa.instance` to `kanagawa.container.instance`.

`kanagawa-eliminate-redundant-ops` removes redundant operations inside containers, especially duplicate operations that would otherwise cause problems for tunneling and later lowering.

`kanagawa-tunneling` resolves relative `kanagawa.get_port` requests through `kanagawa.path` operations. It creates new in/out `!kanagawa.portref<...>` ports through the instance hierarchy so that port references can be forwarded to the scope where they are needed.

`kanagawa-lower-portrefs` lowers nested port references such as `!kanagawa.portref<in !kanagawa.portref<out T>>` to ordinary `T`-typed container ports by analyzing reads and writes. For example, reading a referenced output becomes an input value on the current container, while writing a referenced input becomes an output value that drives the target.

`kanagawa-clean-selfdrivers` removes cases where a container ends up driving its own input ports or reading an instance port that is also written within the same container.

`kanagawa-convert-containers-to-hw` converts `kanagawa.container` operations to `hw.module` operations. It gathers `kanagawa.port.input` and `kanagawa.port.output` operations into an HW module interface, replaces port reads with block arguments, replaces output ports with `hw.output` operands, and converts container instances into HW instances.

## Lowering Flow

The low-level Kanagawa pipeline is:

1. Verify the inner reference namespace.
2. Run `kanagawa-containerize`.
3. Remove redundant container operations with `kanagawa-eliminate-redundant-ops`.
4. Run `kanagawa-tunneling`.
5. Run `kanagawa-lower-portrefs`.
6. Canonicalize and run `kanagawa-eliminate-redundant-ops` again.
7. Run `kanagawa-clean-selfdrivers`.
8. Run `kanagawa-convert-containers-to-hw`.

The high-level pipeline is:

1. Run `kanagawa-inline-sblocks`.
2. Run mem2reg and CIRCT SSA maximization.
3. Run `kanagawa-reblock`.
4. Run `kanagawa-argify-blocks`.
5. Canonicalize.

These pipelines show the role of the dialect. High-level structure is temporarily normalized into CFG/SSA form, reconstructed into schedulable blocks, then lowered through containers, tunneled ports, and finally `hw.module`.

## What It Implies

Seeing `kanagawa.class`, `kanagawa.method`, `kanagawa.var`, or `kanagawa.call` means the IR is still close to the source language's class and method model.

Seeing `kanagawa.sblock` means a set of operations is intended to be statically scheduled together. Seeing `kanagawa.sblock.isolated` means the block has explicit arguments and results and is closer to scheduling. Seeing `kanagawa.sblock.dc` means the block is already in the DC-interfaced stage.

Seeing `kanagawa.path` and `kanagawa.get_port` means relative hierarchy and port-reference lowering still remain. These are the operations that `kanagawa-tunneling` and `kanagawa-lower-portrefs` are designed to remove or simplify.

Seeing `kanagawa.container` and `kanagawa.container.instance` means class-level structure has been containerized, but the design has not yet become plain HW.

Seeing `kanagawa.port.input`, `kanagawa.port.output`, `kanagawa.port.read`, and `kanagawa.port.write` means Kanagawa ports are still explicit as port references. After `kanagawa-convert-containers-to-hw`, those should become HW module ports, block arguments, instance operands/results, and `hw.output`.

## Minimal Example

This simplified fragment declares a design with a class, an input port, a method, and a static block:

```mlir
kanagawa.design @Example {
  kanagawa.class sym @Counter {
    %in = kanagawa.port.input "value" sym @value : i32

    kanagawa.method sym @step(%arg0: i32) -> (i32) {
      %r = kanagawa.sblock(%arg0) : i32 -> i32 {
        kanagawa.sblock.return %arg0 : i32
      }
      kanagawa.return %r : i32
    }
  }
}
```

The important reading is structural. `kanagawa.class` says this is still class-shaped hardware. `kanagawa.method` says behavior is still method-shaped. `kanagawa.sblock` marks a schedulable region. Later passes can inline the block, optimize SSA form, reconstruct isolated blocks, convert methods to dataflow, containerize the class, tunnel ports, and finally lower containers to `hw.module`.


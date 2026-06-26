# CIRCT `interop` Dialect

The CIRCT `interop` dialect represents partially lowered interoperability layers between CIRCT backends, external simulators, generated C or C++ code, and other tools. Its job is to avoid custom pairwise glue for every combination of tool and backend. Instead, an instance-side lowering can describe "this external thing needs persistent state, initialization, update, and cleanup" in a common form, and a container-side lowering can decide how to embed that into the surrounding target.

For a beginner, the key idea is that `interop` is a bridge dialect. It is not a hardware dialect like `hw`, not a simulator dialect like `sim`, and not a C++ dialect like `systemc` or `emitc`. It is a common intermediate layer for partially lowered procedural interoperability. It lets CIRCT say: allocate state for an external instance, initialize that state, call an update body each time the surrounding container runs, and clean the state up later.

This matters because hardware tools often need to mix worlds. A design may be mostly lowered through CIRCT, while one instance is simulated by Verilator, another is exposed through a C ABI, and another is emitted as textual C++. Without a common interop layer, every pair of producer and container would need bespoke lowering code.

## Operation Inventory

The `interop` dialect defines these operations:

```text
interop.procedural.alloc
interop.procedural.init
interop.procedural.update
interop.procedural.dealloc
interop.return
```

It also defines an `InteropMechanism` enum attribute with these mechanisms:

```text
cffi
cpp
```

`cffi` means C foreign function interface. `cpp` means textual C++.

## The Procedural Lifecycle

The four procedural operations form a lifecycle.

`interop.procedural.alloc` requests persistent state. It returns a variadic list of SSA values representing state that must live across multiple executions of `interop.procedural.update`. For example, a Verilated module instance may need a C++ pointer that remains alive across simulation cycles.

`interop.procedural.init` computes the initial values for state returned by `alloc`. It takes state values as operands and has a body ending in `interop.return`. The returned values are assigned to the state by the container-side lowering. This split is intentional: `alloc` creates SSA handles for persistent state, while `init` explains how those handles are initialized.

`interop.procedural.update` is the repeated execution body. It takes state operands and ordinary input operands, exposes them as block arguments, and returns ordinary output values through `interop.return`. If state must be mutated, the state type should be a pointer or handle type; the operation uses pass-by-value semantics for state block arguments.

`interop.procedural.dealloc` runs cleanup logic before state is released. Structurally it resembles update, but it has no ordinary inputs or outputs. It receives state values and has a body for cleanup actions such as deleting a C++ object or freeing memory.

`interop.return` terminates `interop.procedural.init` and `interop.procedural.update`. In `init`, its operands are the initial state values. In `update`, its operands are the computed outputs. `dealloc` has no terminator requirement in the TableGen definition.

## Interop Mechanisms

Every lifecycle operation carries an interop mechanism attribute. This says which world the operation body and state types currently belong to.

With `cpp`, the body may use textual C++-oriented dialects such as SystemC and EmitC. State might be an `!emitc.ptr<!emitc.opaque<"...">>`, and the body might allocate with a C++ `new`, assign C++ members, call an `eval` member function, or delete the object.

With `cffi`, the intended boundary is a C foreign function interface. State is expected to be represented in C-compatible terms, often as an opaque pointer. The rationale describes bridging from `cpp` to `cffi` by moving a C++ body behind an `extern "C"` function and replacing state types with opaque pointers where necessary.

The mechanism attribute is important because it prevents the IR from pretending that all tools accept the same types and operations. A C++ pointer to a Verilated class is meaningful in a C++ emission path. A C ABI may only be able to see it as an opaque pointer.

## When To Use It

Use `interop` when a dialect lowers an external instance but cannot or should not decide exactly how the surrounding container will host it. For example, a SystemC interop operation can lower a Verilated instance into the four procedural lifecycle operations. Later, a SystemC container lowering can move allocation into a module field, initialization into a constructor, update logic into a method, and deallocation into a destructor.

Do not use `interop` to describe normal hardware structure. Use `hw`, `comb`, `seq`, `sv`, `systemc`, or other target dialects for concrete structure and behavior. `interop` is the temporary contract between an instance lowering and a container lowering.

It is especially useful when a tool wants to support multiple external backends or testbench interfaces without writing a custom conversion for every pair. The dialect's rationale uses simulation as the motivating example: a testbench should not have to be rewritten for Verilator, LLHD simulation, and another simulator if CIRCT can generate standardized interface layers.

## SystemC Instance-Side Lowering

The concrete lowering in this checkout is `systemc-lower-instance-interop`. It lowers `systemc.interop.verilated` into Interop procedural operations.

The lowering creates an `emitc` pointer type for the Verilated module class, using a name like `V<module>`. It emits an include for the generated Verilator header. Then it creates:

```text
interop.procedural.alloc
interop.procedural.init
interop.procedural.update
interop.procedural.dealloc
```

The `alloc` op requests a persistent C++ pointer to the Verilated module. The `init` body creates the object with a C++ new-style operation and returns the pointer. The `update` body writes input values into Verilated module fields, calls the `eval` member, reads output fields, and returns them. The `dealloc` body deletes the object.

This pass is an instance-side lowering. It does not fully decide where persistent state will live in the final SystemC module. It produces a structured interop representation that a later container-side lowering can embed properly.

## Container-Side Lowering

The rationale describes container-side lowering as the second half of the design. A container operation, such as a `systemc.module`, implements an interface that knows how to lower allocation, initialization, update, and deallocation in that container.

For SystemC, this means allocation can become a C++ member variable, initialization can move into a constructor, update can become code inside a method, and deallocation can move into a destructor. The key is that the closest surrounding container decides where each lifecycle piece belongs.

The rationale also explains why the dialect uses separate lifecycle operations instead of one operation with multiple regions. Separate operations can be moved independently. Allocation can be hoisted to a top-level field, initialization can be moved to a constructor, update can remain inside an executable method, and deallocation can move to a destructor.

## Transformations And Conversions

The Interop dialect itself does not define local passes in this checkout. Its transformations are provided by dialects that produce or consume Interop operations.

The visible producer pass is:

```text
systemc-lower-instance-interop
```

That pass uses `populateSystemCLowerInstanceInteropPatterns` and converts `systemc.interop.verilated` into the Interop procedural lifecycle.

The rationale describes a broader architecture:

```text
instance-side lowering -> interop procedural ops -> container-side lowering
```

Instance-side lowering belongs to the dialect that knows the external instance. Container-side lowering belongs to the dialect that knows how to host lifecycle code. Bridging patterns can translate between interop mechanisms, such as from `cpp` to `cffi`, when the instance body and the container support different mechanisms.

For a beginner, the important implication is that Interop operations are usually not the final target. They are coordination points. If they remain in the IR near the end of a flow, it usually means a needed container-side lowering or mechanism bridge has not run yet.

## How To Read interop IR

Start by finding `interop.procedural.alloc`. Its result values identify the persistent state for an external instance. Then find the matching `init`, `update`, and `dealloc` operations that consume those state values.

Next, check the mechanism: `cpp` means the body is currently valid in a textual C++ world; `cffi` means the intended interface is C-compatible. If the surrounding container only supports another mechanism, a bridge is needed.

Finally, inspect `interop.procedural.update`. Its inputs and results are the functional boundary of the external instance. The state operands tell you what persists, while the update inputs and outputs tell you what the surrounding design exchanges with the external tool on each execution.

The dialect implies partial lowering. It says "we know the lifecycle and behavior of this interop instance, but the final hosting location still belongs to a later lowering stage."

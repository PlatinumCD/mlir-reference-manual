# CIRCT Calyx Dialect

The CIRCT `calyx` dialect models the Calyx intermediate language inside MLIR. Calyx is an intermediate language for hardware accelerator compilers. Its central idea is to combine a structural hardware language with a software-like control language.

For a beginner, the useful mental model is this: Calyx separates what hardware exists from when that hardware runs. The structural side contains components, ports, cells, memories, registers, wires, and assignments. The control side contains a schedule made from `seq`, `par`, `if`, `while`, `repeat`, `invoke`, and `enable` operations. A group of assignments describes a hardware action, and the control program decides when groups execute.

The local checkout defines 68 `calyx` operations and 18 Calyx-related passes.

## When Calyx Is Important

`calyx` is important in high-level synthesis and accelerator compiler flows. It is useful when a compiler has a program with loops, memory accesses, arithmetic, function-like calls, or parallel regions and wants to generate custom hardware with an explicit schedule.

Use this dialect when you need to answer questions like:

- What hardware resources are allocated for this accelerator?
- Which assignments run as one scheduled action?
- Is this operation part of structural datapath logic or control scheduling?
- Are two groups meant to run sequentially or in parallel?
- Has the schedule been compiled into group go/done wiring or an FSM?
- Are registers, memories, and library operators still Calyx cells?
- Has Calyx been lowered to `hw`, `comb`, `seq`, `sv`, or `fsm`?

You usually encounter this dialect as a compilation target from SCF, LoopSchedule, Dahlia-like frontends, or other accelerator-oriented IRs. It is also useful as a readable way to inspect HLS output before it becomes lower-level hardware.

## Why It Is Needed

Plain RTL can represent the final circuit, but it is not always a good place to express scheduling decisions. A compiler for custom hardware often wants to say: "these assignments form one action," "this action repeats," "these actions can run together," or "call this subcomponent and wait for its done signal." Those ideas are awkward if the compiler lowers immediately to wires, registers, and FSM state bits.

The `calyx` dialect keeps the structural and scheduling layers explicit. A `calyx.group` contains assignments that execute together. A `calyx.control` region says when groups run. Component interfaces have conventional `go`, `done`, `clk`, and `reset` ports. Cells expose named ports as SSA values, which lets MLIR transformations track data dependencies while preserving Calyx's hardware model.

The implication is that Calyx is not only a netlist dialect. It is a scheduled hardware IR. Seeing Calyx usually means the compiler is still carrying both datapath structure and a control program.

## Types And Attributes

This checkout does not define custom Calyx types in TableGen. Calyx operations use ordinary MLIR and CIRCT value types such as signless integers, function types, and hardware-related types from dependent dialects.

The important Calyx metadata is carried through operation attributes and interfaces. Component ports have names, directions, and dictionaries of port attributes. Common port attributes include `clk`, `reset`, `go`, and `done`, which identify the control interface required by a normal `calyx.component`.

Cells also expose port information. A register, for example, has `in`, `write_en`, `clk`, `reset`, `out`, and `done` ports. A memory has address, data, enable, clock, read data, and done-style ports. Library operators expose their input and output ports as SSA results.

The dialect uses a `Direction` concept with input and output cases. It also recognizes Calyx-specific attributes such as `external`, `static`, `share`, `bound`, `generated`, and `precious` during emission or lowering. Static groups carry latency information, and control operations such as repeat carry iteration counts.

For a beginner, the key point is that a Calyx SSA value often represents a named hardware port, not just a computed value. When you see `%r.in` or `%r.done`, that is a port of a cell, and assignments connect ports together.

## Operation Inventory

This checkout defines 68 `calyx` operations. They fall into four broad groups: structural containers, scheduled groups and assignments, control operations, and primitive/library cells.

### Components, Cells, Wires, And Groups

These operations define the main structural shape of a Calyx program:

```text
calyx.component
calyx.comb_component
calyx.wires
calyx.instance
calyx.primitive
calyx.group
calyx.comb_group
calyx.static_group
calyx.assign
calyx.group_done
calyx.group_go
calyx.cycle
calyx.undef
```

`calyx.component` is a stateful Calyx component with input and output ports, cells, wires, groups, and a control schedule. It normally has input ports marked `clk`, `reset`, and `go`, plus an output port marked `done`.

`calyx.comb_component` is the combinational version of a component. It contains combinational cells and wires but no full dynamic control schedule.

`calyx.wires` contains structural wiring, groups, combinational groups, static groups, and assignments. It is where the component's datapath is wired together.

`calyx.instance` instantiates another Calyx component. `calyx.primitive` instantiates an external primitive represented by an `hw.module.extern`.

`calyx.group` is a named scheduled action. It contains `calyx.assign` operations and ends with `calyx.group_done`. `calyx.comb_group` is a combinational group used to compute a condition or immediate value. `calyx.static_group` is a group with a known latency, and `calyx.cycle` creates a one-bit signal active at a specific cycle within that static group.

`calyx.assign` is a guarded or unguarded non-blocking assignment between ports. `calyx.group_go` and `calyx.group_done` model the start and completion handshake for a group. `calyx.undef` is a generic undefined value used before a real source is connected.

### Control Schedule

These operations describe when groups, components, and subprograms run:

```text
calyx.control
calyx.enable
calyx.seq
calyx.static_seq
calyx.par
calyx.static_par
calyx.if
calyx.static_if
calyx.while
calyx.repeat
calyx.static_repeat
calyx.invoke
```

`calyx.control` is the root of a component's schedule. It says how the component's groups execute.

`calyx.enable` runs one named group. `calyx.seq` runs nested control operations sequentially. `calyx.par` runs nested control operations in parallel and completes when all children complete. The `static_seq`, `static_par`, `static_if`, and `static_repeat` forms represent statically timed versions of the same ideas.

`calyx.if` and `calyx.while` use a one-bit condition port. They may reference a combinational group that computes the condition. `calyx.repeat` repeats a dynamic body a fixed number of times, while `calyx.static_repeat` repeats a statically timed body. `calyx.invoke` calls another component, maps input ports, and can map reference cells such as external memories.

This is the part of Calyx that feels software-like. The operations are not arithmetic themselves. They schedule hardware actions.

### Core Primitive Cells

These operations define common stateful or structural cells:

```text
calyx.constant
calyx.undefined
calyx.register
calyx.memory
calyx.seq_mem
```

`calyx.constant` creates a constant cell with an integer or floating-point attribute. `calyx.undefined` is the library-style undefined signal cell. `calyx.register` defines a register with input, write enable, clock, reset, output, and done ports. `calyx.memory` defines a memory with combinational read behavior, while `calyx.seq_mem` defines a memory with sequential read behavior.

Memory ops can have multiple dimensions. Their size attributes describe each dimension, and their address-size attributes describe the width needed to address each dimension.

### Combinational Standard Library Cells

These operations model standard Calyx library primitives:

```text
calyx.std_add
calyx.std_sub
calyx.std_and
calyx.std_or
calyx.std_xor
calyx.std_lsh
calyx.std_rsh
calyx.std_shru
calyx.std_srsh
calyx.std_lt
calyx.std_le
calyx.std_gt
calyx.std_ge
calyx.std_eq
calyx.std_neq
calyx.std_slt
calyx.std_sle
calyx.std_sgt
calyx.std_sge
calyx.std_seq
calyx.std_sneq
calyx.std_mux
calyx.std_pad
calyx.std_slice
calyx.std_not
calyx.std_wire
calyx.std_signext
```

These are hardware cells rather than normal MLIR arithmetic ops. For example, `calyx.std_add` exposes left, right, and out ports. `calyx.std_mux` exposes condition, true, false, and output ports. Shift, compare, pad, slice, not, wire, and sign-extension cells follow the same port-based style.

The unsigned comparison cells include `calyx.std_lt`, `calyx.std_le`, `calyx.std_gt`, `calyx.std_ge`, `calyx.std_eq`, and `calyx.std_neq`. The signed comparison cells include `calyx.std_slt`, `calyx.std_sle`, `calyx.std_sgt`, `calyx.std_sge`, `calyx.std_seq`, and `calyx.std_sneq`.

### Pipelined And Floating-Point Library Cells

These operations model multi-cycle arithmetic and IEEE 754-style primitives:

```text
calyx.std_mult_pipe
calyx.std_divs_pipe
calyx.std_divu_pipe
calyx.std_rems_pipe
calyx.std_remu_pipe
calyx.ieee754.add
calyx.ieee754.mul
calyx.ieee754.divSqrt
calyx.ieee754.compare
calyx.ieee754.fpToInt
calyx.ieee754.intToFp
```

The pipelined standard cells have `clk`, `reset`, `go`, input, output, and `done` ports. They are not pure combinational operations. The floating-point cells model Berkeley HardFloat-style interfaces and expose control, rounding, exception, and done ports where needed.

### Complete Op List

For reference, the exact Calyx operation names in this checkout are:

```text
calyx.assign, calyx.comb_component, calyx.comb_group, calyx.component, calyx.constant, calyx.control, calyx.cycle, calyx.enable, calyx.group, calyx.group_done, calyx.group_go, calyx.ieee754.add, calyx.ieee754.compare, calyx.ieee754.divSqrt, calyx.ieee754.fpToInt, calyx.ieee754.intToFp, calyx.ieee754.mul, calyx.if, calyx.instance, calyx.invoke, calyx.memory, calyx.par, calyx.primitive, calyx.register, calyx.repeat, calyx.seq, calyx.seq_mem, calyx.static_group, calyx.static_if, calyx.static_par, calyx.static_repeat, calyx.static_seq, calyx.std_add, calyx.std_and, calyx.std_divs_pipe, calyx.std_divu_pipe, calyx.std_eq, calyx.std_ge, calyx.std_gt, calyx.std_le, calyx.std_lsh, calyx.std_lt, calyx.std_mult_pipe, calyx.std_mux, calyx.std_neq, calyx.std_not, calyx.std_or, calyx.std_pad, calyx.std_rems_pipe, calyx.std_remu_pipe, calyx.std_rsh, calyx.std_seq, calyx.std_sge, calyx.std_sgt, calyx.std_shru, calyx.std_signext, calyx.std_sle, calyx.std_slice, calyx.std_slt, calyx.std_sneq, calyx.std_srsh, calyx.std_sub, calyx.std_wire, calyx.std_xor, calyx.undef, calyx.undefined, calyx.while, calyx.wires
```

## Transformations And Conversions

The Calyx passes can be read as four related pipelines: prepare structured loops for Calyx, lower source IR into Calyx, compile Calyx control, and lower Calyx into FSM or hardware.

### Lowering Into Calyx

`lower-scf-to-calyx` lowers SCF and standard MLIR operations into Calyx. It can choose a top-level function as the entry component and has options for source-location metadata and JSON memory content output.

`lower-loopschedule-to-calyx` lowers the LoopSchedule dialect into Calyx. This is useful when loop scheduling decisions are already represented before entering Calyx.

`calyx-affine-to-scf` lowers Affine to SCF while attaching information used by later SCF-to-Calyx or LoopSchedule-to-Calyx lowering.

`affine-parallel-unroll` fully unrolls `affine.parallel` operations in a Calyx-oriented way, wrapping unrolled bodies in `scf.execute_region` operations so they align better with `calyx.par`.

`affine-ploop-unparallelize` rewrites selected `affine.parallel` operations into `affine.for` forms by an attribute-provided factor.

`exclude-exec-region-canonicalize` canonicalizes legal operations while deliberately avoiding `scf.execute_region`, because the Calyx-oriented unrolling path gives those regions special meaning.

### Calyx Internal Transformations

`calyx-gicm` performs group-invariant code motion. It lifts operations that are not group assignments, `calyx.group_done`, or `calyx.group_go` out to wire scope. After this pass, groups mainly describe wiring between already-existing structures.

`calyx-go-insertion` inserts `calyx.group_go` signals into the guards of a group's non-hole assignments. This prepares groups for schedule compilation.

`calyx-compile-control` compiles a control program bottom-up. It creates groups that implement control operators, drives group go and done ports, and replaces control statements with the generated compilation groups.

`calyx-remove-groups` inlines group assignments and removes the group interface. It wires the component `done` output from the top-level control done behavior and adds the component `go` signal to assignments.

`calyx-remove-comb-groups` transforms combinational groups into proper groups by registering values read from ports and rewriting `invoke`, `if`, and `while` forms that use a combinational condition group.

`calyx-clk-insertion` passes the component clock to subcomponents that need a `clk` port. `calyx-reset-insertion` similarly connects component reset to subcomponents that need reset.

`calyx-native` calls out to the native Rust-based Calyx compiler, runs a specified native pass pipeline, and returns a valid Calyx dialect program.

### Lowering Through FSM

`lower-calyx-to-fsm` lowers the Calyx control schedule to an `fsm.machine` nested inside the `calyx.control` region. The intermediate FSM still refers to Calyx groups and can be optimized before final materialization.

`materialize-calyx-to-fsm` gives that FSM explicit I/O: group done signals as inputs, group go signals as outputs, plus the top-level go and done ports.

`calyx-remove-groups-fsm` outlines the FSM to module scope, instantiates it in the Calyx component, replaces group go/done behavior with wires, and removes the remaining group structure.

### Lowering Out Of Calyx

`lower-calyx-to-hw` lowers Calyx to the lower CIRCT hardware dialects. Components become HW modules, wires and assignments become hardware wiring, library cells lower to `comb`, `seq`, `sv`, or `hw` constructs where supported, and already-compiled control can become ordinary hardware logic.

### Complete Pass List

The exact Calyx-related pass names covered in this chapter are:

```text
affine-parallel-unroll, affine-ploop-unparallelize, calyx-affine-to-scf, calyx-clk-insertion, calyx-compile-control, calyx-gicm, calyx-go-insertion, calyx-native, calyx-remove-comb-groups, calyx-remove-groups, calyx-remove-groups-fsm, calyx-reset-insertion, exclude-exec-region-canonicalize, lower-calyx-to-fsm, lower-calyx-to-hw, lower-loopschedule-to-calyx, lower-scf-to-calyx, materialize-calyx-to-fsm
```

## What It Implies

Seeing `calyx.component` means the IR still has a Calyx component boundary, including port metadata and the structural/control split.

Seeing `calyx.wires`, `calyx.group`, `calyx.comb_group`, or `calyx.static_group` means assignments are still grouped into scheduled actions. A group is not just a block of code. It is a named action that can be enabled by control.

Seeing `calyx.control`, `calyx.seq`, `calyx.par`, `calyx.if`, `calyx.while`, `calyx.repeat`, or `calyx.invoke` means the schedule is still explicit. The compiler has not fully lowered control into go/done wiring or FSM state.

Seeing `calyx.group_go` means `calyx-go-insertion` or a similar preparation step has made group activation explicit. Seeing no groups after `calyx-remove-groups` means the design is closer to plain structural hardware.

Seeing `calyx.register`, `calyx.memory`, `calyx.seq_mem`, or a `calyx.std_*` operation means hardware resources are still represented as Calyx cells with named ports. After `lower-calyx-to-hw`, those cells should become lower CIRCT hardware operations or module instances.

Seeing `lower-calyx-to-fsm` in a pipeline means the schedule is being represented as FSM control before hardware lowering. Seeing `lower-calyx-to-hw` means the compiler is leaving the Calyx abstraction.

## How To Read Calyx IR

Start with the component signature. Find the ports marked `go`, `done`, `clk`, and `reset`, then identify ordinary data inputs and outputs.

Next, inspect `calyx.wires`. Cells such as registers, memories, arithmetic operators, and component instances define hardware resources. Assignments connect their ports. If assignments are inside a group, they are only active when that group runs.

Then read the control region. A simple schedule might enable one group. A more complex schedule might sequence groups, run some in parallel, branch on a condition group, repeat a body, or invoke another component.

Finally, check which lowering stage you are in. If control operations are still present, Calyx is still scheduled. If groups have been compiled or removed, the design is moving toward explicit wiring, FSMs, or `hw` modules.

## Minimal Example

This simplified example shows the shape of a Calyx component with one group and a schedule that enables it:

```mlir
calyx.component @AddOne(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset})
    -> (%out: i32, %done: i1 {done}) {
  %one.out = calyx.constant @one <1 : i32> : i32
  %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

  calyx.wires {
    calyx.group @compute {
      calyx.assign %add.left = %in : i32
      calyx.assign %add.right = %one.out : i32
      calyx.assign %out = %add.out : i32
      calyx.group_done %go : i1
    }
  }

  calyx.control {
    calyx.enable @compute
  }
}
```

The important reading is the split. The adder and constant are structural cells. The group wires their ports for one action. The control region says that the component runs by enabling that action. Later passes can insert group go signals, compile the schedule, remove groups, or lower the whole component to ordinary hardware dialects.

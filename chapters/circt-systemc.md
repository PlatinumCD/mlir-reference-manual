# CIRCT `systemc` Dialect

## Beginner Summary

The CIRCT `systemc` dialect represents SystemC constructs in MLIR. SystemC is a
C++ library and modeling style used to describe hardware modules, ports,
signals, processes, and simulation behavior. The CIRCT dialect models the parts
of SystemC that are useful for emission: `SC_MODULE`, `SC_CTOR`, `SC_METHOD`,
`SC_THREAD`, ports, signals, module instances, sensitivity lists, and enough C++
expression and statement structure to print useful generated code.

For a beginner, the key idea is that `systemc` is close to an output dialect.
It is not trying to be a general hardware analysis IR like `hw`, `comb`, or
`seq`. It is an IR shape that has already decided the target representation is
SystemC or SystemC-adjacent C++. The dialect gives the compiler a structured way
to build that output before final text emission.

## Why This Dialect Exists

SystemC is C++, but directly generating C++ text too early is inconvenient.
Compiler passes still need to reason about modules, ports, signals, process
registration, function bodies, variables, calls, and interop wrappers. If the
compiler immediately printed strings, those later transformations would become
fragile text rewriting.

The `systemc` dialect keeps the output structured. A pass can create a
`systemc.module`, add a `systemc.ctor`, register a `systemc.method`, create
`systemc.signal` values, bind instance ports, and lower interop operations while
all of those concepts remain typed MLIR operations. Only after the structure is
ready does the exporter print C++/SystemC code.

## When This Dialect Matters

The dialect matters when CIRCT needs to emit SystemC rather than Verilog,
SystemVerilog, LLVM, or another backend. It is useful for flows that model
hardware at a C++ simulation level, integrate generated hardware models into
C++ software, or interoperate with Verilated modules.

It also matters as a bridge from hardware IR to generated C++. The
`convert-hw-to-systemc` pass translates `hw.module` operations into
`systemc.module` operations, creates SystemC ports, turns module internals into
SystemC member functions, and emits the `systemc.h` include through EmitC.

## When To Use It

Use `systemc` when the desired output is SystemC source code. If you are still
optimizing a hardware design, use dialects such as `hw`, `comb`, `seq`, `sv`,
or higher-level domain dialects. If the lowering pipeline has reached the point
where SystemC names, module fields, constructors, and C++ helper code are
important, `systemc` is the right level.

Use it directly when writing tests for the SystemC exporter, when building a
custom lowering to SystemC, or when modeling interop with C++ objects. In normal
flows, users are more likely to see it as the result of `convert-hw-to-systemc`
or as the input to `circt-translate --export-systemc`.

## Core Concepts

### Modules

`systemc.module` models a SystemC `SC_MODULE`. Its arguments are ports, such as
`!systemc.in<T>`, `!systemc.out<T>`, and `!systemc.inout<T>`. Internally, those
ports are emitted as fields of the generated C++ struct.

The body of a `systemc.module` is a graph region. It can contain signals,
submodule instance declarations, a constructor, member functions, and C++ helper
state. The operation is also a symbol, so other operations can refer to it by
name.

### Ports, Signals, And Channels

The type family is central to the dialect:

| Type | SystemC meaning |
| --- | --- |
| `!systemc.in<T>` | `sc_in<T>` input port. |
| `!systemc.out<T>` | `sc_out<T>` output port. |
| `!systemc.inout<T>` | `sc_inout<T>` bidirectional port. |
| `!systemc.signal<T>` | `sc_signal<T>` channel. |
| `!systemc.module<...>` | Handle for a module instance. |

`systemc.signal` declares an internal `sc_signal<T>`. `systemc.signal.read`
reads from an input, inout port, or signal. `systemc.signal.write` writes to an
output, inout port, or signal. These operations preserve the direction rules
that are natural in SystemC: for example, reading from an output port or writing
to an input port is invalid.

### Processes And Constructors

`systemc.ctor` models the `SC_CTOR` block of a module. It is where the compiler
registers processes and binds submodule ports.

`systemc.method` models `SC_METHOD`. `systemc.thread` models `SC_THREAD`.
Both refer to a `systemc.func`, which is a module-internal C++ member function
with no arguments and no return values. `systemc.sensitive` records the static
sensitivity list for the most recently created process in the constructor.

This is the SystemC version of saying: "run this member function as a process,
and wake it when these channels change."

### Instances

`systemc.instance.decl` declares a submodule instance by value inside a parent
module. Its result type is a `!systemc.module<...>` handle that records the
referenced module name and port list.

`systemc.instance.bind_port` appears in the constructor. It binds a named port
of the instance to a channel in the surrounding module. The channel can be a
module port or a `systemc.signal`, as long as direction and base type rules
match.

The `convert-hw-to-systemc` pass uses this structure when lowering `hw.instance`
operations. It creates an instance declaration, creates intermediate signals
when needed, and emits bind operations in the constructor.

### SystemC Data Types

The dialect supports common SystemC integer, bit-vector, and logic types:

| Type | Meaning |
| --- | --- |
| `!systemc.logic` | `sc_logic`, a four-state single-bit value. |
| `!systemc.int_base` | Dynamic-width `sc_int_base`. |
| `!systemc.int<W>` | Fixed-width `sc_int<W>`. |
| `!systemc.uint_base` | Dynamic-width `sc_uint_base`. |
| `!systemc.uint<W>` | Fixed-width `sc_uint<W>`. |
| `!systemc.signed` | Dynamic-width `sc_signed`. |
| `!systemc.bigint<W>` | Fixed-width `sc_bigint<W>`. |
| `!systemc.unsigned` | Dynamic-width `sc_unsigned`. |
| `!systemc.biguint<W>` | Fixed-width `sc_biguint<W>`. |
| `!systemc.bv_base` | Dynamic-width `sc_bv_base`. |
| `!systemc.bv<W>` | Fixed-width `sc_bv<W>`. |
| `!systemc.lv_base` | Dynamic-width `sc_lv_base`. |
| `!systemc.lv<W>` | Fixed-width `sc_lv<W>`. |

`systemc.convert` converts between supported SystemC integer/vector/logic types
and MLIR integer types where appropriate. This matters because hardware IR often
uses MLIR integers, while SystemC output often wants `sc_uint<W>`,
`sc_biguint<W>`, `sc_bv<W>`, or similar C++ library types.

### C++ Helper Operations

The dialect also includes operations under the `systemc.cpp.*` namespace. These
model C++ constructs that are useful for generated SystemC:

`systemc.cpp.variable` declares a C++ variable. `systemc.cpp.assign` assigns one
value to another. `systemc.cpp.new` and `systemc.cpp.delete` model heap
allocation and deallocation. `systemc.cpp.member_access` models `.` and `->`.
`systemc.cpp.func`, `systemc.cpp.call`, `systemc.cpp.call_indirect`, and
`systemc.cpp.return` model free or helper functions.

These are not general-purpose C++ semantics for every possible program. They
are the subset CIRCT needs to represent generated SystemC and interop code in a
structured way.

### Verilated Interop

`systemc.interop.verilated` instantiates a Verilated module represented by a
CIRCT `hw.module`, usually an external module. The
`systemc-lower-instance-interop` pass lowers this operation into the `interop`
dialect plus SystemC/C++ helper operations. The lowering allocates a pointer to
the Verilated C++ object, constructs it with `systemc.cpp.new`, writes inputs
through member accesses, calls `eval`, reads output members, and eventually
deletes the object.

This is useful when a SystemC-facing flow needs to connect to a model produced
by Verilator.

## Operation Inventory

| Operation | Meaning |
| --- | --- |
| `systemc.module` | Defines a SystemC `SC_MODULE`. |
| `systemc.ctor` | Defines the module constructor, emitted as `SC_CTOR`. |
| `systemc.func` | Defines a module-internal void member function. |
| `systemc.interop.verilated` | Represents an instance-side wrapper around a Verilated module. |
| `systemc.cpp.destructor` | Defines a C++ destructor body for a module. |
| `systemc.cpp.func` | Defines a C++ helper function, including declarations. |
| `systemc.cpp.return` | Returns from a `systemc.cpp.func`. |
| `systemc.signal.write` | Writes a value to an output, inout, or signal channel. |
| `systemc.signal` | Declares an internal `sc_signal<T>`. |
| `systemc.method` | Registers a `systemc.func` as an `SC_METHOD`. |
| `systemc.thread` | Registers a `systemc.func` as an `SC_THREAD`. |
| `systemc.sensitive` | Records static sensitivities for the previous process registration. |
| `systemc.instance.decl` | Declares a submodule instance. |
| `systemc.instance.bind_port` | Binds a submodule port to a channel. |
| `systemc.cpp.delete` | Models a C++ `delete` expression. |
| `systemc.cpp.assign` | Models a C++ assignment statement. |
| `systemc.cpp.variable` | Declares a C++ variable, optionally initialized. |
| `systemc.cpp.call_indirect` | Calls a value with function type. |
| `systemc.cpp.call` | Calls a named `systemc.cpp.func`. |
| `systemc.signal.read` | Reads the current value of an input, inout, or signal channel. |
| `systemc.convert` | Converts between supported SystemC and integer/vector/logic types. |
| `systemc.cpp.new` | Models a C++ `new` expression. |
| `systemc.cpp.member_access` | Models C++ `.` or `->` member access. |

## Transformation And Conversion Inventory

| Name | Kind | Role |
| --- | --- | --- |
| `convert-hw-to-systemc` | Conversion pass | Converts `hw.module` IR into `systemc.module` IR and adds `emitc.include <"systemc.h">`. |
| `systemc-lower-instance-interop` | Dialect pass | Lowers `systemc.interop.verilated` into `interop`, EmitC, and SystemC/C++ helper operations. |
| `canonicalize` | Generic MLIR cleanup | Uses local folding/canonicalization hooks such as `systemc.convert` folding and `systemc.sensitive` cleanup. |
| `export-systemc` | Translation target | Emits one SystemC/C++ output stream from structured SystemC IR. |
| `export-split-systemc` | Translation target | Emits split SystemC/C++ output into a directory. |

The first two are compiler passes. The last two are translation targets
registered for `circt-translate`; they print final source code rather than
rewrite IR.

## Conversion From HW

`convert-hw-to-systemc` is the main path into this dialect. It translates each
`hw.module` into a `systemc.module`, converts HW ports into SystemC port types,
and preserves module visibility. It creates a `systemc.func` named like an
internal logic function and moves the hardware module body into that function.
It then registers the function in `systemc.ctor` with `systemc.method`.

For combinational input logic, the pass reads input ports with
`systemc.signal.read`, converts SystemC values back to the types expected by the
logic, runs the original logic, converts results to SystemC-friendly types, and
writes output ports with `systemc.signal.write`.

For instances, the pass lowers `hw.instance` to `systemc.instance.decl` plus
`systemc.instance.bind_port` operations. Intermediate `systemc.signal` channels
are introduced where needed so the parent module can connect instance inputs
and outputs in a SystemC-compatible way.

The current implementation has practical limits. The source code rejects
parameterized modules and HW inout arguments in this conversion path. That does
not mean the dialect itself can never represent inout ports; it means this
particular HW-to-SystemC pass only supports a subset of HW input/output shapes.

## Export To Text

Once IR is in the `systemc` dialect, `circt-translate --export-systemc` can
print source code. The exporter maps:

| IR construct | Emitted construct |
| --- | --- |
| `systemc.module` | `SC_MODULE(name) { ... }` |
| `!systemc.in<T>` | `sc_in<T>` |
| `!systemc.out<T>` | `sc_out<T>` |
| `!systemc.inout<T>` | `sc_inout<T>` |
| `!systemc.signal<T>` | `sc_signal<T>` |
| `systemc.ctor` | `SC_CTOR(name) { ... }` |
| `systemc.method` | `SC_METHOD(func)` |
| `systemc.thread` | `SC_THREAD(func)` |
| `systemc.sensitive` | `sensitive << ...` |
| `systemc.signal.read` | `.read()` or equivalent value access. |
| `systemc.signal.write` | `.write(...)`. |

This is why the dialect carries operations that look like C++ constructs: the
final output is C++ with SystemC library calls and macros.

## How To Read SystemC IR

A minimal module with ports looks like this:

```mlir
systemc.module @Adder(%a: !systemc.in<!systemc.uint<32>>,
                      %b: !systemc.in<!systemc.uint<32>>,
                      %sum: !systemc.out<!systemc.uint<32>>) {
}
```

Read this as a SystemC module named `Adder` with two input ports and one output
port.

A simple combinational process looks like this:

```mlir
systemc.module @Pass(%in: !systemc.in<!systemc.uint<32>>,
                     %out: !systemc.out<!systemc.uint<32>>) {
  systemc.ctor {
    systemc.method %logic
    systemc.sensitive %in : !systemc.in<!systemc.uint<32>>
  }

  %logic = systemc.func {
    %v = systemc.signal.read %in : !systemc.in<!systemc.uint<32>>
    systemc.signal.write %out, %v : !systemc.out<!systemc.uint<32>>
  }
}
```

Read this as a module with one method process. The method runs when `%in` is
sensitive, reads the input port, and writes the output port.

## What It Implies

Seeing `systemc` in IR means the lowering has chosen a C++/SystemC output model.
Names, constructor structure, process registration, port binding, and C++ helper
operations now matter. Transformations should preserve emitted-code intent, not
just hardware semantics.

The dialect also implies a different kind of abstraction boundary from Verilog
or RTL dialects. SystemC has library types and simulation semantics. A
`systemc.signal.write` is not merely a wire assignment; it represents SystemC
signal update behavior. A `systemc.sensitive` operation is not a dataflow edge;
it is process scheduling metadata for the generated SystemC module.

For beginners, the practical rule is simple: use hardware dialects to describe
the circuit, then use the SystemC dialect when the compiler is preparing to
emit C++ SystemC code.

## Common Pitfalls

Do not confuse `systemc.func` and `systemc.cpp.func`. `systemc.func` is a
module-internal member process body used with `systemc.method` or
`systemc.thread`. `systemc.cpp.func` is a C++ helper function with a normal
function signature.

Do not write directly to input ports or read directly from output ports. The
dialect verifier enforces port direction rules for `systemc.signal.read` and
`systemc.signal.write`.

Do not assume every C++ construct is modeled. The `systemc.cpp.*` operations are
a practical subset for generated code. More complex C++ may need EmitC or a
target-specific extension.

Do not treat export as just another optimization pass. `export-systemc` and
`export-split-systemc` are translation targets that print source code. Passes
should prepare and clean up IR before that final step.


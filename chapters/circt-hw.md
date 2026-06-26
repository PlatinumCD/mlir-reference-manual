# CIRCT `hw` Dialect

The CIRCT `hw` dialect is the common structural hardware dialect. It represents modules, ports, instances, hierarchy, constants, wires, aggregate hardware values, parameters, and hardware-specific types. It is the backbone that many other CIRCT dialects use when they need to describe a hardware design without committing to a single frontend language.

For a beginner, the simplest model is: `hw.module` is like a Verilog module, `hw.instance` is like instantiating another module, `hw.output` drives output ports, and the rest of the dialect provides hardware values and metadata needed to make that hierarchy precise.

This dialect is not primarily about behavioral arithmetic. Arithmetic usually appears in dialects such as `comb`, `seq`, or other domain-specific CIRCT dialects. The `hw` dialect provides the structural shell in which those operations live.

## When HW Is Important

Use `hw` when the IR is describing hardware hierarchy. If you see `hw.module`, `hw.module.extern`, `hw.instance`, or `hw.output`, you are looking at the module-level structure of a design.

The dialect is important for lowering to SystemVerilog, for sharing infrastructure across hardware frontends, and for connecting higher-level dialects to concrete hardware modules. FIRRTL, Calyx, ESI, MSFT, and other CIRCT dialects can eventually interact with or lower through HW-style modules and instances.

You should also pay attention to HW when debugging name preservation, parameterization, aggregate ports, and non-local annotations. Operations like `hw.wire`, `hw.hierpath`, `hw.type_scope`, and `hw.typedecl` exist because hardware designs need stable names, hierarchical references, and reusable type declarations.

## Why It Is Needed

MLIR's standard `func.func` is not enough for hardware. A function call consumes values and produces values in a mostly software-like way. A hardware module has named ports, instance names, output wires, parameters, inout ports, hierarchy, symbols, and emit-time concerns such as Verilog names.

The HW dialect gives CIRCT a target-independent representation of those concepts. It can model a module before it is emitted as Verilog. It can hold an external module declaration. It can represent generated modules that an outside generator will create. It can preserve a hierarchical path to a component inside an instance tree.

This is why HW often appears near the end of hardware lowering pipelines. Earlier dialects may describe intent or behavior; HW describes the structural hardware container that will be emitted or optimized further.

## Type Inventory

The HW dialect defines hardware-specific types and type constraints.

- `!hw.int<...>` is a parameterized-width signless integer. If the width is a known constant, CIRCT may use a builtin integer type such as `i8`; `!hw.int` matters when the width is a parameter expression.
- `!hw.array<N x T>` is a packed fixed-size hardware array. It is useful for vectors of wires and packed aggregate values.
- `!hw.uarray<N x T>` is an unpacked fixed-size array, commonly used for memory-like structures.
- `!hw.inout<T>` represents connection-style values, especially Verilog `inout` ports and wires.
- `!hw.struct<field: type, ...>` is a packed named-field aggregate.
- `!hw.enum<A, B, ...>` is an enumeration whose encoding is synthesis-defined.
- `!hw.union<field: type, ...>` is an untagged union of fields.
- `!hw.typealias<...>` references a symbolic type declaration in a `hw.type_scope`.
- `!hw.modty<...>` is a module type containing port information.
- `!hw.string` is a string type for HW-centric dialects.

The important beginner distinction is that these are hardware types, not software data structures. A packed array or struct describes bits on wires. An inout type describes a connection. A module type describes a port interface.

## Attribute Inventory

The local HW attribute definitions include:

- `#hw.output_file` records an output filename or output directory for emitted artifacts.
- `#hw.param.decl` describes a module or instance parameter, including name, type, and optional value.
- `#hw.param.decl.ref` references a named parameter value from inside a module body.
- `#hw.param.verbatim` carries text to emit directly to SystemVerilog for a parameter.
- `#hw.param.expr` represents a parameter expression built from typed operands.
- `#hw.enum.field` identifies one field of an enum type.
- `#hw.innerSymProps` stores one inner symbol name, field id, and visibility.
- `#hw.innerSym` stores one or more inner symbols on an operation, port, or aggregate field.

These attributes explain why HW is more than a simple module graph. Hardware compilers must preserve names and parameters across many transformations.

## Operation Inventory

The local HW dialect defines twenty-eight operations.

### Module and Hierarchy Operations

`hw.module` represents a hardware module with a symbol name, ports, optional parameters, and a body. The body is a graph region and must end with `hw.output`. This is the main operation you should look for when trying to understand a hardware design.

`hw.module.extern` declares an external module. It has a module name and port list, but no implementation body. It is used for black boxes, vendor IP, or modules provided outside the current IR.

`hw.generator.schema` declares the metadata schema for a generated module kind. It lists required attributes for generated modules.

`hw.module.generated` declares a module that will be produced by an external generator. It references a generator schema and carries the ports and parameters needed to describe the generated module.

`hw.instance` instantiates a module. Its operands correspond to input ports, and its results correspond to output ports. The operation stores an instance name and a symbol reference to the target module. Parameters may be passed on the instance.

`hw.output` terminates an `hw.module` body. Its operands drive the module's output ports in order.

`hw.hierpath` stores a symbolic path through module hierarchy. It is used for non-local annotations and for references to named things inside nested instances.

`hw.triggered` holds a procedural region with an event control such as `posedge`, `negedge`, or `edge`. It is isolated from above, so values used inside must be explicitly passed through its input list.

### Type Declaration Operations

`hw.type_scope` is a top-level symbol table for type declarations. It contains `hw.typedecl` operations.

`hw.typedecl` gives a symbolic name to a type. The resulting type alias can be used elsewhere, and an optional Verilog name can control emitted spelling.

### Constants, Parameters, and Names

`hw.constant` produces a signless integer constant. It is the basic integer constant operation in the HW dialect.

`hw.aggregate_constant` produces a constant aggregate value such as an array or struct. It supports nested aggregates and can include clock and reset values.

`hw.param.value` turns a parameter expression into an SSA value usable by other operations. This is how a parameter can participate in the dataflow of a module body.

`hw.enum.constant` produces an enum value.

`hw.enum.cmp` compares two enum values of the same canonical enum type and returns an `i1`.

`hw.wire` is an identity operation that gives a value a stable wire name or inner symbol. It is important because names in hardware are observable. Some transformations may propagate name hints, but a symbolic wire preserves a referenceable boundary.

`hw.bitcast` reinterprets one value as another value with the same hardware bit width. It can cross between integers and aggregates when the bit sizes match.

### Array Operations

`hw.array_create` creates an HW array from individual element values.

`hw.array_concat` concatenates multiple HW arrays into one larger array.

`hw.array_slice` extracts a contiguous range from an array starting at a low index.

`hw.array_get` extracts one array element. Its index width must be the exact number of bits needed to index the input array.

`hw.array_inject` returns a copy of an array with one element replaced. If the index is out of bounds, the result is undefined.

### Struct Operations

`hw.struct_create` creates a struct value from its field values.

`hw.struct_extract` extracts a named field from a struct.

`hw.struct_inject` returns a copy of a struct with one named field replaced.

`hw.struct_explode` expands a struct into separate SSA results, one per field.

### Union Operations

`hw.union_create` creates an untagged union with a selected field value. Bits not covered by that field are undefined.

`hw.union_extract` interprets a union value as one of its fields and returns that field type.

## Transformations and Conversions

The HW pass set in this checkout contains twelve named passes.

`hw-print-instance-graph` prints a DOT graph of module hierarchy through instances. `hw-print-module-graph` prints a graph of HW modules and can include verbose SSA edge information. These are analysis and debugging passes.

`hw-imdce` performs intermodule dead-code elimination. It starts from public module ports and symbol-bearing values, propagates liveness through dataflow, and erases dead operations, dead modules, and unused ports.

`hw-flatten-io` flattens struct-typed input and output ports. It can recursively flatten nested structs, optionally flatten extern modules, and choose the join character used in generated port names.

`hw-flatten-modules` eagerly inlines private modules into their instance sites. This is useful for verification and simulation flows where preserving hierarchy is not necessary or where flat structure simplifies later analysis.

`hw-specialize` specializes parameterized `hw.module` operations for specific `hw.instance` parameter values. It creates specialized modules and rewrites instances to target them. This is essential when parameterized widths or array sizes must become concrete.

`hw-verify-irn` verifies inner-reference namespace operations when they are not already self-verifying.

`hw-aggregate-to-comb` lowers HW aggregate operations to `comb` operations inside modules. For example, aggregate array and struct manipulation can become bit extracts, concatenations, or other combinational operations. Ports are not flattened by this pass; that is handled by `hw-flatten-io`.

`hw-vectorization` performs structural vectorization of bit-level operations in HW modules, merging scalar bit-level assignments into vectorized operations where possible.

`hw-parameterize-constant-ports` turns private-module input ports into parameters when every instance passes constants or parameter values. This exposes constants at module boundaries and can improve synthesis and timing analysis.

`hw-bypass-inner-symbols` moves inner symbols from ports to wires, then bypasses wire operations while preserving symbols. This lets optimization cross symbol boundaries in synthesis-safe cases.

`hw-convert-bitcasts` converts `hw.bitcast` operations on aggregates into explicit aggregate and bit-access operations. It currently supports `!hw.array` and `!hw.struct`, and it depends on the `comb` dialect.

## What It Implies

When you see `hw.module`, think hardware hierarchy, not software function. Its ports are named hardware interface points. Its body computes values for output ports and instantiates other modules.

When you see `hw.instance`, follow its module symbol. The operation is the edge from one module to another in the instance graph. Its result names usually correspond to output port names.

When you see `hw.wire` or `#hw.innerSym`, names are part of the semantic surface. Optimizations must be careful because a user or another tool may refer to that named thing.

When you see parameter attributes or `!hw.int<#hw.param.decl.ref<...>>`, the design is still parametric. A pass such as `hw-specialize` may later clone modules and resolve those parameters to concrete types.

## How To Read HW IR

Start at the top-level `mlir::ModuleOp`, then find public `hw.module` operations. These are the likely design entry points.

Inside each module, read the port list first. Then inspect `hw.instance` operations to see the submodule hierarchy. Use `hw.output` to understand which values drive output ports.

Next, look for structural value operations. Array, struct, and union operations describe aggregate wiring. `hw.bitcast` tells you bits are being reinterpreted without changing width. `hw.constant`, `hw.aggregate_constant`, and `hw.param.value` tell you which values are compile-time constants or parameter-derived.

Finally, inspect names and references. `hw.wire`, `hw.hierpath`, `#hw.innerSym`, and `hw.typedecl` often explain why a value or type must survive transformations in a particular form.

## Minimal Example

This small example shows the basic hierarchy pattern:

```mlir
hw.module @And2(in %a: i1, in %b: i1, out y: i1) {
  %y = comb.and %a, %b : i1
  hw.output %y : i1
}

hw.module @Top(in %x: i1, in %z: i1, out out: i1) {
  %inst.y = hw.instance "u_and" @And2(a: %x: i1, b: %z: i1) -> (y: i1)
  %named = hw.wire %inst.y name "and_result" : i1
  hw.output %named : i1
}
```

The `comb.and` operation performs the combinational logic, but `hw` provides the hardware container: modules, instance, named wire, and output connection.

# CIRCT `msft` Dialect

The CIRCT `msft` dialect is a Microsoft support dialect inside CIRCT. It is an umbrella for hardware constructs and physical-design annotations that are useful to Microsoft flows but have not necessarily been generalized into a public, stable CIRCT dialect.

For a beginner, the most useful mental model is this: `msft` is a staging and annotation dialect around hardware IR. It does not replace `hw`, `comb`, `seq`, or `sv`. Instead, it adds a few high-level hardware constructs, dynamic instance references, FPGA placement metadata, and Tcl export support. Most `msft` operations are meant to disappear after lowering, either into normal `hw` operations or into generated Tcl/verbatim emission.

The local checkout defines 12 `msft` operations, three custom attributes, one primitive-type enum attribute, two dynamic-instance interfaces, and three MSFT-owned passes.

## When MSFT Is Important

The `msft` dialect is important when a CIRCT hardware design needs extra placement or hierarchy information that is not naturally part of ordinary RTL semantics.

You will see it in flows that need to answer questions like:

- Which dynamic instance path should a placement annotation apply to?
- Where should an FPGA primitive, register bit, or instance be placed?
- Which instances should be constrained to a named physical region?
- Which register-to-register path should receive a multicycle timing constraint?
- Can a high-level systolic-array construct be expanded into ordinary hardware instances?
- Should CIRCT emit Quartus-style Tcl alongside Verilog?

This dialect is not the first place to learn general hardware modeling in CIRCT. For that, start with `hw`, `comb`, `seq`, and `sv`. Use `msft` when the IR contains Microsoft-specific support constructs or physical design annotations.

## Why It Is Needed

Hardware IR often needs information that is not part of the logical circuit itself. Placement constraints, region constraints, instance-assignment attributes, and Tcl directives affect downstream FPGA tools, but they are not wires, registers, or modules.

The `msft` dialect gives those facts a structured representation in MLIR. Instead of storing everything as comments or backend-specific strings, CIRCT can represent physical design intent as operations and attributes. Passes can then check symbols, build `hw.hierpath` references, group annotations by top module, and emit tool-specific Tcl.

The dialect also contains a small number of high-level hardware constructs. `msft.systolic.array` preserves the idea of a row/column broadcast processing-element array until the lowering pass expands it into ordinary `hw` array operations and cloned processing-element logic.

The implication is that `msft` is useful at the boundary between hardware structure and tool integration. It is not generally intended to survive to final emitted RTL.

## Types, Attributes, And Interfaces

The `msft` dialect does not define custom types in this checkout. Its operations use existing CIRCT and MLIR types, especially `hw` arrays, integer types, and `!seq.clock`.

It defines physical-design attributes:

- `#msft.physloc<...>` describes one physical primitive location. It stores a primitive type plus X, Y, and number coordinates.
- `#msft.physical_bounds<...>` describes an inclusive rectangular physical region using X and Y ranges.
- `#msft.location_vec<...>` stores one optional location per bit of a hardware type. It is used by register placement because each bit of a register may have its own physical location.

The primitive-type enum is `PrimitiveType`, with current values `M20K`, `DSP`, and `FF`. These correspond to FPGA memory blocks, DSP blocks, and flip-flops in the supported physical-design model.

The dialect also defines two operation interfaces.

`DynInstDataOpInterface` marks an operation as carrying data associated with a dynamic instance path. It provides a method for discovering the top module to which the data applies.

`UnaryDynInstDataOpInterface` is for annotations that refer to one `hw.hierpath`. It verifies that an operation either lives inside a `msft.instance.dynamic` before lowering or has a global `hw.hierpath` reference after lowering, but not both.

## Operation Inventory

The local `msft` dialect defines 12 operations.

### High-Level Hardware Constructs

`msft.systolic.array` models a row/column broadcast systolic array. It takes an array of row inputs and an array of column inputs. Its region describes one processing element, with one row value and one column value as arguments. The result is a matrix, represented as an `hw` array of `hw` arrays.

This operation is not a graph region. The processing-element body is feed-forward, and lowering clones the body for each row/column pair.

`msft.pe.output` terminates the processing-element region of `msft.systolic.array`. It yields the output value for one processing element.

`msft.hlc.linear` models a linear, feed-forward datapath that can be pipelined. It has a clock operand, produces one or more results, and contains a single-block datapath region. The verifier only allows `hw`, `comb`, and `msft` dialect operations inside the datapath.

`msft.output` terminates `msft.hlc.linear` regions and returns values from MSFT region constructs.

### Dynamic Instance Hierarchy

`msft.instance.hierarchy` is the root of a dynamic instance hierarchy. It names the top hardware module and can optionally name a particular instance of that top module. Dynamic instance operations live inside its region.

`msft.instance.dynamic` represents one instance in a hierarchy path. It uses an `hw` inner reference such as `@module::@inst`. Dynamic instances can nest, building a path through the design. During lowering, this path becomes an `hw.hierpath`.

`msft.instance.verb_attr` attaches an arbitrary name/value assignment to a dynamic instance. In the Quartus Tcl exporter, it becomes a `set_instance_assignment` command.

These operations are useful because placement annotations often need to refer to an instance path without requiring every static instance operation to carry a back-pointer attribute from the beginning.

### Physical-Design Operations

`msft.physical_region` declares a named physical region at module scope. It stores one or more `#msft.physical_bounds` rectangles.

`msft.pd.location` assigns one physical location to a dynamic instance or to an entity under that instance. It can include a `path` string for a subpath inside an external module or primitive.

`msft.pd.reg_location` assigns per-bit locations to a register. It uses `#msft.location_vec` syntax, where `*` means that no location is specified for that bit.

`msft.pd.physregion` assigns an instance to a named `msft.physical_region`. The Tcl exporter turns this into placement-region, reservation, core-only, and region-name assignments.

`msft.pd.multicycle` records a multicycle timing constraint from one register path to another. The source and destination are `hw.hierpath` symbols, and the cycle count must be at least one. The operation checks that the source and destination paths belong to the same top module.

## Transformations And Conversions

The dialect has three MSFT-owned passes.

`msft-lower-constructs` lowers high-level MSFT constructs. In this checkout, it marks `msft.systolic.array` illegal and rewrites it. The pass extracts each row input with `hw.array_get`, extracts each column input with `hw.array_get`, clones the processing-element region for every row/column pair, collects each `msft.pe.output` value, and builds the resulting matrix with `hw.array_create`. It also preserves useful name hints by suffixing cloned operation names with row and column indices.

`msft-lower-instances` lowers dynamic instance hierarchy structure into `hw.hierpath` symbols. It walks `msft.instance.hierarchy` regions in post-order so nested `msft.instance.dynamic` ops are processed before their parents. If a dynamic instance contains operations using `DynInstDataOpInterface`, the pass creates a unique `hw.hierpath` such as `@instref`, moves the annotation operations to the hierarchy root, sets their path reference, and erases the dynamic instance.

`msft-export-tcl` emits Tcl from MSFT physical-design annotations. It groups dynamic-instance data by top module and optional instance name, writes Tcl into `emit.file` and `sv.verbatim` operations, then removes MSFT physical-design operations. It handles `msft.pd.location`, `msft.pd.reg_location`, `msft.pd.physregion`, `msft.pd.multicycle`, `msft.instance.verb_attr`, `msft.instance.hierarchy`, and `msft.physical_region`. The pass depends on `emit`, `hw`, and `sv`.

There is no separate general CIRCT conversion pass for the MSFT dialect in this checkout. The conversion story is internal to those MSFT passes: high-level constructs lower to `hw`, dynamic instances lower to `hw.hierpath`, and physical-design annotations lower to Tcl emission plus removal.

## What It Implies

Seeing `msft.systolic.array` means the design still contains a high-level matrix of processing elements. Look at the row input array, column input array, PE region, and `msft.pe.output` to understand the repeated computation. After `msft-lower-constructs`, this becomes ordinary `hw.array_get`, cloned PE logic, and `hw.array_create`.

Seeing `msft.hlc.linear` means a feed-forward datapath has been grouped as something schedulable or pipelineable. Its clock is explicit, and its body should only use combinational or MSFT-supported operations.

Seeing `msft.instance.hierarchy` or `msft.instance.dynamic` means the IR is still representing instance paths abstractly. After `msft-lower-instances`, those paths should become `hw.hierpath` symbols.

Seeing `msft.pd.location`, `msft.pd.reg_location`, `msft.pd.physregion`, or `msft.pd.multicycle` means the IR carries physical design intent. These are not logical RTL operations. They are constraints for downstream tools.

Seeing `msft.instance.verb_attr` means a backend-specific instance assignment is being carried through CIRCT as structured IR until Tcl export.

## How To Read MSFT IR

Start by separating logical constructs from annotations. `msft.systolic.array` and `msft.hlc.linear` are high-level hardware constructs. The `msft.pd.*` operations are physical-design constraints. The `msft.instance.*` operations connect constraints to instance paths.

Next, find the path reference. Before `msft-lower-instances`, a physical-design operation may be inside a `msft.instance.dynamic` body. After lowering, it should carry a reference to an `hw.hierpath`.

Then read the target of the annotation. A location assignment gives an FPGA primitive coordinate. A register-location assignment maps individual bits. A physical-region assignment references a named region. A multicycle assignment links source and destination register paths.

Finally, check whether Tcl export has happened. Before export, `msft` operations are still present. After `msft-export-tcl`, Tcl should appear in `emit.file` and `sv.verbatim`, and MSFT physical-design operations should be gone.

## Minimal Examples

This simplified placement annotation targets a memory block under a dynamic instance path:

```mlir
msft.instance.hierarchy @top {
  msft.instance.dynamic @top::@u_child {
    msft.pd.location M20K x: 15 y: 9 n: 3 path: "|memBank2"
  }
}
```

After `msft-lower-instances`, the dynamic path is represented by an `hw.hierpath`, and the placement operation refers to that path.

This simplified physical region declares a named region:

```mlir
msft.physical_region @region1, [
  #msft.physical_bounds<x: [0, 10], y: [0, 10]>,
  #msft.physical_bounds<x: [20, 30], y: [20, 30]>]
```

Read it as "declare a placement region named `region1` made of two rectangles." A later `msft.pd.physregion` can assign an instance to that region, and `msft-export-tcl` can turn the assignment into Quartus-style Tcl.


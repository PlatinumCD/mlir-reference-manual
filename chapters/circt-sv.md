# CIRCT `sv` Dialect

The `sv` dialect is CIRCT's SystemVerilog-oriented dialect. It represents
SystemVerilog-specific constructs in an AST-like form that can be mixed with
the `hw`, `comb`, `seq`, `ltl`, and `verif` dialects. Its main purpose is to
give CIRCT a predictable way to produce readable SystemVerilog text.

For a beginner, the key distinction is this: `hw` and `comb` represent hardware
structure and logic in a compiler-friendly way, while `sv` represents
SystemVerilog surface constructs that matter for emission. The SV dialect is
less about abstract hardware analysis and more about preserving constructs that
should appear in the output: wires, regs, always blocks, initial blocks,
preprocessor directives, macros, interfaces, assertions, generate blocks,
verbatim text, DPI imports, and other SystemVerilog features.

## Why This Dialect Exists

SystemVerilog is both a hardware description language and a text format with
many syntactic features. Some of those features are not natural to model in a
small structural hardware IR. Examples include `ifdef` blocks, macro
definitions, `always_ff`, procedural assignments, generate loops, interfaces,
modports, `$display`-style system calls, and inline verbatim text.

The `sv` dialect exists so CIRCT can represent those constructs explicitly
before export. This keeps the output path simple and predictable. A pass can
lower a register into an `sv.alwaysff`, or a verification op into an
`sv.assert_property`, and the exporter can print that operation as
SystemVerilog.

The dialect is intentionally close to SystemVerilog syntax. The rationale
document describes it as aligned with the textual nature of SystemVerilog and
not primarily focused on being easy to analyze and transform. That is a useful
warning: use `sv` when you need SystemVerilog-specific output control; use
more semantic dialects when you want compiler analysis and optimization.

## When It Is Important

The `sv` dialect becomes important late in CIRCT lowering, especially when IR is
close to Verilog emission. It also matters earlier when a frontend or transform
needs to preserve a SystemVerilog-specific construct that cannot be expressed
cleanly in `hw`, `comb`, or `seq`.

You will see `sv` when:

- hardware declarations need explicit SystemVerilog names or forms,
- procedural and continuous assignments must be preserved,
- generated code needs `always`, `always_comb`, `always_ff`, or `initial`
  regions,
- verification constructs need to print as assertions, assumptions, or covers,
- interfaces and modports are part of the design,
- macros, include directives, or `ifdef` blocks must survive,
- inline SystemVerilog text is needed as an escape hatch,
- the pipeline is preparing for `ExportVerilog`.

## Types And Attributes

Most ordinary hardware values come from other CIRCT dialects. The SV dialect
adds only a small amount of type machinery. The most visible piece is support
around inout-style values, using `!hw.inout<T>` with SV operations such as
`sv.wire`, `sv.reg`, `sv.logic`, and the inout read/write/select operations.
The dialect also defines `!sv.open_uarray<T>` for SystemVerilog unpacked open
arrays, mainly for DPI import arguments.

SV-specific attributes include `sv.macro.ident` for macro identifiers and
`sv.attribute` for Verilog attributes. The latter represents SystemVerilog
attribute syntax attached to declarations, statements, or expressions. These
attributes are output metadata for downstream tools; they are not a universal
optimization barrier, so users should not assume every attribute survives every
transformation unless the relevant pass preserves it.

## Complete Operation Inventory

The current `sv` dialect defines 83 operations. The most useful way to learn
them is by category.

Declarations and named references:
`sv.wire`, `sv.reg`, `sv.logic`, `sv.xmr`, `sv.xmr.ref`,
`sv.read_inout`, `sv.indexed_part_select`, `sv.indexed_part_select_inout`,
`sv.array_index_inout`, and `sv.struct_field_inout`.

Expressions and constants:
`sv.system`, `sv.verbatim.expr`, `sv.verbatim.expr.se`,
`sv.macro.ref.expr`, `sv.macro.ref.expr.se`, `sv.constantX`,
`sv.constantZ`, `sv.constantStr`, `sv.sformatf`, `sv.localparam`,
`sv.system.sampled`, `sv.system.time`, `sv.system.stime`,
`sv.unpacked_array_create`, and `sv.unpacked_open_array_cast`.

Generate blocks and interfaces:
`sv.generate`, `sv.generate.case`, `sv.generate.for`, `sv.interface`,
`sv.interface.signal`, `sv.interface.modport`, `sv.interface.instance`,
`sv.modport.get`, `sv.interface.signal.assign`,
`sv.interface.signal.read`, and `sv.bind.interface`.

Statements, regions, and procedural structure:
`sv.ordered`, `sv.ifdef`, `sv.ifdef.procedural`, `sv.if`, `sv.always`,
`sv.alwayscomb`, `sv.alwaysff`, `sv.initial`, `sv.case`, `sv.assign`,
`sv.bpassign`, `sv.passign`, `sv.force`, `sv.release`, `sv.alias`,
`sv.write`, `sv.fwrite`, `sv.fflush`, `sv.fclose`, `sv.verbatim`,
`sv.macro.ref`, `sv.bind`, `sv.exit`, `sv.nonstandard.deposit`,
`sv.readmem`, `sv.for`, `sv.macro.error`, `sv.macro.decl`,
`sv.macro.def`, `sv.include`, `sv.func.dpi.import`, `sv.func`,
`sv.func.call.procedural`, `sv.func.call`, `sv.return`, and
`sv.reserve_names`.

Verbatim source and module emission:
`sv.verbatim.source` and `sv.verbatim.module`.

Verification:
`sv.assert`, `sv.assume`, `sv.cover`, `sv.assert.concurrent`,
`sv.assume.concurrent`, `sv.cover.concurrent`, `sv.assert_property`,
`sv.assume_property`, and `sv.cover_property`.

## How To Think About The Operation Groups

The declaration operations are about naming and connecting values in a way that
matches SystemVerilog. `sv.wire` declares a net, while `sv.reg` and `sv.logic`
declare variable-style storage targets. The inout operations provide the bridge
between value-typed hardware IR and assignable SystemVerilog lvalues.

The expression operations are mostly escape hatches or SystemVerilog-specific
values. `sv.system` calls a system function. `sv.verbatim.expr` and
`sv.verbatim.expr.se` emit textual expressions with operand substitution, with
the `.se` form indicating possible side effects. `sv.macro.ref.expr` and
`sv.macro.ref.expr.se` do the same for macro references. `sv.constantX` and
`sv.constantZ` represent unknown and high-impedance constants that are native to
Verilog semantics but not ordinary two-state hardware integers.

The statement operations model the behavioral surface of SystemVerilog:
conditional blocks, procedural regions, assignments, force/release, file I/O,
macros, includes, and function bodies. `sv.assign` is a continuous assignment.
`sv.bpassign` is a blocking procedural assignment. `sv.passign` is a
nonblocking procedural assignment.

The generate and interface operations model elaboration-time SystemVerilog
features. `sv.generate`, `sv.generate.case`, and `sv.generate.for` express
generate constructs. The interface operations define interfaces, signals,
modports, instances, and signal reads or assignments.

The verification operations are SystemVerilog assertion forms. Immediate
assertions, assumptions, and covers use `sv.assert`, `sv.assume`, and
`sv.cover`. Concurrent forms use the `.concurrent` operations. Property forms
such as `sv.assert_property` work with `ltl` properties and optional clocking
or disable conditions.

## Transformations

The SV pass file defines nine named passes.

| Pass | What it does |
| --- | --- |
| `hw-cleanup` | Cleans HW/SV module bodies, including merging compatible `sv.alwaysff`, `sv.always`, and `sv.ifdef` regions. |
| `hw-legalize-modules` | Lowers away SV, Comb, and HW features unsupported by selected lowering options, often late before emission. |
| `prettify-verilog` | Improves the quality and readability of `ExportVerilog` output. |
| `hw-stub-external-modules` | Turns external HW modules into empty module bodies for linting and missing-file avoidance. |
| `hw-generator-callout` | Calls an external generator for generated HW modules. |
| `sv-trace-iverilog` | Adds tracing instrumentation for Icarus Verilog simulation. |
| `hw-export-module-hierarchy` | Emits module and instance hierarchy information as `sv.verbatim` output files. |
| `hw-eliminate-inout-ports` | Rewrites inout port usage into explicit input and output ports. |
| `sv-mask-non-synthesizable` | Deletes or wraps non-synthesizable assertion-style SV ops with an `ifdef` macro such as `SYNTHESIS`. |

These passes are a mixture of cleanup, compatibility, simulation support, and
emission preparation. Several have `hw-` names because SV is usually embedded
inside `hw.module` bodies rather than used as a standalone program dialect.

## Conversions And Emission

Several conversion passes produce or consume SV dialect operations.

| Pass | Direction |
| --- | --- |
| `lower-hw-to-sv` | Converts selected HW constructs into SV operations. |
| `lower-verif-to-sv` | Converts Verif operations into SV assertion operations. |
| `lower-seq-to-sv` | Lowers sequential operations into SV constructs such as always blocks and assignments. |
| `lower-sim-to-sv` | Lowers simulator-specific `sim` operations to SV. |
| `convert-fsm-to-sv` | Converts FSM IR into SV and HW. |

The final output path is `export-verilog` or `export-split-verilog`, which
prints the hardware design as SystemVerilog. `prepare-for-emission` and related
export preparation passes may also affect SV output. Conceptually, the flow is:
semantic hardware dialects are lowered into HW/SV structure, SV cleanup and
legalization passes prepare the result, and ExportVerilog prints it.

## How To Use And Read SV IR

When reading SV IR, first ask whether the operation is semantic hardware or
output syntax. An `sv.alwaysff` or `sv.ifdef` exists because the output needs
that SystemVerilog construct. A `comb.and` nearby may still represent ordinary
logic. The dialects are meant to mix.

Second, pay attention to procedural context. Some operations are valid only in
procedural regions, such as blocking assignments, nonblocking assignments,
`sv.write`, and file operations. Others are non-procedural declarations or
module-scope constructs, such as `sv.wire`, `sv.assign`, or `sv.include`.

Third, treat verbatim operations carefully. `sv.verbatim` and
`sv.verbatim.expr` are powerful because they can express unsupported or
tool-specific SystemVerilog. They are also harder for compiler passes to
understand. Use them when a real SV construct must be preserved and no typed
operation exists.

Finally, remember that SV is close to the output. Transformations here should
usually reduce redundancy, legalize output, improve readability, or preserve
emission intent. Heavy semantic optimization belongs earlier in the pipeline.

## What It Implies

The presence of `sv` implies that CIRCT is either preparing to emit
SystemVerilog or preserving a SystemVerilog-specific construct. It does not
mean the whole design has stopped being compiler IR: `sv` often appears
alongside `hw`, `comb`, `seq`, `ltl`, and `verif`. But it does mean that textual
output concerns are now important.

It also implies a tradeoff. The dialect can represent details that matter to
SystemVerilog users, such as macros, attributes, generate constructs, and
procedural blocks. In exchange, many operations are less abstract and less
portable than semantic hardware operations. Once code is expressed as `sv`,
later passes need to respect the intended printed form.

## Summary

The `sv` dialect is CIRCT's SystemVerilog emission dialect. It provides typed
operations for declarations, procedural statements, generate constructs,
interfaces, macros, verbatim text, functions, DPI imports, inout access, and
verification statements. It is needed when the compiler must preserve
SystemVerilog-specific structure and print high-quality output. Use it as the
late-stage bridge from semantic CIRCT hardware IR to actual SystemVerilog text.

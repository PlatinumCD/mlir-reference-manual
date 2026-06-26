# CIRCT Moore Dialect

The CIRCT `moore` dialect represents SystemVerilog after parsing, type checking, and elaboration. It is an ingestion dialect: its job is to faithfully preserve SystemVerilog constructs in MLIR before CIRCT simplifies, analyzes, and lowers them to smaller core dialects.

For a beginner, the key contrast is between `moore` and `sv`. The `sv` dialect is primarily an emission dialect: it is a good target when CIRCT wants to print SystemVerilog text. The `moore` dialect is a frontend dialect: it tries to capture the source language's types and semantics after elaboration, including four-state logic, procedures, tasks, queues, associative arrays, classes, DPI imports, system tasks, and timing/event controls.

That makes Moore one of the broadest dialects in this book. It is not a small hardware primitive dialect. It is a language semantic dialect for SystemVerilog.

## Why This Dialect Exists

SystemVerilog is not just structural RTL. It has modules, nets, variables, blocking and nonblocking assignments, initial and always procedures, event controls, delays, queues, associative arrays, strings, classes, virtual dispatch, DPI imports, and built-in system tasks. Lowering all of that directly into simple hardware IR would lose useful source-level meaning too early.

The `moore` dialect gives CIRCT a place to preserve that meaning. It is designed for the `ImportVerilog` path to translate a parsed, type-checked, and elaborated Slang AST into MLIR operations. Once the design is in Moore IR, CIRCT can run passes that resolve quirks and simplify language features before lowering to core dialects.

The dialect also models SystemVerilog's value domains. The `moore.iN` types are two-valued bit vectors, while `moore.lN` types are four-valued logic vectors. This distinction matters because unknown `X` and high-impedance `Z` values affect equality, comparisons, default values, and simulation behavior.

## When To Use It

You use Moore when you are importing, inspecting, or transforming SystemVerilog before final lowering. It is especially important when the source contains high-level SystemVerilog constructs that do not have a direct counterpart in `hw`, `comb`, `llhd`, or `sv`.

Use Moore to study how CIRCT represents source-level behavior. A `moore.procedure` tells you about `initial`, `final`, and `always` blocks. `moore.wait_event`, `moore.detect_event`, and `moore.wait_delay` preserve timing control. Queue and associative-array operations preserve dynamic container semantics. Class and vtable operations preserve object-oriented dispatch. DPI operations preserve foreign-function boundaries.

Avoid treating Moore as a final hardware netlist. It is intentionally higher level than the core dialects. The usual direction is Moore to simpler CIRCT dialects, not the other way around.

## Core Concepts

Moore models SystemVerilog after the language has been resolved. Types are explicit, modules are elaborated, and operations are no longer ambiguous parser artifacts. That is why it can include both high-level language constructs and precise low-level operations.

A `moore.module` is the structural container for a SystemVerilog module. It has ports and a graph-region body ending in `moore.output`. A `moore.instance` instantiates another Moore module. This is the structural side of the language.

Procedural behavior is captured by `moore.procedure`, `moore.coroutine`, and timing/event operations. A SystemVerilog task can suspend at timing controls, so Moore represents it as `moore.coroutine`. Procedures can model `initial`, `final`, `always`, `always_comb`, `always_latch`, and `always_ff`.

References are explicit. Variables, nets, global variables, class properties, aggregate elements, queue elements, struct fields, and union fields can all be represented as values or references. Assignment operations then distinguish continuous assignment, blocking procedural assignment, nonblocking assignment, and delayed forms.

## Operation Inventory By Domain

### Module, Procedure, And Process Structure

These operations describe module structure, procedure bodies, coroutines, and process completion:

`moore.module`, `moore.output`, `moore.instance`, `moore.procedure`, `moore.return`, `moore.unreachable`, `moore.coroutine`, `moore.call_coroutine`, `moore.complete`, `moore.fork`, `moore.wait_fork`.

### Declarations, References, And Assignments

These operations model SystemVerilog objects, reads, globals, references, and assignment styles:

`moore.variable`, `moore.net`, `moore.assigned_variable`, `moore.read`, `moore.global_variable`, `moore.get_global_variable`, `moore.assign`, `moore.delayed_assign`, `moore.blocking_assign`, `moore.nonblocking_assign`, `moore.delayed_nonblocking_assign`, `moore.concat_ref`, `moore.extract_ref`, `moore.dyn_extract_ref`, `moore.dyn_queue_ref_element`, `moore.struct_extract_ref`, `moore.union_extract_ref`, `moore.class.property_ref`.

### Timing And Event Control

These operations preserve simulation scheduling and sensitivity behavior:

`moore.wait_event`, `moore.detect_event`, `moore.wait_level`, `moore.detect_level`, `moore.wait_delay`.

### Constants And Type Conversions

These operations create constants or convert between Moore domains, builtin integers, real values, time values, packed values, and strings:

`moore.constant`, `moore.constant_time`, `moore.constant_string`, `moore.constant_real`, `moore.null`, `moore.conversion`, `moore.packed_to_sbv`, `moore.sbv_to_packed`, `moore.logic_to_int`, `moore.int_to_logic`, `moore.to_builtin_int`, `moore.from_builtin_int`, `moore.time_to_logic`, `moore.logic_to_time`, `moore.real_to_int`, `moore.sint_to_real`, `moore.uint_to_real`, `moore.int_to_string`, `moore.string_to_int`, `moore.convert_real`, `moore.trunc`, `moore.zext`, `moore.sext`, `moore.bool_cast`.

### Integer, Logic, Real, And Comparison Operations

These operations model arithmetic, bitwise logic, reductions, shifts, comparisons, real arithmetic, and handle comparisons:

`moore.neg`, `moore.fneg`, `moore.not`, `moore.reduce_and`, `moore.reduce_or`, `moore.reduce_xor`, `moore.add`, `moore.sub`, `moore.mul`, `moore.divu`, `moore.divs`, `moore.modu`, `moore.mods`, `moore.powu`, `moore.pows`, `moore.and`, `moore.or`, `moore.xor`, `moore.shl`, `moore.shr`, `moore.ashr`, `moore.uarray_cmp`, `moore.queue.cmp`, `moore.string_cmp`, `moore.eq`, `moore.ne`, `moore.case_eq`, `moore.case_ne`, `moore.casez_eq`, `moore.casexz_eq`, `moore.wildcard_eq`, `moore.wildcard_ne`, `moore.ult`, `moore.ule`, `moore.ugt`, `moore.uge`, `moore.slt`, `moore.sle`, `moore.sgt`, `moore.sge`, `moore.fadd`, `moore.fsub`, `moore.fdiv`, `moore.fmul`, `moore.fpow`, `moore.feq`, `moore.fne`, `moore.flt`, `moore.fle`, `moore.fgt`, `moore.fge`, `moore.handle_eq`, `moore.handle_ne`, `moore.handle_case_eq`, `moore.handle_case_ne`.

### Packed, Aggregate, Queue, Associative Array, Struct, And Union Operations

These operations build and decompose packed values and dynamic SystemVerilog containers:

`moore.concat`, `moore.replicate`, `moore.extract`, `moore.dyn_extract`, `moore.queue.resize`, `moore.dyn_queue_extract`, `moore.queue.set`, `moore.queue.from_unpacked_array`, `moore.queue.concat`, `moore.queue.insert`, `moore.queue.delete`, `moore.queue.clear`, `moore.push_back`, `moore.push_front`, `moore.pop_back`, `moore.pop_front`, `moore.array_create`, `moore.assoc_array_extract`, `moore.assoc_array_extract_ref`, `moore.assoc_array.delete`, `moore.assoc_array.clear`, `moore.assoc_array.size`, `moore.assoc_array.exists`, `moore.assoc_array.first`, `moore.assoc_array.last`, `moore.assoc_array.next`, `moore.assoc_array.prev`, `moore.open_uarray_create`, `moore.open_uarray_size`, `moore.open_uarray_delete`, `moore.struct_create`, `moore.struct_extract`, `moore.struct_inject`, `moore.union_create`, `moore.union_extract`, `moore.conditional`, `moore.yield`.

### Assertions And Coverage

These operations model immediate SystemVerilog verification statements inside procedures:

`moore.assert`, `moore.assume`, `moore.cover`.

### Formatting Operations

These operations build formatted strings and formatted fragments for display, diagnostic, and file-output system tasks:

`moore.fmt.literal`, `moore.fstring_to_string`, `moore.fmt.concat`, `moore.fmt.int`, `moore.fmt.real`, `moore.fmt.string`, `moore.fmt.hier_path`, `moore.fmt.char`.

### Builtin System Tasks And Functions

These operations model SystemVerilog builtins and system tasks:

`moore.builtin.realtobits`, `moore.builtin.bitstoreal`, `moore.builtin.shortrealtobits`, `moore.builtin.bitstoshortreal`, `moore.builtin.urandom_range`, `moore.builtin.time`, `moore.builtin.stop`, `moore.builtin.finish`, `moore.builtin.finish_message`, `moore.builtin.display`, `moore.builtin.severity`, `moore.builtin.fopen`, `moore.builtin.fclose`, `moore.builtin.fflush`, `moore.builtin.fdisplay`, `moore.builtin.plusargs_test`, `moore.builtin.plusargs_value`, `moore.builtin.clog2`, `moore.builtin.ln`, `moore.builtin.log10`, `moore.builtin.exp`, `moore.builtin.sqrt`, `moore.builtin.floor`, `moore.builtin.ceil`, `moore.builtin.sin`, `moore.builtin.cos`, `moore.builtin.tan`, `moore.builtin.asin`, `moore.builtin.acos`, `moore.builtin.atan`, `moore.builtin.sinh`, `moore.builtin.cosh`, `moore.builtin.tanh`, `moore.builtin.asinh`, `moore.builtin.acosh`, `moore.builtin.atanh`, `moore.builtin.size`.

### Classes, VTables, And DPI

These operations preserve SystemVerilog class structure, virtual dispatch, and DPI-C imports:

`moore.class.propertydecl`, `moore.class.methoddecl`, `moore.class.classdecl`, `moore.class.new`, `moore.class.upcast`, `moore.vtable.load_method`, `moore.vtable`, `moore.vtable_entry`, `moore.func.dpi`, `moore.func.dpi.call`.

### String Methods

These operations model SystemVerilog string methods:

`moore.string.len`, `moore.string.put`, `moore.string.get`, `moore.string.toupper`, `moore.string.tolower`, `moore.string.compare`, `moore.string.icompare`, `moore.string.substr`, `moore.string.atoi`, `moore.string.atohex`, `moore.string.atooct`, `moore.string.atobin`, `moore.string.atoreal`, `moore.string.itoa`, `moore.string.hextoa`, `moore.string.octtoa`, `moore.string.bintoa`, `moore.string.realtoa`, `moore.string.concat`.

## Transformations And Conversions

| Pass | Role |
| --- | --- |
| `moore-simplify-procedures` | Inserts local shadow variables for module-level variables modified by procedures so later mem2reg-style promotion can simplify procedure bodies. |
| `moore-simplify-refs` | Splits aggregate assignment left-hand sides and simplifies hard-to-lower references such as concatenated references and dynamic queue element references. |
| `moore-create-vtables` | Computes `moore.vtable` operations from class declarations before symbol dead-code elimination can remove needed class symbols. |
| `convert-moore-to-core` | Lowers Moore to CIRCT core dialects such as `comb`, `hw`, `llhd`, `cf`, `scf`, `math`, `llvm`, `sim`, and `verif`. |

## How The Lowering Flow Works

A typical Moore flow starts after SystemVerilog import. The design is parsed, type checked, elaborated, and represented with `moore` types and operations. At this point, the IR still remembers source-level concepts such as procedures, classes, timing controls, queues, DPI calls, and four-state logic.

Then Moore-specific simplification passes resolve awkward source-level constructs. `moore-simplify-procedures` helps procedural code become easier to optimize. `moore-simplify-refs` breaks aggregate references into simpler pieces. `moore-create-vtables` materializes virtual dispatch tables so class methods can be lowered without re-solving class hierarchy semantics later.

Finally, `convert-moore-to-core` translates the high-level SystemVerilog model into smaller CIRCT dialects. Arithmetic and hardware structure move toward `comb` and `hw`. Simulation and DPI concepts move toward `sim`. Verification concepts move toward `verif`. Control flow can use `cf` and `scf`, and lower-level math or LLVM IR can appear where appropriate.

## How To Read Moore IR

Start at `moore.module`. The module operation tells you the structural boundary, and `moore.output` tells you what drives output ports. Look for `moore.instance` to understand hierarchy.

Then inspect procedures. `moore.procedure` is where source-level behavior lives. Its kind tells you whether it came from an `initial`, `final`, `always`, `always_comb`, `always_latch`, or `always_ff` block. Inside procedures, pay attention to blocking versus nonblocking assignments and to timing controls.

Next, inspect values versus references. A `moore.read` turns a reference into a value. Assignment operations consume references as destinations. If you see `moore.concat_ref`, `moore.extract_ref`, queue element references, or struct/union references, you are looking at source-level l-values that may need simplification before lowering.

For arithmetic, remember the type domain. Operations over `!moore.iN` are two-valued. Operations over `!moore.lN` are four-valued and must preserve `X` and `Z` behavior. Logical equality, case equality, wildcard equality, and relational operations are not interchangeable.

For object-oriented code, look at class declarations and vtables together. `moore.class.classdecl` records the class structure. `moore.vtable` and `moore.vtable_entry` capture resolved virtual dispatch. `moore.vtable.load_method` reads the selected method entry for a class handle.

## What It Implies

Seeing Moore IR means CIRCT is still close to SystemVerilog semantics. This is the right place to debug source import, type resolution, procedural behavior, classes, dynamic containers, and simulation-specific tasks.

The dialect also implies that lowering is not just a syntax translation. A simple-looking SystemVerilog statement may involve references, four-state logic, timing semantics, assignment scheduling, or class dispatch. Moore keeps those details visible long enough for CIRCT to lower them deliberately.


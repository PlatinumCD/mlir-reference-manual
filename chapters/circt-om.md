# CIRCT `om` Dialect

The CIRCT `om` dialect is the Object Model dialect. It is used to capture design intent and domain-model information around a hardware design, rather than the RTL behavior itself. If `hw`, `comb`, `seq`, and `sv` describe modules, signals, logic, registers, and emitted SystemVerilog, then `om` describes typed metadata and object relationships that tools can query later.

The beginner mental model is "typed objects for hardware-adjacent information." OM can model things such as physical hierarchy, clocks, power domains, bus interfaces, software bringup metadata, constraints, and generated collateral. It is deliberately generic: instead of defining a new dialect for every domain immediately, OM provides a small set of typed classes, objects, fields, lists, strings, integers, references, and paths.

## Operation Inventory

The `om` dialect defines these operations:

```text
om.any_cast
om.basepath_create
om.class
om.class.extern
om.class.fields
om.constant
om.elaborated_object
om.frozenbasepath_create
om.frozenpath_create
om.frozenpath_empty
om.integer.add
om.integer.and
om.integer.mul
om.integer.or
om.integer.shl
om.integer.shr
om.integer.xor
om.list_concat
om.list_create
om.object
om.object.field
om.path_create
om.path_empty
om.prop.eq
om.property_assert
om.string.concat
om.unknown
```

## Types And Attributes

The core types are `!om.class.type`, `!om.ref`, `!om.list`, `!om.sym_ref`, `!om.integer`, `!om.string`, `!om.basepath`, `!om.path`, `!om.frozenbasepath`, `!om.frozenpath`, and `!om.any`.

`!om.class.type<@ClassName>` is the type of an object reference to an OM class. `!om.ref` is a reference to a hardware entity. `!om.list<T>` is a typed list. `!om.integer` is an arbitrary-width integer for property computation. `!om.string` is a property string. `!om.basepath` and `!om.path` are symbolic paths into hardware hierarchy. The frozen path types are used after hierarchy names have been fixed.

The main typed attributes are `#om.ref`, `#om.sym_ref`, `#om.list`, `#om.path`, and `#om.integer`. `#om.ref` wraps a hardware inner reference. `#om.sym_ref` wraps a flat symbol reference. `#om.path` stores a resolved instance path. `#om.integer` stores arbitrary integer values. The target-kind enum has `dont_touch`, `instance`, `member_instance`, `member_reference`, and `reference`, which tell path operations what kind of hardware target they name.

## Classes, Objects, And Fields

`om.class` defines a class in the object model. A class has a symbol name, formal parameters, field names, field types, and a body region. Its block arguments are the formal parameters. The class body computes field values and ends with `om.class.fields`, the terminator that returns the public field values.

`om.class.extern` declares an external class. It is used when one OM module references a class whose definition is expected to come from another module. This is important for separate compilation and linking of object-model fragments.

`om.object` instantiates a class with actual parameters and returns a value of the corresponding `!om.class.type`. `om.object.field` reads a named field from an object. Together these two operations are the basic "construct and query" surface of the dialect.

`om.elaborated_object` is an internal form produced by elaboration. Instead of storing constructor parameters, it stores explicit field operands in declared field order. Frontends are not expected to create it directly. It exists so the compiler can inline object construction and fold field accesses.

The implication is that OM classes are closer to typed records with constructors than to software classes with methods. They describe a public data API over a domain model.

## Values And Expressions

`om.constant` materializes typed OM constants. It can hold typed attributes such as OM strings, symbols, integers, references, and lists. The dialect has a constant materializer that produces this op.

`om.list_create` builds a typed list from values of the same type. `om.list_concat` concatenates lists of the same list type. Lists let a model expose repeated objects, paths, constraints, or property values without inventing a domain-specific container for each case.

`om.string.concat` concatenates OM string values. It folds constants and has canonicalization patterns that flatten nested concatenations and simplify string expression trees.

`om.integer.add`, `om.integer.mul`, `om.integer.shl`, and `om.integer.shr` operate on arbitrary-precision `!om.integer` values. `om.integer.and`, `om.integer.or`, and `om.integer.xor` operate on integer values of the same MLIR integer type. `om.prop.eq` compares OM property values of string, integer, or `i1` type and returns `i1`.

`om.property_assert` checks that a property condition is true during OM evaluation. If the evaluator or elaboration pass can prove the condition false, evaluation fails with the provided message. This is how a model can state invariants about its parameters or derived fields.

`om.any_cast` casts a concrete value to `!om.any`. In the evaluator it is a no-op wrapper around the underlying value. `om.unknown` represents an unknown value, often used to tie off object connections that are intentionally unspecified or not important for the current model.

## Paths Into Hardware

Path operations connect the object model to CIRCT hardware hierarchy. `om.basepath_create` extends a base path with a hardware instance identified by a symbol. `om.path_create` creates a path from a base path to a target, using a target kind such as reference or instance. `om.path_empty` creates an empty path.

These symbolic path operations are useful while the hardware hierarchy can still change. A model can refer to a `hw.hierpath` or inner reference without baking in final Verilog names.

`om-freeze-paths` converts symbolic paths to frozen paths once hierarchy names are stable. It replaces `om.basepath_create` with `om.frozenbasepath_create`, `om.path_create` with `om.frozenpath_create`, and `om.path_empty` with `om.frozenpath_empty`. It also updates list, object-field, class argument, and class field types from path types to frozen path types. The pass uses the HW instance graph and inner symbol tables to turn symbolic references into final path attributes.

That sequencing matters. Before freeze, paths are robust against hardware renaming. After freeze, paths are hard-coded strings suitable for downstream collateral.

## Transformations

The named OM passes are:

```text
om-freeze-paths
om-link-modules
om-elaborate-object
strip-om
```

`om-link-modules` links separated OM modules into one module. It flattens nested modules, resolves `om.class.extern` declarations against class definitions, checks that external declarations match definitions, renames private classes when symbol collisions require it, updates `!om.class.type` references and `om.object` class references, and erases external class declarations after resolution.

`om-elaborate-object` performs compile-time evaluation of OM objects. It can elaborate one target class or all public classes. It recursively inlines `om.object` instantiations by cloning class bodies with actual parameters substituted for formal arguments. It then replaces object instantiations with `om.elaborated_object`, folds `om.object.field` into direct field operands, propagates `om.unknown` through pure OM operations, evaluates property assertions, and reports unevaluated operations unless `allow-unevaluated` is enabled.

`strip-om` removes top-level OM operations from a module. It is useful when a later hardware pipeline does not need object-model information, or when OM metadata should be intentionally discarded before export.

The `om-linker` tool builds a pipeline around `om-link-modules` and, by default, runs `om-elaborate-object` over all public classes with unevaluated operations allowed. It is a practical tool for combining object-model fragments from multiple files.

## Reductions And Canonicalization

OM has constant folding and canonicalization for value expressions. Integer operations fold when operands are known. `om.string.concat` folds and flattens nested string concatenations. `om.constant` folds to its attribute value. `om.prop.eq` folds equality when both sides are known. The verifier checks class signatures, field declarations, object-field symbol uses, elaborated object symbol uses, attributes, paths, and integer attributes.

The dialect also registers reducer patterns for `circt-reduce`. These are not normal optimization passes, but they are transformations used to shrink failing OM test cases. The named reduction patterns include `om-object-to-unknown`, `om-list-element-pruner`, `om-class-field-pruner`, `om-class-parameter-pruner`, `om-unused-class-remover`, `om-anycast-of-unknown-simplifier`, and `om-op-to-unknown`.

These reducer patterns show what the dialect considers structurally removable during debugging: unused classes, unused fields, unused parameters, list elements, and object values that can become unknown without losing the interesting failure.

## When To Use It

Use OM when you need stable, typed metadata tied to a hardware design. It is a good fit for information that should be queried by tools, not emitted directly as gates or always blocks. Examples include domain descriptions, external collateral inputs, design hierarchy references, address or bus metadata, and high-level properties that must remain connected to RTL names.

Do not use OM for ordinary combinational logic, registers, or module structure. Those belong in `comb`, `seq`, `hw`, `sv`, or another behavior-level dialect. OM should describe what the design means to surrounding tools.

The reason OM is needed is that hardware projects have many structured models around the RTL, and JSON side files or ad hoc annotations lose type safety and compiler integration. OM keeps those models in MLIR, gives them symbols and types, supports linking, and can elaborate object graphs at compile time.

## How To Read OM IR

Start by finding `om.class` operations. Their parameters tell you what is required to construct an object, and their fields tell you what the class publicly exposes. Then follow `om.object` operations to see how classes are instantiated.

Next, inspect `om.object.field` uses. They are the query points: some other part of the model wants a named field from an object. If the IR has already been elaborated, look for `om.elaborated_object` and direct field operands instead.

Finally, look at paths. If you see `om.path_create` or `om.basepath_create`, the model still contains symbolic hierarchy references. If you see `om.frozenpath_create` or `om.frozenbasepath_create`, the model has been resolved against final hardware names.

The broader implication is that `om` gives CIRCT a typed object layer for hardware design knowledge. It is not a replacement for RTL IR; it is a structured companion to RTL that lets downstream tools ask meaningful questions.

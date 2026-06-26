# `dlti` Dialect

## Beginner Summary

`dlti` means Data Layout and Target Information. It is the dialect MLIR uses to
store facts about how values are laid out in memory and what target hardware or
runtime assumptions are in effect.

Unlike dialects such as `arith` or `scf`, `dlti` is not mostly a list of
executable operations. It is mostly a set of attributes and interfaces. Those
attributes attach target facts to operations such as modules, functions, GPU
modules, or other layout scopes. Analyses and lowering passes can then query
those facts.

The most common beginner example is the bitwidth of the `index` type:

```mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>
} {
}
```

That says that, in this module's data-layout scope, `index` should be treated
as a 32-bit type. Without such an entry, MLIR's default data layout treats
`index` as 64-bit.

## Why This Dialect Exists

Compilers need answers to target-dependent questions:

- How large is this type?
- What ABI alignment does this type require?
- What preferred alignment should be used?
- How many bits does `index` have on this target?
- What memory space is the default memory space?
- Is the target little-endian or big-endian?
- What target device has a specific cache size or vector width?

Those answers are not always global constants. MLIR can contain nested modules,
device modules, kernels, or regions that target different execution contexts.
`dlti` gives MLIR a structured way to attach these facts to the IR and to
combine nested layout scopes safely.

## When It Matters

`dlti` matters when target facts affect legality or generated code.

You are likely to see it when:

- lowering `index`-typed arithmetic to fixed-width integer arithmetic;
- converting `memref`, `func`, `cf`, `ub`, `arith`, or other dialects to
  `llvm`;
- importing or exporting LLVM IR data-layout information;
- outlining GPU kernels with a specific data layout;
- lowering GPU code to target dialects such as `nvvm` or `rocdl`;
- making target-aware transform decisions, such as choosing a vector width;
- representing multi-device target information, such as CPU and GPU facts in
  the same module.

For many high-level examples, DLTI is invisible. As soon as code generation
needs concrete sizes, alignments, pointer widths, memory spaces, or target
metadata, it becomes important.

## When To Use It

Use `dlti` when an operation or module needs scoped target/data-layout facts
that other analyses or lowering passes should be able to query.

Good uses:

- attach a data layout specification to a module;
- describe `index` bitwidth for lowering;
- describe pointer or memory-space layout information for pointer-like types;
- attach target system or target device information;
- give transform scripts a structured way to query hardware facts;
- preserve LLVM data-layout information during import/export.

Avoid using `dlti` as a general dumping ground for arbitrary frontend metadata.
It can store arbitrary key-value maps, but the best use is information that is
really about data layout, target properties, or target-dependent transform
decisions.

## Core Concepts

### Data Layout Scope

A data layout is scoped. An operation such as a module can carry a
`dlti.dl_spec` attribute. Queries inside that scope use the nearest relevant
layout and combine it with enclosing layouts when needed.

This matters for nested modules and device code. An outer module may describe
host facts while an inner module or GPU module describes device facts.

### Data Layout Specification

A data layout specification is a set of entries:

```mlir
#dlti.dl_spec<
  index = 32 : i32,
  "dlti.endianness" = "little"
>
```

Each entry has a key and a value. The key can be a type, such as `index` or
`i32`, or a string identifier, such as `"dlti.endianness"`.

### DLTI Map

`#dlti.map` is a generic nested key-value map. It is useful for arbitrary
target information that still needs query semantics.

```mlir
module attributes {
  dlti.map = #dlti.map<
    "CPU" = #dlti.map<"L1_cache_size_in_bytes" = 4096 : i32>,
    "GPU" = #dlti.map<"max_vector_op_width" = 128 : i32>
  >
} {
}
```

### Target System And Target Device Specs

`#dlti.target_system_spec` describes a system containing one or more named
devices. Each device maps to a `#dlti.target_device_spec`.

```mlir
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "L1_cache_size_in_bytes" = 4096 : ui32>,
    "GPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 128 : i64>
  >
} {
}
```

This is useful when one IR module carries information for multiple execution
targets.

### DataLayout Analysis

`DataLayoutAnalysis` builds `DataLayout` objects for scopes and lets passes
query layout facts. Conversions to LLVM use this when they need to decide type
sizes, alignments, and `index` bitwidth.

## Operations

The `dlti` dialect does not define ordinary payload operations in this checkout.
Its primary IR surface is attributes.

There is one closely related transform operation:

| Operation | Dialect | Purpose |
|---|---|---|
| `transform.dlti.query` | `transform` | Query DLTI information attached to payload IR and return the associated attribute as a transform parameter. |

Because the operation name is `transform.dlti.query`, it belongs to the
Transform dialect extension, not to the payload `dlti` dialect itself.

### `transform.dlti.query`

This op takes a handle to payload IR and a list of keys. It walks from the
target operation to the closest DLTI-queryable attribute or ancestor and
returns the matched attribute as a transform parameter.

Example:

```mlir
module attributes { test.dlti = #dlti.map<"test.id" = 42 : i32> } {
  func.func private @f()
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg
      : (!transform.any_op) -> !transform.any_op
    %module = transform.get_parent_op %funcs
      : (!transform.any_op) -> !transform.any_op
    %param = transform.dlti.query ["test.id"] at %module
      : (!transform.any_op) -> !transform.any_param
    transform.yield
  }
}
```

The query can also use type keys:

```mlir
%param = transform.dlti.query [i32, "width_in_bits"] at %module
  : (!transform.any_op) -> !transform.any_param
```

## Attributes

### `#dlti.dl_entry`

A single key-value pair used inside DLTI specifications and maps.

```mlir
#dlti.dl_entry<"dlti.endianness", "little">
#dlti.dl_entry<index, 32 : i32>
```

Keys are either string identifiers or types. Values are attributes.

### `#dlti.dl_spec`

A data layout specification. This is the main attribute used for layout
queries.

```mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    index = 32 : i32,
    "dlti.endianness" = "little",
    "dlti.stack_alignment" = 128 : i64
  >
} {
}
```

Common string keys include:

- `"dlti.endianness"`;
- `"dlti.mangling_mode"`;
- `"dlti.default_memory_space"`;
- `"dlti.alloca_memory_space"`;
- `"dlti.program_memory_space"`;
- `"dlti.global_memory_space"`;
- `"dlti.stack_alignment"`;
- `"dlti.function_pointer_alignment"`;
- `"dlti.legal_int_widths"`.

### `#dlti.map`

A general DLTI key-value map. It supports nested maps and recursive query.

```mlir
#dlti.map<
  "CPU" = #dlti.map<"cache" = #dlti.map<"L1" = 32768 : i32>>
>
```

### `#dlti.target_device_spec`

A map of properties for one target device.

```mlir
#dlti.target_device_spec<
  "max_vector_op_width" = 128 : i64
>
```

### `#dlti.target_system_spec`

A map from device IDs to target device specs.

```mlir
#dlti.target_system_spec<
  "CPU" = #dlti.target_device_spec<"max_vector_op_width" = 64 : i64>,
  "GPU" = #dlti.target_device_spec<"max_vector_op_width" = 128 : i64>
>
```

### `#dlti.function_pointer_alignment`

An attribute for function pointer alignment information.

```mlir
#dlti.function_pointer_alignment<64, function_dependent = false>
```

It can be used as the value for the `"dlti.function_pointer_alignment"` data
layout entry.

## Transformations

DLTI does not usually transform program computation by itself. It influences
other transformations and conversions by providing target facts.

Important related transformations and analyses:

- `DataLayoutAnalysis` builds layout objects for operations and lets passes
  query the closest applicable layout.
- `transform.dlti.query` lets transform dialect scripts query DLTI attributes
  from payload IR.
- `llvm-target-to-data-layout` derives a DLTI data-layout spec from an LLVM
  target attribute and attaches it as `dlti.dl_spec`.
- `gpu-kernel-outlining` can attach a `dlti.dl_spec` to outlined GPU modules
  through its `data-layout-str` option.
- LLVM lowering passes use data layout to decide target-dependent details such
  as index bitwidth, pointer lowering, and ABI layout.

## Conversions And Lowering Paths

DLTI is normally not "lowered away" like `linalg` or `scf`. Instead, it is
metadata consumed by conversions.

Common paths:

- LLVM IR import can translate LLVM data-layout strings into DLTI attributes.
- LLVM IR export can translate `dlti.dl_spec` back into LLVM data-layout
  information when possible.
- `convert-to-llvm` can use `DataLayoutAnalysis` in dynamic conversion mode.
- `convert-func-to-llvm`, `memref` to LLVM lowering, `cf` to LLVM lowering,
  `arith` to LLVM lowering, `index` to LLVM lowering, and related conversions
  use data-layout information when choosing concrete target representations.
- GPU-to-NVVM and GPU-to-ROCDL lowering use data layout for device module and
  LLVM lowering details.

The practical rule is:

If a conversion needs to turn abstract MLIR types into concrete target-level
types, DLTI may affect the answer.

## Example IR

### Data Layout For `index`

```mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>
} {
  func.func @use_index(%i: index) -> index {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %i, %c1 : index
    return %0 : index
  }
}
```

This tells later lowering that `index` should be treated as 32 bits in this
module's layout scope.

### Target Information For Transform Decisions

```mlir
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 64 : i64>,
    "GPU" = #dlti.target_device_spec<
      "max_vector_op_width" = 128 : i64>
  >
} {
  func.func @kernel(%arg0: tensor<?xf32>) {
    return
  }
}
```

A transform script can query `["CPU", "max_vector_op_width"]` or
`["GPU", "max_vector_op_width"]` and choose different transformations.

## Mental Model

Think of `dlti` as scoped target metadata with query semantics.

It does not say "do this computation." It says "inside this scope, use these
facts about the target and memory layout when deciding how computation should
be transformed or lowered."

## Gotchas

- DLTI is mostly attributes, not payload operations.
- `transform.dlti.query` is a Transform dialect op, even though it queries DLTI
  data.
- Keys must be valid: empty string keys are rejected, duplicate keys are
  rejected, and some known keys require specific value forms.
- Nested data-layout specs must be compatible with enclosing specs.
- Not every type supports custom data layout entries. Unsupported type entries
  are verifier errors.
- `dlti.map` can hold arbitrary-looking metadata, but using it for unrelated
  frontend metadata makes later compiler behavior harder to reason about.
- A pass that caches `DataLayout` must rebuild it after changing layout-affecting
  ancestor operations.
- If no entry is provided for `index`, MLIR's default model treats it as
  64-bit.

## Source Map

Primary DLTI files:

- `mlir/include/mlir/Dialect/DLTI/DLTIBase.td`
- `mlir/include/mlir/Dialect/DLTI/DLTIAttrs.td`
- `mlir/include/mlir/Dialect/DLTI/DLTI.h`
- `mlir/lib/Dialect/DLTI/DLTI.cpp`
- `mlir/include/mlir/Dialect/DLTI/Traits.h`
- `mlir/lib/Dialect/DLTI/Traits.cpp`

Transform extension:

- `mlir/include/mlir/Dialect/DLTI/TransformOps/DLTITransformOps.td`
- `mlir/lib/Dialect/DLTI/TransformOps/DLTITransformOps.cpp`

Data layout infrastructure:

- `mlir/docs/DataLayout.md`
- `mlir/include/mlir/Interfaces/DataLayoutInterfaces.td`
- `mlir/lib/Interfaces/DataLayoutInterfaces.cpp`
- `mlir/lib/Analysis/DataLayoutAnalysis.cpp`

Tests:

- `mlir/test/Dialect/DLTI/valid.mlir`
- `mlir/test/Dialect/DLTI/invalid.mlir`
- `mlir/test/Dialect/DLTI/query.mlir`
- `mlir/test/Dialect/DLTI/roundtrip.mlir`
- `mlir/test/Interfaces/DataLayoutInterfaces/`

Important users:

- `mlir/include/mlir/Conversion/Passes.td`
- `mlir/lib/Conversion/ConvertToLLVM/ConvertToLLVMPass.cpp`
- `mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp`
- `mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp`
- `mlir/lib/Target/LLVMIR/DataLayoutImporter.cpp`
- `mlir/lib/Target/LLVMIR/ModuleTranslation.cpp`
- `mlir/lib/Target/LLVMIR/Transforms/TargetToDataLayout.cpp`
- `mlir/include/mlir/Dialect/GPU/Transforms/Passes.td`

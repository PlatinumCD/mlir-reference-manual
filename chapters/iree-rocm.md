# IREE `rocm` Dialect

The IREE `rocm` dialect is the AMD GPU target-support dialect used by IREE's ROCM backend. It is not a general-purpose tensor dialect, and it is not where most computation is written. Instead, it gives the ROCM target backend a place to store AMD-specific code generation metadata, embedded tuning modules, specialization patterns, and microkernel providers.

For a beginner, the important idea is that `rocm` is a bridge between high-level IREE GPU codegen and the final AMD GPU executable. Earlier dialects describe what the program computes. The IREE GPU and codegen dialects describe tiling, workgroups, layouts, and lowering strategies. The `rocm` dialect adds target-specific facts such as "use this embedded tuning spec" or "this executable can use ROCM microkernels." Those facts are then consumed by IREE passes before the program is translated through LLVM GPU lowering, ROCDL, LLVM IR, and finally HSACO or a ROCM-compatible SPIR-V bundle.

This dialect matters because AMD GPU codegen is shaped by hardware details. The target architecture, such as `gfx942`, controls available MFMA instructions, subgroup size choices, memory limits, and the tuning specs that should be tried. The `rocm` dialect keeps those backend choices explicit in MLIR instead of hiding them completely inside command-line options or C++ side tables.

## Operation Inventory

The `rocm` dialect defines no operations in this checkout. There are no `rocm.*` operation names to list.

Its public inventory is attribute-based:

```text
#rocm.builtin.tuning_module
#rocm.ukernel_provider
#rocm.tensor_ukernel_provider
#rocm.ukernel_info
#rocm.ukernel_iteration_size_constraint
```

This is a useful dialect-design example. A dialect does not need operations to be important. It can exist mainly to define attributes, interfaces, parsing rules, and target-specific services used by other passes.

## Attribute Inventory

`#rocm.builtin.tuning_module<"...">` references an embedded MLIR tuning module. The filename mirrors files under IREE's ROCM builtin tuning directory, such as `iree_default_tuning_spec_gfx942.mlir`. The attribute implements IREE's stored-module interface, so a later pass can ask for the referenced module and get a parsed `module` operation. The dialect caches parsed builtin modules so repeated access does not repeatedly parse the same text.

`#rocm.ukernel_provider` is the provider for bitcode-style ROCM microkernels. It implements IREE's ukernel provider interface and knows how to replace selected high-level ukernel descriptors with `iree_codegen.ukernel.generic` operations. In this checkout, the implementation handles argmax ukernels and a multi-MMA MFMA integer ukernel. These become calls into embedded AMDGPU bitcode with `vm.import.module = "rocm"`.

`#rocm.tensor_ukernel_provider` is the provider for MLIR tensor-based ROCM microkernels. Instead of selecting an external bitcode symbol, it can look up a private MLIR function and use it as the implementation for a tensor ukernel. It also implements data-layout selection for ukernels by matching element types, iteration sizes, target architecture, and supported MMA intrinsics.

`#rocm.ukernel_info<...>` is metadata attached to built-in MLIR ukernel functions. Its `match` dictionary describes when a ukernel applies. Common keys include element `types`, target `archs`, and `iteration_sizes_constraints`. The attribute also has an optional MMA-like layout attribute, such as an IREE GPU data-tiled MMA layout, and a `benefit` used when multiple ukernels match.

`#rocm.ukernel_iteration_size_constraint<...>` is a helper attribute used inside `#rocm.ukernel_info`. It can constrain one iteration dimension by index, minimum size, maximum size, or divisibility. For example, a matmul ukernel can require that the M dimension be at least a certain size and divisible by 64. These constraints keep the tensor ukernel provider from selecting a microkernel whose shape assumptions do not fit the actual operation.

## Basic IR Shape

A typical place to see the dialect is on a HAL executable target:

```mlir
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.default_tuning_spec =
    #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">,
  iree_codegen.ukernel_provider = #rocm.ukernel_provider,
  ukernels = "all"
}>
```

The `hal.executable.target` says that this executable variant targets the ROCM backend. The format string says how the compiled executable will be represented. The configuration dictionary carries codegen information. The `rocm` attributes are part of that dictionary, so they are visible to normal MLIR passes.

The dialect also has built-in dependencies on AMDGPU, bufferization, GPU, IREE TensorExt, IREE Util, ROCDL, and vector. Those dependencies are loaded because embedded ROCM builtin modules and ukernels may contain operations from those dialects.

## Builtin Modules

The ROCM dialect owns an embedded data directory for builtins. During dialect initialization, it registers several groups of embedded files:

```text
default tuning specs
specialization PDL patterns
MLIR ukernel PDL patterns
MLIR ukernel implementations
```

The builtin tuning spec path is used by `#rocm.builtin.tuning_module`. The specialization and ukernel pattern files are parsed by the ROCM PDL-pattern passes. The MLIR ukernel implementation files contain `util.func` operations with `#rocm.ukernel_info` metadata.

This design lets IREE ship target knowledge inside the compiler. The user does not need to manually write a transform module for every common AMD GPU target. If a matching embedded tuning spec or ukernel pattern exists, the ROCM target backend can attach and apply it automatically.

## Microkernel Selection

ROCM supports two related microkernel paths.

The first path is bitcode ukernels. A linalg or IREE codegen operation can carry an `iree_codegen.ukernel` descriptor whose implementation is bitcode. The `#rocm.ukernel_provider` can lower supported descriptors to `iree_codegen.ukernel.generic` operations. The concrete examples in this checkout include AMDGPU argmax variants for `f16`, `bf16`, and `f32` inputs with `i32` or `i64` indices, plus a multi-MMA MFMA `i8` to `i32` ukernel.

The second path is tensor ukernels. These are MLIR functions embedded with the compiler. PDL patterns can annotate matching contractions with tensor ukernel descriptors, and the driver pass can then embed the required private `util.func` into the module. The tensor provider can select a data-tiled layout by scanning the built-in ukernel functions, reading their `#rocm.ukernel_info`, checking element types and shape constraints, and verifying that the target GPU supports the required MMA intrinsic.

The implication is that `rocm` is not just a naming layer. It participates in choosing whether a high-level operation stays generic, gets a specialized lowering configuration, becomes a bitcode ukernel call, or gets replaced by an embedded tensor-level ukernel implementation.

## Transformations And Passes

The dialect registers these ROCM-specific passes:

```text
iree-rocm-apply-builtin-pdl-patterns
iree-rocm-apply-builtin-pdl-patterns-driver
```

`iree-rocm-apply-builtin-pdl-patterns` is a function-like interface pass. It takes a list of target names and can enable specialization patterns, tensor ukernel patterns, or both. When specialization is enabled, it looks for builtin files named like `specialization_patterns_<target>.mlir`. When tensor ukernels are enabled, it filters embedded builtin names for files such as `ukernel_patterns_gfx942.mlir` and `ukernel_patterns_gfx950.mlir`.

Those builtins are PDL modules. The pass parses them, registers native constraints and rewrite hooks, and applies the resulting patterns greedily. The native helpers can check whether an operation has an attribute, match contraction-like linalg operations, test dimension bounds, test divisibility, test products of dimensions, and annotate operations in place. In practice, these patterns can add attributes such as `iree_codegen.specialization_ranges`, `iree_codegen.ukernel`, `compilation_info`, and an internal `rocm.builtin_name` marker.

`iree-rocm-apply-builtin-pdl-patterns-driver` is a module pass. It reads the GPU target attribute from the module, runs the function-level builtin PDL pass for that target architecture, then embeds any required MLIR ukernel implementations. It searches for operations annotated with `rocm.builtin_name` and an IREE ukernel descriptor, loads the referenced builtin module, finds the named ukernel function, moves that function into the current module, makes it private, and removes the temporary marker.

The driver is important because matching a tensor ukernel is not enough. The implementation function must also be present in the module before later lowering can replace the annotated operation with a call.

## Target Backend Flow

Most ROCM behavior is wired through IREE's HAL target backend rather than through operations owned by the `rocm` dialect. The backend registers a target backend named `rocm` and target devices named `hip` and `amdgpu`.

The main option is `--iree-rocm-target`, with values such as `gfx942` or product and architecture aliases normalized by IREE GPU target utilities. `--iree-rocm-target-features` accepts explicit `+sramecc`, `-sramecc`, `+xnack`, or `-xnack` feature choices. The backend builds AMDGPU target IDs by appending those feature suffixes to the architecture when it needs an AMDGPU container target string.

When the backend creates a default executable target, it usually produces a `#hal.executable.target<"rocm", ...>` attribute. For legacy HIP device IDs, the format is `rocm-hsaco-fb`. In AMDGPU container mode, the format can be the AMDGPU target ID. If the SPIR-V path is enabled, the format is `rocm-spirv-fb`.

The backend adds IREE GPU target information to the configuration dictionary, may attach an encoding layout resolver, may attach a default ROCM tuning module, records the `ukernels` option, and adds either `#rocm.ukernel_provider` or `#rocm.tensor_ukernel_provider` when the corresponding ukernel features are enabled.

During configuration, the backend can run builtin PDL patterns for runtime specialization and tensor ukernels. It then runs common IREE GPU codegen configuration, materializes tuning specs, materializes user configs, and selects the LLVM GPU lowering strategy. During translation, it runs the LLVM GPU codegen pipeline in ROCM mode.

## Conversion And Serialization

The `rocm` dialect itself does not define a standalone conversion pass such as `convert-rocm-to-llvm`. Instead, ROCM conversion happens through the IREE target backend pipeline.

The lowering path moves executable bodies through IREE's LLVM GPU codegen pipeline, then translates MLIR LLVM dialect to an `llvm::Module`. For native ROCM output, the backend creates an AMDGPU target machine for a triple like `amdgcn-amd-amdhsa`, sets AMDGPU features such as wavefront size, links user bitcode and HIP device libraries when needed, sets HIP platform globals, runs LLVM optimization, emits an object, and packages it as HSACO.

For SPIR-V mode, the backend uses a `spirv64-amd-amdhsa` target, emits SPIR-V binary data, then wraps it in a clang offload bundle expected by the HIP runtime. The final binary can be wrapped in a HIP flatbuffer container, an AMDGPU executable container, or a raw HSACO image, depending on the selected container type.

This means that by the time `rocm` attributes have done their job, most visible IR is no longer in the `rocm` dialect. The attributes have guided tuning, specialization, ukernel embedding, target feature selection, and executable serialization.

## When To Use It

Use the `rocm` dialect when working inside IREE's AMD GPU backend. You will usually not author `rocm` IR by hand unless you are writing tests, adding tuning specs, or debugging target configuration. Instead, you will see its attributes appear on HAL executable targets and builtin ukernel functions.

Inspect `rocm` attributes when an AMD GPU compilation behaves differently from expected. If a default tuning spec did not apply, check `iree_codegen.default_tuning_spec`. If a microkernel was not selected, check the `ukernels` configuration, the `iree_codegen.ukernel_provider`, and the target architecture. If a tensor ukernel did not materialize, check the PDL pattern match, the `rocm.builtin_name` marker before driver embedding, and the `#rocm.ukernel_info` constraints on the candidate ukernel function.

Do not use `rocm` as a portable representation of GPU computation. Computation should remain in dialects such as `linalg`, `vector`, `gpu`, `iree_codegen`, `iree_gpu`, `amdgpu`, `rocdl`, and LLVM-related dialects. The `rocm` dialect's job is to carry AMD-target-specific decisions through IREE's compilation pipeline.

The main implication is that ROCM compilation in IREE is not one monolithic lowering step. It is a sequence of target configuration, pattern annotation, tuning-spec materialization, ukernel selection, LLVM GPU lowering, device-library linking, and executable packaging. The `rocm` dialect is the small but important piece that lets those target-specific decisions stay visible and structured in MLIR.

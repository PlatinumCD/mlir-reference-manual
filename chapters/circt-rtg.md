# CIRCT `rtg` Dialect

The CIRCT `rtg` dialect is a Random Test Generation dialect. It describes tests, targets, reusable instruction sequences, random choices, virtual registers, memory resources, validation hooks, and ISA-assembly-oriented output. It is useful when a compiler or verification flow wants to generate many legal test programs from a compact randomized specification.

For a beginner, the core idea is that RTG is not hardware design IR like `hw`, and it is not a normal software function dialect. It is a test-template dialect. A test can contain random choices over sets and bags, sequence references, context switches, virtual registers, labels, and memory allocations. RTG passes then elaborate those choices into a concrete test that can be emitted as assembly-like text.

## When RTG Is Important

Use `rtg` when you are reading CIRCT flows for instruction-level random testing, target-specific test generation, or validation against a simulator. The dialect is designed to let users write reusable sequence families and target constraints, then let the compiler produce concrete tests.

The dialect is important when a test must adapt to target capabilities. A target may describe available memory blocks, contexts, supported modes, or other resources. A test can be matched with targets, elaborated with a random seed, and then lowered toward files containing ISA assembly.

You normally do not use RTG for synthesizable hardware. It is a verification and test-generation tool that can interact with ISA payload dialects, such as the local `rtgtest` dialect in this repository.

## Why It Is Needed

Random test generation has different requirements from normal IR lowering. A test template may say "pick one element from this set", "bias this choice by duplicating elements in a bag", "randomize this sequence once and embed the same result twice", or "allocate a virtual register from this allowed register class." Those operations are not ordinary runtime instructions. They are compile-time generation instructions.

RTG separates the random specification from the concrete result. Before elaboration, the IR may contain sets, bags, random scopes, sequence handles, virtual registers, memory blocks, and validation operations. After elaboration and lowering passes, the random choices, sequence references, and resource abstractions are replaced by concrete instructions, labels, registers, memory addresses, or emitted files.

## Type Inventory

The local RTG type definitions include:

- `!rtg.sequence<...>` is a handle to a sequence or sequence family. If it has remaining element types, those are arguments that still need substitution.
- `!rtg.randomized_sequence` is a handle to a sequence whose random constructs have already been resolved.
- `!rtg.set<T>` is an unordered set of values of type `T`.
- `!rtg.bag<T>` is a multiset of values of type `T`; duplicates bias random selection.
- `!rtg.dict<name: type, ...>` is a statically shaped dictionary used for target capabilities and test arguments.
- `!rtg.array<T>` is a dynamically sized array.
- `!rtg.map<K -> V>` is a map from keys to values.
- `!rtg.tuple<...>` is a tuple that may have zero or more fields.
- `!rtg.string` is a string value.
- `!rtg.isa.label` is an assembly label reference.
- `!rtg.isa.immediate<N>` is an ISA immediate of fixed bit width.
- `!rtg.isa.memory<N>` is a handle to an allocated memory region with address width `N`.
- `!rtg.isa.memory_block<N>` is a handle to a memory block from which memories can be allocated.

The dialect also uses type interfaces for target-specific context resources, registers, validation values, and ISA assembly printing.

## Attribute Inventory

The main RTG attributes are:

- `#rtg.default` refers to the default context resource of a type.
- `#rtg.any_context` refers to any single context of the requested type.
- `#rtg.set<...>` stores an unordered constant set.
- `#rtg.map<...>` stores a constant map.
- `#rtg.tuple<...>` stores a tuple constant.
- `#rtg.isa.immediate<...>` stores an ISA immediate value.
- `#rtg.virtual_register_config[...]` stores the allowed concrete registers for a virtual register.
- `#rtg.isa.label<...>` stores a label name.

The dialect also defines enum attributes used in operation syntax, including label visibility values `local`, `global`, and `external`, and segment kinds `data` and `text`.

## Operation Inventory

The local RTG dialect defines fifty-nine operations. The easiest way to learn them is by role.

### Test and Target Structure

`rtg.test` is the root of a randomized or directed test. It has a symbol name, a template name, a target dictionary type, and an optional target symbol.

`rtg.target` defines a test target and yields target capabilities as a dictionary. Capabilities can include resources such as memory blocks, contexts, or implementation-specific values.

`rtg.yield` terminates RTG regions such as `rtg.target` and `rtg.random_scope`.

`rtg.test.success` exits a test and reports success.

`rtg.test.failure` exits a test and reports failure with an error message.

`rtg.validate` validates the value in a payload resource. It can be resolved by a multi-run flow in which one run emits validation labels and another embeds externally collected values.

### Sequence Operations

`rtg.sequence` defines a reusable sequence or sequence family.

`rtg.get_sequence` creates an SSA sequence value from a sequence symbol.

`rtg.substitute_sequence` substitutes leading arguments of a sequence family and returns either a fully substituted sequence or a smaller sequence family.

`rtg.randomize_sequence` resolves random constructs in a fully substituted sequence and returns `!rtg.randomized_sequence`.

`rtg.embed_sequence` embeds a randomized sequence into a surrounding sequence or test.

`rtg.interleave_sequences` interleaves randomized sequences round-robin, optionally by batch size.

`rtg.on_context` places a sequence on a context, inserting context-switch instructions around it.

`rtg.context_switch` appears inside `rtg.target` and defines how to switch from one context resource to another using a sequence.

### Random and Constraint Operations

`rtg.random_scope` introduces an isolated randomization scope, optionally with a fixed seed.

`rtg.random_number_in_range` returns a uniformly random index in an inclusive range.

`rtg.constraint` enforces a boolean constraint. It should be used sparingly because backward constraints are more expensive than direct construction.

`rtg.constant` creates an SSA value from a typed attribute.

### Set and Bag Operations

`rtg.set_create` constructs a set from values.

`rtg.set_select_random` selects one element uniformly at random from a set.

`rtg.set_difference` subtracts one set from another.

`rtg.set_union` computes the union of one or more sets.

`rtg.set_size` returns the number of elements in a set.

`rtg.set_cartesian_product` computes the cartesian product of one or more sets and returns a set of tuples.

`rtg.set_convert_to_bag` converts a set to a bag with one copy of each element.

`rtg.bag_create` constructs a bag from elements and multiplicities.

`rtg.bag_select_random` selects a random element from a bag, where duplicates bias probability.

`rtg.bag_difference` subtracts multiplicities from a bag and can remove all matching elements with `inf`.

`rtg.bag_union` computes the union of bags.

`rtg.bag_unique_size` returns the number of unique elements in a bag.

`rtg.bag_convert_to_set` drops duplicates and returns a set.

### Array and Tuple Operations

`rtg.array_create` creates a dynamic array from values.

`rtg.array_extract` returns the element at an index.

`rtg.array_inject` returns a new array with one element replaced.

`rtg.array_size` returns the array size.

`rtg.array_append` returns a new array with one element appended.

`rtg.tuple_create` creates a tuple.

`rtg.tuple_extract` extracts one tuple element by index.

### String and Label Operations

`rtg.string_concat` concatenates strings.

`rtg.int_format` formats an index value as a string.

`rtg.immediate_format` formats an immediate as a string.

`rtg.register_format` formats a register as its assembly name.

`rtg.label_unique_decl` declares a unique label using a prefix.

`rtg.string_to_label` converts a string to a non-uniqued label.

`rtg.label` places a label in the instruction stream with local, global, or external visibility.

`rtg.comment` emits a comment in the instruction stream.

### ISA Helper Operations

`rtg.isa.int_to_immediate` converts an index to a fixed-width immediate.

`rtg.isa.concat_immediate` concatenates immediate values.

`rtg.isa.slice_immediate` extracts a bit slice from an immediate.

`rtg.virtual_reg` creates a virtual register constrained by `#rtg.virtual_register_config`.

`rtg.isa.register_to_index` converts a register to its class index.

`rtg.isa.index_to_register` converts an index to a register of a target-specific register type.

`rtg.isa.memory_block_declare` declares a target memory block inside `rtg.target`.

`rtg.isa.memory_alloc` allocates a memory region from a memory block.

`rtg.isa.memory_base_address` returns a memory's base address as an immediate.

`rtg.isa.memory_size` returns the size of a memory in bytes.

`rtg.isa.space` reserves a number of bytes in emitted assembly.

`rtg.isa.string_data` reserves a zero-terminated string.

`rtg.isa.segment` defines a `data` or `text` segment.

## Transformations and Conversions

The RTG transform directory defines thirteen passes.

`rtg-elaborate` interprets most randomization operations. After it runs, tests should no longer contain random constructs. It accepts a seed and can delete tests that do not match any target.

`rtg-inline-sequences` inlines sequences into tests and removes `rtg.sequence` operations. It also materializes `rtg.interleave_sequences`.

`rtg-linear-scan-register-allocation` assigns virtual registers created by `rtg.virtual_reg`, using a simple linear-scan allocator after elaboration.

`rtg-lower-unique-labels` converts `rtg.label_unique_decl` into concrete label declarations by choosing unique strings.

`rtg-unique-validate` assigns unique IDs to `rtg.validate` operations without one.

`rtg-memory-allocation` lowers memory allocation operations to immediates or labels by computing offsets within memory blocks.

`rtg-embed-validation-values` replaces `rtg.validate` operations with externally supplied values, typically collected from a simulator run.

`rtg-lower-validate-to-labels` lowers validation operations to special intrinsic labels that a target simulator can use to print values.

`rtg-print-test-names` writes the generated test names and original template names to a CSV-style output.

`rtg-insert-test-to-file-mapping` inserts `emit` dialect operations to group tests into output files.

`rtg-simple-test-inliner` inlines test contents into `emit.file` operations. It is described as a simple helper rather than a production-quality inliner.

`rtg-emit-isa-assembly` emits instruction streams in an assembler-readable format from `emit.file` operations. It can emit binary representations for unsupported instruction names.

`rtg-materialize-constraints` materializes implicit constraints.

## What It Implies

When you see `rtg.set_select_random`, `rtg.bag_select_random`, or `rtg.random_number_in_range`, the IR still contains compile-time random choices. Those operations are not instructions in the final test. They are generation steps that `rtg-elaborate` should resolve.

When you see `rtg.sequence`, think reusable test fragment. When you see `rtg.randomize_sequence`, the compiler is choosing the concrete contents of that fragment. When you see `rtg.embed_sequence`, it is placing the chosen fragment into a larger test.

When you see `rtg.virtual_reg`, the test still has abstract registers. Register allocation will choose concrete target registers later.

When you see `rtg.validate`, the flow may split into validation and final generation runs. One run can emit labels for a simulator to report values; another can embed those values back into the test.

## How To Read RTG IR

Start with `rtg.target` operations. They tell you which resources and capabilities exist.

Then read `rtg.test` operations. The test arguments correspond to entries in the target dictionary. Inside the body, look for sequence operations, random selections, virtual registers, memory allocations, and validation.

Next, inspect `rtg.sequence` operations. A test often gets a sequence with `rtg.get_sequence`, substitutes arguments, randomizes it, and embeds it.

Finally, check how far the lowering has progressed. If random operations remain, the test is still a template. If sequences are gone and labels/registers/memories have been lowered, it is much closer to final assembly emission.

## Minimal Example

This sketch shows the typical sequence flow:

```mlir
rtg.sequence @one_instr() {
  %r = rtg.virtual_reg [#rtgtest.t0, #rtgtest.t1]
  %imm = rtg.constant #rtg.isa.immediate<12, 7>
      : !rtg.isa.immediate<12>
  "rtgtest.addi"(%r, %r, %imm)
      : (!rtgtest.ireg, !rtgtest.ireg, !rtg.isa.immediate<12>) -> ()
}

rtg.test @test_sequence() {
  %seq = rtg.get_sequence @one_instr : !rtg.sequence
  %chosen = rtg.randomize_sequence %seq
  rtg.embed_sequence %chosen
}
```

Before elaboration, the test contains a reusable sequence and a virtual register. After RTG lowering, the sequence can be inlined, random choices can be fixed, and the virtual register can be allocated to a concrete register.

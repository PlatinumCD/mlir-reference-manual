# IREE `util` Dialect

The IREE `util` dialect is the common support dialect used across IREE's compiler pipeline. It provides shared types, attributes, structural operations, globals, lists, byte buffers, compiler hints, and runtime-facing utility operations that other IREE dialects build on.

For a beginner, the easiest way to understand `util` is to think of it as IREE's bridge between generic MLIR and IREE's runtime model. A frontend may start with `func.func`, tensors, memrefs, constants, and ordinary MLIR control flow. IREE eventually needs module initialization, global state, callable functions with ABI details, byte buffers, status checks, resource constants, and enough metadata to drive codegen and runtime serialization. The `util` dialect carries those concepts.

This dialect is not a high-level math dialect. It does not describe neural network layers, tensor algebra, GPU kernels, or executable binaries by itself. Instead, it supplies the common infrastructure that lets higher-level and lower-level IREE dialects cooperate.

## Why This Dialect Exists

MLIR has many generic dialects, but IREE needs a portable runtime-facing layer. For example, MLIR `memref` is a compiler abstraction for memory references; IREE's runtime often wants a byte buffer with explicit offsets and lengths. MLIR `func.func` is a generic function; IREE needs a function operation that can carry tied operand information, ABI reflection, inlining policy, and runtime interop metadata. MLIR attributes can inline large dense constants; IREE needs resource-backed and serializable forms that do not explode compiler memory use.

The `util` dialect exists to express those cross-cutting ideas in one place. It is deliberately broad. It has function-like operations, global variables, list and buffer operations, type conversion helpers, compiler assumptions, and debug-only operations. Many IREE passes depend on it because it provides a stable shared vocabulary.

The dialect also gives IREE a place to encode analyses and decisions. `util.assume.int` stores integer range and divisibility assumptions. `util.numeric.optional_narrow` records that a numeric narrowing is valid but optional. `util.hoistable_conversion` marks conversions that can be moved into globals. `util.optimization_barrier` blocks otherwise tempting rewrites when preserving a value boundary matters.

## When To Use It

You will usually read `util` IR when you are looking at IREE before or after input conversion, global optimization, executable packaging, or VM/runtime lowering. It appears in modules that need initialization order, global constants, imported resources, serialized data, or runtime callable functions.

Use `util` when you need to understand how IREE represents host-side program structure. A `util.initializer` describes module initialization. A `util.global` describes state or constant storage. A `util.func` describes an IREE function. `util.call` and `util.return` describe calls and returns in that structural world.

Use the buffer operations when the compiler has moved from typed memory references to byte-addressed runtime storage. A `!util.buffer` is a reference-counted byte buffer. Operations such as `util.buffer.load`, `util.buffer.store`, `util.buffer.copy`, and `util.buffer.fill` operate in byte offsets and lengths.

Avoid reading `util` as though it were one cohesive source language. It is more like a toolbox. The operations are grouped by purpose, and many only make sense in a particular compiler phase.

## Core Types And Attributes

The major `util` types are:

| Type | Meaning |
| --- | --- |
| `!util.buffer` | A reference-counted byte buffer modeled as pointer, offset, and length. |
| `!util.list<T>` | A dense typed list container. |
| `!util.ptr<T>` | A typed indirect reference to a runtime-addressable value. |
| `!util.object` | A placeholder for an unspecified runtime object. |
| `!util.unused` | A placeholder type used when a verifier needs a type even though the value is unused. |
| `!util.variant` | A runtime variant placeholder, often printed conceptually as `?`. |

The common `util` attributes are:

| Attribute | Meaning |
| --- | --- |
| `#util.int.assumption` | Integer assumptions such as unsigned minimum, maximum, or divisibility. |
| `#util.byte_pattern` | Serializable repeated byte pattern storage. |
| `#util.byte_range` | Offset and length in bytes. |
| `#util.composite` | A concatenation of serializable attributes into one byte sequence. |
| `#util.inline.never` | Inlining policy that disables inlining. |
| `#util.inline.always` | Inlining policy that requests inlining when legal. |
| `#util.null` | A typed null reference attribute. |
| `#util.preprocessing_pipeline` | Textual preprocessing pass pipeline attached to a function-like operation. |
| `#util.uninitialized` | Storage attribute whose contents may be undefined at runtime. |

These types and attributes explain much of the dialect's design. The dialect is concerned with object references, buffers, serialization, globals, initialization, and compiler/runtime metadata.

## Operation Inventory

### Type And Value Utilities

| Operation | What It Means |
| --- | --- |
| `util.null` | Produces a null value of a util reference-like type. |
| `util.cast` | Casts one util type to another, similar in spirit to a static or dynamic cast. |
| `util.cmp.eq` | Compares two values for equality. |
| `util.cmp.ne` | Compares two values for inequality. |
| `util.numeric.optional_narrow` | Records that a numeric narrowing is valid, but optional. Later passes may keep or remove the narrowing based on target policy. |
| `util.range.min` | Computes the minimum of range-like integer values. |
| `util.range.max` | Computes the maximum of range-like integer values. |
| `util.range.extents` | Computes the combined minimum and maximum extent of a set of ranges. |
| `util.align` | Aligns an offset up to a required power-of-two alignment. |
| `util.sizeof` | Returns the size in bytes of a datatype. |
| `util.switch` | A primitive value selection operation, useful before or during structural rewrites. |

### Compiler Hints

| Operation | What It Means |
| --- | --- |
| `util.assume.int` | Binds integer assumptions, such as range or divisibility, to values. |
| `util.hoistable_conversion` | Defines a conversion between inputs and outputs that may be hoisted when the types and users allow it. |
| `util.optimization_barrier` | Prevents optimizations from crossing a value boundary. |
| `util.unfoldable_constant` | Represents a constant that should not be folded by the compiler. |

### Structural Operations

| Operation | What It Means |
| --- | --- |
| `util.initializer` | A global initialization function. It is used to initialize module state and has ordering constraints with globals. |
| `util.func` | IREE's function operation. It can carry IREE ABI and policy information beyond generic `func.func`. |
| `util.call` | Calls a `util.func` or imported callable symbol. It can carry tied operand metadata. |
| `util.return` | Returns from a `util.func` or `util.initializer`. |
| `util.unreachable` | Terminator that marks code as unreachable at runtime. |
| `util.scf.unreachable` | Non-terminator unreachable marker for use inside SCF regions. |

### Globals

| Operation | What It Means |
| --- | --- |
| `util.global` | Declares a stateful global variable or constant. |
| `util.global.address` | Produces an indirect pointer-like reference to a global. |
| `util.global.load` | Loads a value directly from a named global. |
| `util.global.load.indirect` | Loads through an indirect global reference. |
| `util.global.store` | Stores a value directly into a named global. |
| `util.global.store.indirect` | Stores through an indirect global reference. |

### Lists

| Operation | What It Means |
| --- | --- |
| `util.list.create` | Creates a new empty list. |
| `util.list.construct` | Constructs a list with initial values. |
| `util.list.size` | Returns the list size in elements. |
| `util.list.resize` | Resizes a list to a new element count. |
| `util.list.get` | Reads an element from a list. |
| `util.list.set` | Writes an element into a list. |

### Buffers

| Operation | What It Means |
| --- | --- |
| `util.buffer.constant` | Creates a constant host-side byte buffer. |
| `util.buffer.alloc` | Allocates a buffer with undefined contents. |
| `util.buffer.dealloc` | Deallocates a buffer. |
| `util.buffer.slice` | Clones a subregion of a buffer. |
| `util.buffer.subspan` | Produces a reference to a subrange of a buffer. |
| `util.buffer.size` | Returns the total buffer storage size in bytes. |
| `util.buffer.storage` | Returns the underlying storage range for a buffer. |
| `util.buffer.copy` | Copies a byte range between buffers. |
| `util.buffer.compare` | Compares byte ranges from two buffers. |
| `util.buffer.fill` | Fills a byte range of a buffer with a value. |
| `util.buffer.load` | Loads a typed value from a byte buffer using buffer size, byte offset, and byte length. |
| `util.buffer.store` | Stores a typed value into a byte buffer using buffer size, byte offset, and byte length. |
| `util.buffer.hash` | Computes a hash over a byte range of a buffer. |

### Strings And Status

| Operation | What It Means |
| --- | --- |
| `util.string.format` | Formats a string from a template and arguments. |
| `util.string.itoa` | Converts an integer to its decimal string representation. |
| `util.status.check_ok` | Raises a global failure if a status value is not OK. |

## Transformations And Passes

| Pass | Role |
| --- | --- |
| `iree-util-apply-patterns` | Applies risky or IREE-specific canonicalization patterns. |
| `iree-util-attribute-call-graph` | Propagates call-related attributes from callees to call sites. |
| `iree-util-combine-initializers` | Combines global initializers into one initializer. |
| `iree-util-drop-compiler-hints` | Deletes compiler-only operations that have no runtime equivalent. Optionally keeps integer assumptions. |
| `iree-util-dump-module` | Writes the module to a textual or bytecode file path. |
| `iree-util-fixed-point-iterator` | Runs a nested pass pipeline until it reaches a fixed point. |
| `iree-util-ipo` | Performs basic inter-procedural optimization. |
| `iree-util-lift-cfg-to-scf` | Converts reducible unstructured CFG in `util.initializer` and `util.func` into structured SCF operations. |
| `iree-util-link-modules` | Links external function declarations from explicitly provided modules or library search paths. |
| `iree-util-optimize-int-arithmetic` | Optimizes integer arithmetic using dataflow analysis and rewrite patterns. |
| `iree-util-propagate-subranges` | Propagates resource subranges across the program. |
| `iree-util-strip-and-splat-constants` | Replaces constant globals with splat forms to reduce stored data. |
| `iree-util-strip-debug-ops` | Removes debug-only operations. |
| `iree-util-verify-initialization-order` | Verifies initializer and global mutation ordering constraints. |
| `iree-util-verify-structured-control-flow` | Ensures function-like operations contain no unstructured branch operations. |
| `iree-util-fold-globals` | Folds duplicate globals and propagates constants. |
| `iree-util-fuse-globals` | Fuses correlated globals together. |
| `iree-util-hoist-into-globals` | Hoists eligible constant expressions into globals, subject to a size-increase threshold. |
| `iree-util-simplify-global-accesses` | Hoists loads and sinks stores to shrink data dependency regions around variables. |
| `iree-util-import-resources` | Converts large inline dense attributes to resource-backed attributes that IREE can manage efficiently. |
| `iree-util-annotate-op-ordinals` | Adds globally unique IDs to operations for debugging. |
| `iree-util-test-conversion` | Tests util dialect conversion patterns. |
| `iree-util-test-float-range-analysis` | Tests floating-point range analysis. |
| `iree-util-test-integer-divisibility-analysis` | Tests integer divisibility analysis. |

## Conversion Paths

The util conversion helpers are as important as the operations themselves.

`populateFuncToUtilPatterns` converts top-level `func.func`, `func.call`, and `func.return` into `util.func`, `util.call`, and `util.return`. During this conversion, IREE can preserve selected attributes such as reflection, stream affinity, VM interop metadata, versioning, side-effect information, and argument/result attributes. Generic `noinline` is translated into the util inlining policy `#util.inline.never`.

`populateMemRefToUtilPatterns` converts rank-0 and rank-1 identity-layout memrefs into `!util.buffer` by default, or into a caller-specified buffer type for multi-stage lowerings. `memref.global` becomes `util.global` plus a `util.initializer` that stores a `util.buffer.constant`. `memref.get_global` becomes `util.global.load`. `memref.alloca` becomes `util.buffer.alloc`. `memref.load` and `memref.store` become `util.buffer.load` and `util.buffer.store` with byte offsets computed from element size and indices.

`populateUtilConversionPatterns` updates util operations and nested util types under a type converter. It knows how to convert `!util.ptr<T>` by converting the target type, and `!util.list<T>` by converting the element type. It also provides generic conversion patterns for operations such as `util.assume.int`, `util.optimization_barrier`, `util.list.create`, `util.list.get`, and `util.list.set`.

`populateGenericStructuralConversionPatterns` rewrites structural operations in `util`, `func`, `cf`, `arith`, and `scf` when type conversion changes their operand, result, or region types. This is why util is common in cross-dialect conversion: it helps keep structure valid while types are changing.

## How To Read Util IR

Start with the module-level structure. Look for `util.global` and `util.initializer` first. They tell you what persistent state exists and how that state is initialized. If initialization-order verification fails, the problem is usually an initializer touching a global too early, writing an immutable global more than once, or mutating a global that should only receive its initial value.

Next, read callable structure. `util.func` and `util.call` behave like ordinary functions and calls, but they often carry ABI, inlining, reflection, tied operand, or VM interop metadata. Those attributes are often more important than the body when you are debugging import or runtime boundary behavior.

Then inspect globals and resources. `util.global.load`, `util.global.store`, `util.global.address`, and their indirect forms are the access points. If constants are large, `iree-util-import-resources`, `iree-util-hoist-into-globals`, `iree-util-fold-globals`, and `iree-util-fuse-globals` may change where data lives without changing the visible program result.

Finally, read buffers as byte-addressed storage. `util.buffer.load` and `util.buffer.store` are not high-level tensor operations. They are byte-buffer operations with explicit sizes and offsets. If you see them after memref conversion, the original typed memory abstraction has already been lowered.

## What It Implies

Seeing `util` usually means the compiler is crossing an infrastructure boundary. It may be importing source IR into IREE, preserving ABI metadata, lowering generic memory into runtime buffers, moving constants into globals, or preparing module state for later serialization.

The implication for beginners is that `util` is less about one execution model and more about IREE's shared contracts. The structural ops define how IREE modules call and initialize. The global ops define how state is stored and accessed. The buffer ops define how byte storage is represented. The hint ops tell optimizers what they may assume or must avoid. The passes keep those contracts consistent while other dialects do the domain-specific work.


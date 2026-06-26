# CIRCT comb Dialect

The `comb` dialect is CIRCT's generic dialect for combinational hardware logic. It is meant to describe pure logic between state elements, module ports, memories, or other hardware structures. It is usually seen together with CIRCT's `hw`, `sv`, `seq`, `synth`, `datapath`, and verification dialects.

For a beginner, the key idea is that `comb` is not a hardware description language by itself. It is a compiler IR for logic expressions. SystemVerilog, FIRRTL, Verilog import, synthesis, and verification flows can all use it because it gives them a common set of integer and bit-vector operations without committing to one source language's syntax or corner cases.

`comb` is important whenever a circuit needs arithmetic, bitwise logic, compares, bit slicing, concatenation, muxing, or truth-table style logic. It is also important because many later CIRCT passes understand it well: they can fold constants, simplify mux trees, map logic to SMT, map logic to AIG-style synthesis operations, or lower logic to `arith` and then LLVM.

## When To Use It

Use `comb` when the value being described is a combinational function of other values. If there is no clock, register update, memory side effect, handshake protocol, or module definition involved, `comb` is often the right level. A module might be represented by `hw.module`, a register by `seq`, an emitted SystemVerilog statement by `sv`, and the expression feeding those constructs by `comb`.

You normally use `comb` after frontend language details have been normalized. For example, source languages may have implicit width extension rules, signedness rules, or special unknown-value behavior. `comb` tries to make these details explicit. Inputs generally have the same width where the operation requires it, signedness is carried by operation choice or comparison predicate rather than by distinct signed integer types, and optional `bin` or `twoState` flags record when an operation can assume two-valued logic.

The dialect is needed because hardware compilers need an IR that is close enough to gates and bit vectors to optimize, but not so low level that every design is immediately flattened into individual gates. `comb.add` can still be an adder. `comb.mux` can still be a mux. `comb.truth_table` can still be a compact Boolean function before it is lowered.

## Operation Inventory

The `comb` dialect defines these operations:

```text
comb.add
comb.mul
comb.divs
comb.mods
comb.divu
comb.modu
comb.shl
comb.shru
comb.shrs
comb.sub
comb.and
comb.or
comb.xor
comb.icmp
comb.parity
comb.extract
comb.concat
comb.replicate
comb.mux
comb.truth_table
comb.reverse
```

## Data Model

`comb` mostly operates on hardware integer types. Its operations are pure: a `comb.add` or `comb.and` computes a value and has no memory or scheduling side effects. This makes the dialect friendly to canonicalization and rewriting.

A central rule is that many operations require uniform input and result widths. For example, `comb.add`, `comb.mul`, `comb.and`, `comb.or`, and `comb.xor` are variadic, but their operands and result have the same type. Binary operations such as `comb.sub`, `comb.divs`, `comb.divu`, `comb.mods`, `comb.modu`, `comb.shl`, `comb.shru`, and `comb.shrs` also work on hardware integer values. CIRCT's rationale explicitly avoids implicit operand extension because implicit widths make compiler optimization hard.

The dialect also supports zero-width integers in well-defined cases. This matters in generated hardware, where parameterization can produce zero-width slices or concatenations. Operations that are not meaningful for zero-width values, such as division by zero-like cases, are treated carefully.

The optional `twoState` flag appears in assembly as `bin` on many operations. It means the operation can use ordinary binary 0/1 logic instead of preserving the extended unknown-value behavior that languages such as Verilog or VHDL may require. The `comb-assume-two-valued` pass uses this flag heavily.

## Arithmetic And Logical Ops

`comb.add`, `comb.mul`, `comb.sub`, `comb.divs`, `comb.divu`, `comb.mods`, and `comb.modu` describe integer arithmetic. Signed and unsigned behavior is not a property of the integer type; it is encoded in the operation. That is why there are separate signed and unsigned division and modulus operations. `comb.divs` and `comb.mods` are signed; `comb.divu` and `comb.modu` are unsigned.

`comb.and`, `comb.or`, and `comb.xor` are variadic bitwise logical operations. They are commutative and have extensive folding support. `comb.xor` is also how the dialect represents bitwise complement in canonical form: an xor with an all-ones constant is treated as a not pattern. The dialect deliberately does not add a separate complement operation because fewer equivalent forms make optimization easier.

`comb.shl`, `comb.shru`, and `comb.shrs` are shifts. `comb.shl` shifts left, `comb.shru` is logical right shift, and `comb.shrs` is arithmetic right shift. Like the divide and modulus split, the signed behavior is captured by which operation is used.

These operations imply hardware structure, but they do not force a final gate implementation. Later flows may preserve an adder, lower arithmetic into carry-save datapath nodes, translate to SMT bit-vector operations, or decompose logic into AIG-style synthesis operations.

## Comparison, Unary, And Bit Manipulation Ops

`comb.icmp` compares two integer values and returns an `i1`. It has predicates `eq`, `ne`, `slt`, `sle`, `sgt`, `sge`, `ult`, `ule`, `ugt`, `uge`, `ceq`, `cne`, `weq`, and `wne`. The ordinary signed and unsigned predicates cover normal integer comparison. The `ceq` and `cne` predicates correspond to SystemVerilog case equality and inequality, while `weq` and `wne` correspond to wildcard equality and inequality. Under a two-valued assumption, the specialized equality forms can be simplified to ordinary equality or inequality.

`comb.parity` computes the parity of the bits in an integer input and returns one bit. `comb.reverse` reverses bit order, turning the least significant bit into the most significant bit and vice versa. These operations are small, but important for code generation because not every target dialect has direct equivalents. CIRCT's Comb-to-LLVM patterns specifically handle parity and reverse because most other Comb operations are handled through the Comb-to-Arith path.

`comb.extract` selects a range of bits from an integer input. Its `lowBit` attribute names the lowest included bit, and the result type determines the width. `comb.concat` concatenates operands into a wider integer. Operands are written most-significant to least-significant in assembly, so a concat should be read left to right as the final bit string. `comb.replicate` repeats one operand enough times to create the result width.

These bit manipulation operations are why `comb` does not need separate zero extension or sign extension operations. Zero extension can be represented as `comb.concat` with high zero bits. Sign extension can be represented with `comb.extract`, `comb.replicate`, and `comb.concat`.

## Selection And Truth Tables

`comb.mux` selects between a true value and a false value based on an `i1` condition. Its true value, false value, and result have the same type, but the selected type can be any value type accepted by the operation. The mux is intentionally first class because single-bit selection is extremely common in hardware and many optimizations understand mux trees.

`comb.truth_table` represents a fully elaborated Boolean truth table. It takes variadic `i1` inputs and a dense Boolean lookup table, and returns one bit. Inputs are interpreted from most significant to least significant, and the lookup table has `2^n` entries for `n` inputs. It is useful when a frontend or synthesis step wants to keep a Boolean function compact before deciding whether to lower it into muxes or simpler logic.

The dialect does not provide a first-class multibit mux operation. CIRCT's rationale recommends using `hw.array_create` and `hw.array_get` for multibit indexed selection. `comb.mux` covers the very common single-condition selection case.

## Local Transformations

`lower-comb` lowers operations that are not directly supported by downstream emission. In the current Comb transform set, its main job is lowering `comb.truth_table` into a tree of `comb.mux` operations with `hw.constant` true and false values.

`comb-simplify-tt` simplifies truth tables that are constant or depend on only one effective input. A constant table becomes `hw.constant`. A table that is just identity becomes the selected input. A table that is one-input negation remains a smaller `comb.truth_table`, which is useful for later LUT-oriented mapping.

`comb-assume-two-valued` applies the assumption that all integer bits are only 0 or 1. It marks operations with the two-state flag and rewrites specialized `comb.icmp` predicates such as `ceq`, `cne`, `weq`, and `wne` to ordinary `eq` or `ne` where that is valid.

`comb-int-range-narrowing` uses MLIR integer range analysis to reduce bit widths for selected arithmetic operations. It currently narrows binary `comb.add`, `comb.mul`, and `comb.sub` when leading bits are provably unnecessary, replacing them with extracts, a narrower operation, and a concat back to the original width.

`comb-overflow-annotating` also uses integer range analysis. It annotates selected `comb.add` and `comb.mul` operations with `comb.nuw` when unsigned overflow cannot happen. This gives later passes a stronger fact about the operation.

`comb-balance-mux` optimizes mux chains. It can fold mux chains based on comparisons, rebalance priority mux chains from linear depth to tree depth, and convert certain ors of independent muxes into mux chains. The pass has a `mux-chain-threshold` option that decides how large a chain must be before rebalancing is worthwhile.

The dialect also has extensive canonicalization and folding in `CombFolds.cpp`. These folds flatten associative operations, combine constants, simplify extract/concat/replicate patterns, simplify muxes, canonicalize comparisons, and recognize common hardware idioms. Beginners do not need every rule memorized; the important point is that `comb` is intentionally shaped to make these rewrites reliable.

## Conversions Into Comb

`map-arith-to-comb` maps integer `arith` operations into `comb` and `hw.constant`. Its helper `populateArithToCombPatterns` covers `arith.addi`, `subi`, `muli`, signed and unsigned divisions and remainders, bitwise and/or/xor, shifts, select, integer comparisons, integer min/max, integer extension, truncation, and integer constants. Min and max are lowered as `comb.icmp` plus `comb.mux`; zero extension becomes `comb.concat` with zeros; truncation becomes `comb.extract`; signed extension is built from Comb helper logic.

`convert-synth-to-comb` lowers selected `synth` operations back into `comb` and `hw`. This is useful for checking post-synthesis results or moving a synthesized representation back to a more general combinational form. Its pattern population includes choice, inverter, mux, dot, majority, one-hot, and gamble-style synthesis operations.

Other CIRCT flows can also produce `comb` as part of hardware lowering. For example, HWArith lowering and Verilog import can create Comb expressions when they normalize arithmetic or source expressions into CIRCT's common hardware IR.

## Conversions Out Of Comb

`convert-comb-to-arith` lowers many `comb` operations into MLIR `arith` operations and `hw.constant`. Its `populateCombToArithConversionPatterns` handles replicate, constants, comparisons, extract, concat, shifts, sub, signed and unsigned division and remainder, mux/select, and variadic add, multiply, and/or/xor. It intentionally leaves `comb.parity` and `comb.reverse` legal because Arith does not have direct equivalents for them.

`populateCombToLLVMConversionPatterns` handles the Comb operations that are not covered well by Comb-to-Arith followed by Arith-to-LLVM. In this source tree, it adds direct LLVM lowering patterns for `comb.parity` and `comb.reverse`.

`convert-comb-to-smt` lowers Comb logic to SMT bit-vector operations for formal reasoning. Its `populateCombToSMTConversionPatterns` covers replicate, comparisons, extract, mux, sub, parity, reverse, shifts, signed and unsigned division and remainder, concat, and the variadic arithmetic and logical operations. The source notes that `comb.truth_table` is not supported by this conversion, so it should be simplified or lowered first when an SMT path needs it.

`convert-comb-to-synth` lowers Comb operations to the `synth` dialect, usually toward AIG-style logic. It can lower bitwise ops, muxes, arithmetic, comparisons, shifts, add, multiply, signed and unsigned div/mod, and can force AIG-only forms. It keeps data movement operations such as `comb.extract`, `comb.concat`, and `comb.replicate` legal.

`convert-comb-to-datapath` lowers arithmetic Comb operations into `datapath` operations, primarily for synthesis flows that want redundant number representations such as carry-save. It converts multi-input `comb.add`, `comb.mul`, and uses `convertSubToAdd` for subtraction. `convert-datapath-to-comb` performs the reverse direction by replacing datapath compressors and partial products with Comb and HW logic, with options for add-based compressor lowering, Booth lowering, and timing-aware synthesis.

## How To Read Comb IR

When reading `comb`, first group operations by role. Arithmetic and bitwise operations tell you the expression being computed. `comb.extract`, `comb.concat`, `comb.replicate`, and `comb.reverse` tell you how bits are rearranged. `comb.icmp` and `comb.mux` usually reveal control-like selection encoded as dataflow. `comb.truth_table` means a Boolean function has not yet been expanded.

Next, check widths. Comb operations usually expect explicit, compatible widths, so a surrounding concat, extract, or replicate often explains how a frontend handled source-language width rules. If you expected a sign extension operation, look for the canonical Comb spelling using extract, replicate, and concat.

Finally, look for the target flow. If the next pass is `convert-comb-to-smt`, the same IR is becoming a formal bit-vector problem. If the next pass is `convert-comb-to-synth` or `convert-comb-to-datapath`, it is becoming synthesis-oriented structure. If the path goes through `convert-comb-to-arith` and LLVM, it is becoming software-like integer code. The same dialect serves all of these paths because it captures pure combinational meaning without tying it to one backend.

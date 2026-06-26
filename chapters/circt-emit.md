# CIRCT emit Dialect

The CIRCT `emit` dialect models the structure of files emitted by CIRCT. It is not a hardware behavior dialect. Instead, it tells the export layer how to organize output files, file lists, opaque text, and fragments that should be inserted before other emitted operations.

For a beginner, the key idea is that compiler IR sometimes needs to describe output packaging, not just computation. A hardware compiler may need to emit SystemVerilog modules, bind files, generated headers, file lists, helper fragments, or other collateral. The `emit` dialect provides a small IR vocabulary for that output structure.

The dialect rationale describes `emit` as controlling the structure and formatting of files emitted from CIRCT. In practice it is closely tied to `ExportVerilog`, but its model is general enough to describe generic collateral files too.

## Operation Inventory

The `emit` dialect defines these operations:

```text
emit.file
emit.file_list
emit.fragment
emit.ref
emit.verbatim
```

## Core Model

`emit` has no custom types. Its power comes from symbol-aware operations and regions. `emit.file` and `emit.fragment` are containers. `emit.ref` and `emit.file_list` use symbol references. Other dialects can attach an `emit.fragments` attribute to request that named fragments be emitted before an operation.

This makes `emit` a layout layer over the rest of the IR. A hardware module still belongs to `hw`, a SystemVerilog construct still belongs to `sv`, and logic still belongs to `comb` or another behavior dialect. `emit` answers a different question: where should this text or referenced operation appear in the final output?

## File And File List Ops

`emit.file` represents the contents of one emitted file. It has a file name attribute, an optional symbol name, and a single body region. Operations nested in the body are emitted into that file. The optional symbol lets other operations refer to the file.

A typical use is to create an additional SystemVerilog or collateral file alongside normal module output. The body might contain `emit.verbatim` text or `emit.ref` references to existing targetable operations.

`emit.file_list` represents an emitted file list. It has a file name, an array of flat symbol references to files, and an optional symbol name. The emitted file list enumerates the paths of referenced files. This is useful for build systems and simulators that consume a list of generated sources rather than being handed every source path separately.

Together, `emit.file` and `emit.file_list` model output artifacts: files and lists of files.

## Fragments

`emit.fragment` is a named, module-level container for reusable output. A fragment can be referenced by an `emit.fragments` attribute on another operation. During emission, referenced fragments are inserted before the operation that requested them.

The important behavior is different in single-file and split-file modes. In single-file mode, a fragment is emitted at most once. In split-file emission, a fragment can appear before referencing operations in different files, but still at most once per file. This lets lowering passes create shared helper text without duplicating it unnecessarily.

Several CIRCT lowering paths use fragments for support code. For example, sequence-to-SystemVerilog and simulation-related lowering can attach fragment references to modules when generated output needs helper declarations or shared runtime snippets.

## Verbatim Text And References

`emit.verbatim` emits opaque text inline inside an `emit.file`. It is the simplest way to put exact text into an output file. It also supports symbol reference substitutions using placeholder syntax, so generated text can still refer to IR symbols rather than hard-coding every final name immediately.

`emit.ref` emits a referenced operation inline into the surrounding `emit.file`. It targets an operation by symbol. The set of operations that can be referenced and the exact emission rules are defined by `ExportVerilog`, not by the `emit` dialect itself. This is an important separation: `emit.ref` says "emit that thing here"; the exporter decides how that target is printed.

## Transformations And Conversions

The named `emit` dialect pass is:

```text
strip-emit
```

`strip-emit` removes top-level Emit dialect operations from a module. It also removes the `emit.fragments` attribute from non-Emit operations. This is useful in flows that need to discard output-packaging metadata before continuing with a pipeline that does not understand or want emission structure.

`export-verilog` and `export-split-verilog` are not Emit dialect passes, but they are the main consumers of Emit IR. `ExportVerilog` emits `emit.file` contents, resolves `emit.ref`, emits `emit.verbatim`, handles `emit.fragment`, and emits file lists from `emit.file_list`. The exporter reports an error if an operation references a fragment that cannot be found.

The reducer support includes the `emit-op-eraser` reduction pattern. This is used by `circt-reduce` to erase Emit dialect operations when doing so does not break inner-symbol references. It is not a normal optimization pass; it is a debugging reduction pattern for shrinking test cases.

## When To Use It

Use `emit` when a pass needs to create or organize output files. Good examples are collateral files, extra source files, generated file lists, or reusable snippets that should be inserted around emitted modules. It is also appropriate when a lowering pass needs to preserve a relationship between output text and existing IR symbols.

Do not use `emit` to model hardware semantics. A module, register, wire, or expression belongs in the hardware and behavior dialects. `emit` should only describe output structure or opaque output text.

The dialect is needed because output is more structured than a single stream of text. Real hardware flows often require split files, helper files, file lists, and repeated fragments with careful de-duplication. Encoding that structure in IR lets passes build and transform the output plan before the final exporter prints anything.

## How To Read Emit IR

Start with `emit.file` operations. Each one is an output artifact. Its file name tells you where the contents should go, and its body tells you what will be printed there.

Next, look for `emit.file_list`. It tells you how emitted files are grouped for downstream tools.

Then inspect `emit.fragment` operations and `emit.fragments` attributes. They reveal reusable pieces of output that will be inserted before selected operations.

Finally, distinguish literal text from referenced IR. `emit.verbatim` is direct text. `emit.ref` asks the exporter to print another symbol in the current file. That distinction is what lets CIRCT mix generated collateral with normal hardware emission without collapsing everything into strings too early.

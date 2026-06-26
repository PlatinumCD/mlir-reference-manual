# CIRCT DC Dialect

The CIRCT `dc` dialect is the Dynamic Control dialect. It represents latency-insensitive control flow using tokens and FIFO-like channel semantics, while keeping data values separate unless data is needed to steer control.

For a beginner, the useful mental model is this: `dc` is a control-flow network dialect. A `!dc.token` means "a control event can happen." A `!dc.value<T>` means "a value of type `T` travels with a control token." Operations such as join, fork, branch, select, merge, source, sink, buffer, pack, and unpack build a dynamic control network.

The local checkout defines 12 `dc` operations, two custom types, and 5 DC-related transformation or conversion passes.

## When DC Is Important

`dc` is important when CIRCT is lowering or optimizing dataflow-style hardware where control and data should be explicit but separate. It is closely related to the Handshake dialect, but it is more strictly focused on control. The rationale describes DC as modeling independent, unsynchronized processes that communicate through FIFO channels.

Use this dialect when you need to answer questions like:

- Which parts of a dataflow program are pure control?
- Where are data values wrapped in a latency-insensitive channel?
- Where does a condition choose between control paths?
- Where does the IR synchronize multiple tokens?
- Are fork and sink operations implicit or explicitly materialized?
- Has Handshake IR been lowered to DC?
- Has DC been lowered to ESI, HW, Comb, or Seq?

You usually inspect DC when studying Handshake lowering, latency-insensitive control networks, or the bridge from high-level dataflow IR into ESI-backed hardware channels.

## Why It Is Needed

The Handshake dialect gives control semantics to all SSA values. That is useful, but it also means data and control are tightly coupled everywhere. DC deliberately separates them. A token can express control without carrying payload data. A `!dc.value<T>` carries data only when needed.

This separation makes it easier to optimize the control side independently from the data side. Joins, branches, selects, merges, buffers, forks, and sinks can be simplified as control operations. Data-side arithmetic can remain in other dialects such as `arith`, `comb`, or `hw`, then be packed into or unpacked from `!dc.value` channels where control needs it.

The implication is that DC is not just a hardware-ready netlist. The rationale explicitly says DC does not imply one implementation. A DC value could be implemented by ready/valid hardware, FIFO interfaces, software queues, RPC, or another streaming protocol. In the current CIRCT lowering path, ESI is used to implement the latency-insensitive hardware protocol.

## Types

DC defines two custom types.

`!dc.token` represents a control value with handshaking semantics. It carries no payload data. It says that a control transaction exists.

`!dc.value<T>` represents a payload value of type `T` wrapped with token semantics. The inner value may be any type. This is how data joins the dynamic-control network.

All DC-typed values have latency-insensitive semantics and FIFO ordering. Thinking of them as channels is usually the right mental model. A channel may be buffered without changing the logical program behavior, and the order of transactions is preserved.

## Operation Inventory

This checkout defines 12 `dc` operations.

### Buffering, Forking, And Sinking

```text
dc.buffer
dc.fork
dc.sink
dc.source
```

`dc.buffer` buffers a `!dc.token` or `!dc.value<T>` channel. The operation has a size attribute and optional initial values. The dialect itself does not define the exact implementation as cycles, stages, or storage cells; lowering decides how a buffer becomes hardware or another protocol.

`dc.fork` splits one incoming token into multiple outgoing tokens. DC values have implicit fork semantics by default, so this operation is only required when a lowering or analysis wants explicit one-use structure.

`dc.sink` consumes and discards an incoming token. Like forks, sinks are implicit at the semantic level but can be materialized for lowerings that require every DC value to have exactly one consumer.

`dc.source` always produces a token. It is useful for introducing a control source into a graph.

### Synchronization And Routing

```text
dc.join
dc.branch
dc.select
dc.merge
```

`dc.join` synchronizes multiple incoming tokens and emits one outgoing token. It is commutative and has canonicalization support. In a dataflow graph, a join says that several control events must all be ready before the next event proceeds.

`dc.branch` takes a `!dc.value<i1>` condition and emits either a true token or a false token. Only the selected output transacts.

`dc.select` is the inverse shape: it takes a `!dc.value<i1>` condition plus true and false input tokens, then propagates the selected input token to one output token.

`dc.merge` deterministically selects one of two incoming tokens and emits a `!dc.value<i1>` saying which input was selected. If both tokens are ready at the same time, the first input has priority. This deterministic behavior is important because DC intentionally does not lower nondeterministic Handshake merge behavior.

### Packing And Unpacking Data

```text
dc.pack
dc.unpack
```

`dc.pack` combines a `!dc.token` with a regular data value and produces a `!dc.value<T>`. This is common when a data-dependent control decision needs a condition value, such as producing a `!dc.value<i1>` for `dc.branch` or `dc.select`.

`dc.unpack` splits a `!dc.value<T>` into a `!dc.token` and the regular payload value `T`. This is how data-side operations can operate on the payload while control remains explicit.

The pack/unpack pair is the main way DC keeps data and control separate without losing their relationship.

### ESI Boundary Operations

```text
dc.to_esi
dc.from_esi
```

`dc.to_esi` converts a DC-typed value to an ESI channel. `dc.from_esi` converts an ESI channel back to a DC-typed value.

These operations mark the boundary between the abstract DC control language and the ESI channel implementation used by CIRCT's hardware lowering path.

### Complete Op List

For reference, the exact DC operation names in this checkout are:

```text
dc.branch, dc.buffer, dc.fork, dc.from_esi, dc.join, dc.merge, dc.pack, dc.select, dc.sink, dc.source, dc.to_esi, dc.unpack
```

## Transformations And Conversions

The local checkout defines 5 DC-related passes.

`dc-materialize-forks-sinks` inserts explicit `dc.fork` and `dc.sink` operations so that all DC-typed values have exactly one use. For a `!dc.value<T>`, the implementation unpacks the value, forks or sinks the token, and repacks the payload with the forked token where needed. This is useful for lowerings that want an explicit one-consumer discipline.

`dc-dematerialize-forks-sinks` removes explicit `dc.fork` and `dc.sink` operations. Fork results are replaced by the fork input, and sinks are erased. This is useful before canonicalization because DC's semantic model already allows values to be used multiple times or not at all.

`dc-print-dot` prints a `.dot` graph for a DC module. It emits graph nodes for DC operations and selected `comb` operations, and it can show called functions as subgraphs. This is an analysis/debugging pass rather than a lowering pass.

`lower-handshake-to-dc` lowers Handshake IR into DC. A `handshake.func` is currently converted into an `hw.module` because DC does not define its own graph-function container. The lowering converts Handshake dataflow constructs into DC control operations plus ordinary data operations. For example, Handshake conditional branch behavior becomes unpacking data and condition channels, joining tokens, packing a condition, branching control, and repacking data with the selected branch tokens. Nondeterministic Handshake operations are not generally representable because DC is deterministic.

`lower-dc-to-hw` lowers DC to ESI, HW, Comb, and Seq operations. DC values are converted into ESI channels. `!dc.value<T>` becomes an ESI channel carrying the hardware-compatible form of `T`, and `!dc.token` becomes a zero-width ESI channel. When the IR contains clocked operations such as buffers or forks, the parent function-like operation must provide clock and reset arguments marked with `dc.clock` and `dc.reset`.

The exact pass names covered in this chapter are:

```text
dc-dematerialize-forks-sinks, dc-materialize-forks-sinks, dc-print-dot, lower-dc-to-hw, lower-handshake-to-dc
```

## What It Implies

Seeing `!dc.token` means the IR is carrying pure control. There is no payload data, only a latency-insensitive transaction.

Seeing `!dc.value<T>` means a payload value is attached to a control token. Use `dc.unpack` to find the payload and token separately, and `dc.pack` to find where a payload joins control.

Seeing `dc.join` means several control events must synchronize. Seeing `dc.branch` or `dc.select` means a `!dc.value<i1>` controls which path transacts.

Seeing `dc.merge` means two control paths compete deterministically. The first input wins if both are available.

Seeing missing explicit `dc.fork` or `dc.sink` operations is not automatically a bug. DC values have implicit fork and sink semantics. Materialization passes insert explicit operations only when needed by a later lowering or analysis.

Seeing `dc.to_esi` or `dc.from_esi` means the IR is crossing between DC's abstract control-channel model and ESI's hardware channel model.

## How To Read DC IR

Start by identifying the types. A `!dc.token` value is control-only. A `!dc.value<i32>` value carries both a control transaction and an `i32` payload.

Then trace pack and unpack operations. They show where data enters or leaves the control network.

Next, inspect control structure. Joins synchronize, forks duplicate control, branches split control based on a condition, selects choose one incoming control path, and merges combine competing paths deterministically.

Finally, check whether fork and sink operations are explicit. If they are not present, remember that DC still permits multiple uses and zero uses. If they are present, a materialization pass probably prepared the IR for a lowering that requires explicit channel fanout and consumption.

## Minimal Example

This simplified fragment branches a control token using a boolean payload:

```mlir
%start = dc.source
%cond = arith.constant true
%cond_value = dc.pack %start, %cond : i1
%true_token, %false_token = dc.branch %cond_value
%joined = dc.join %true_token
dc.sink %false_token
```

Read this as "create a control event, attach a boolean condition, send the event down one branch, join the selected path, and explicitly discard the unselected path." In many DC stages, the sink could be implicit. If a lowering wants one-use explicitness, `dc-materialize-forks-sinks` will make that structure visible.

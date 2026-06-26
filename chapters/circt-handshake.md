# CIRCT `handshake` Dialect

The CIRCT `handshake` dialect is a dataflow hardware dialect. It models a program as a network of independent actors that communicate by passing tokens over FIFO-like channels. Instead of saying "execute this block, then this block" in a normal control-flow style, Handshake asks a hardware-oriented question: which operation can fire as soon as its inputs are ready, and where does the result token go next?

For a beginner, the most useful mental model is a circuit of small boxes connected by valid-ready wires. Each box waits for the tokens it needs, performs its operation, and forwards new tokens. Some tokens carry data, such as an integer address or a loaded value. Other tokens are control-only values with MLIR `none` type. These control tokens do not carry payload data, but they are still important because they say "this event happened" or "this path is allowed to continue."

Handshake is important because it sits between ordinary compiler IR and hardware-oriented CIRCT IR. A compiler can lower structured `func`, `cf`, and memory code into `handshake`, transform the resulting dataflow graph, and then lower it to `hw`, `comb`, `seq`, and `esi`, or to the CIRCT `dc` dialect. That makes Handshake a teaching point for both MLIR lowering and hardware compilation.

## Why This Dialect Exists

Traditional compiler IR is usually control-flow centered. It has blocks, branches, dominance, and a program counter-like idea of execution. Hardware dataflow wants something different. It wants explicit communication, explicit synchronization, and a representation where independent pieces can proceed concurrently.

The `handshake` dialect exists to make that structure visible. A `handshake.func` is not just a normal function with a different spelling. It has a graph region and extra verification rules. The most famous rule is that each SSA value must have a single use. If a value needs to feed two consumers, the IR should contain an explicit `handshake.fork` or `handshake.lazy_fork`. If a value is produced but intentionally ignored, the IR should contain a `handshake.sink`. This rule turns fanout and dead consumption into explicit hardware structure.

That explicitness is why Handshake matters. In software IR, a multi-use SSA value is just a convenience. In hardware, broadcasting a value has real wires, buffering, and backpressure implications. Handshake makes those implications part of the IR.

## When To Use It

Use Handshake when you are studying or building a lowering path from imperative or control-flow IR into hardware dataflow. It is especially relevant when the source has branches, joins, memory accesses, or pipeline-like behavior and the compiler needs to expose the token movement that will become hardware communication.

You will usually not write large Handshake programs by hand. More often, you inspect it after `lower-cf-to-handshake`, debug it with graph printing or op counting, run cleanup passes, insert buffers, and then lower it further. Hand-written snippets are still useful for learning because the dialect has a small set of core dataflow ideas: fork, merge, mux, branch, join, buffer, memory, and function boundaries.

Avoid treating Handshake as a general-purpose high-level programming dialect. It is already fairly close to hardware behavior. If you only want tensor algebra, affine loops, or source-like control flow, other MLIR dialects are easier places to start.

## Core Model

A Handshake program is a graph of operations inside `handshake.func`. Operations fire when their required inputs are available and their successors can accept output. This is the valid-ready discipline that appears in many hardware interfaces.

The dialect separates data movement from control movement. A branch operation forwards a token to a successor. A join operation waits until multiple predecessor tokens have arrived. A control merge records which predecessor path fired. A mux uses that recorded choice to select the matching data value.

The single-use rule shapes how you read the IR. If an operation result appears to be needed by multiple successors, look for a `handshake.fork`. If it has no meaningful successor, look for a `handshake.sink`. If the lowering has temporarily removed those operations, `handshake-materialize-forks-sinks` can insert them again before lowerings that require explicit fanout and discard points.

Buffers are also part of the semantic story, not just an optimization detail. A `handshake.buffer` can break cycles, hold values, and affect throughput. Buffer placement can change whether a cyclic dataflow graph is implementable and how well it pipelines.

Memory is modeled as request and response ports. Loads and stores talk to `handshake.memory` or `handshake.extmemory` operations. This is different from a normal `memref.load` or `memref.store`: the address, data, and control dependencies are explicit tokens.

## Operation Inventory

| Operation | What It Means |
| --- | --- |
| `handshake.func` | Defines a handshaked function with a graph region. It is similar to `func.func`, but values must obey Handshake's single-use discipline and the body is a dominance-free dataflow graph. |
| `handshake.return` | Terminates a `handshake.func`. It returns data results and includes the control-only exit of the function's token network. |
| `handshake.instance` | Instantiates or calls another Handshake function. Different instances of the same function have distinct state. The last operand and last result are control tokens. |
| `handshake.esi_instance` | Instantiates a Handshake circuit from outside a Handshake design using ESI channels, plus clock and reset. This is a boundary operation for integrating Handshake with ESI-based systems. |
| `handshake.buffer` | Inserts storage into a token path. It has a positive slot count and is either `seq` or `fifo`. Sequential buffers may have reset initialization values. |
| `handshake.fork` | Eagerly replicates one input token to multiple outputs. Each output can proceed as its successor becomes available. |
| `handshake.lazy_fork` | Replicates one input token to multiple outputs only when all successors are available. This is useful when the consumers must stay synchronized. |
| `handshake.merge` | Nondeterministically forwards one available input to one output. It represents a merge point where any predecessor may provide the next token. |
| `handshake.mux` | Deterministically selects one data input using a select token, usually from `handshake.control_merge`. It is for control-plus-dataflow selection, not ordinary pure data selection. |
| `handshake.control_merge` | Merges control or data tokens and also produces the index of the chosen input. That index commonly drives a `handshake.mux`. |
| `handshake.br` | Unconditionally forwards a token to one successor. It is the Handshake form of a simple branch edge. |
| `handshake.cond_br` | Forwards a data token to either the true or false output according to an `i1` condition token. |
| `handshake.sink` | Consumes and discards a token. It makes intentionally unused values explicit in the dataflow graph. |
| `handshake.source` | Produces a continuously valid token that a successor can consume. It is often used to seed control or constants. |
| `handshake.never` | Produces a disconnected source that never asserts valid. It represents a path that never fires. |
| `handshake.constant` | Emits a constant value when triggered by a control token. The constant is not just globally present; it participates in the handshaked network. |
| `handshake.memory` | Represents an independent flat memory or memory region. It receives load and store requests and returns loaded data plus completion tokens. |
| `handshake.extmemory` | Wraps a memref argument to a Handshake function as an external memory interface. Lowering can turn it into explicit load and store ports. |
| `handshake.load` | Sends address and control request tokens to a memory operation, then forwards returned data to its successor. |
| `handshake.store` | Sends store data, address, and control request tokens to a memory operation. |
| `handshake.join` | Waits for all inputs and produces one control-only output. It is the basic synchronization operation. |
| `handshake.sync` | Synchronizes an arbitrary set of inputs and forwards the same typed values after all are ready. |
| `handshake.unpack` | Splits a tuple token into separate element tokens. Its outputs can move onward as successors become ready. |
| `handshake.pack` | Builds a tuple token from separate inputs. Like a join, it waits until all tuple elements are ready. |

## Transformations And Conversions

| Pass | Role |
| --- | --- |
| `lower-cf-to-handshake` | Lowers `func` and control-flow IR into Handshake. It exposes branches, merges, muxes, memory requests, and dataflow-style control. Options include `source-constants` and `disable-task-pipelining`. |
| `handshake-remove-block-structure` | Removes block structure from Handshake IR after control-flow lowering, producing a flatter dataflow form. |
| `handshake-materialize-forks-sinks` | Inserts explicit forks and sinks so every value has exactly one use. This is often needed before hardware lowering. |
| `handshake-dematerialize-forks-sinks` | Removes explicit forks and sinks when a pass wants a less materialized graph. |
| `handshake-insert-buffers` | Inserts buffers, commonly to break graph cycles. It supports strategies such as `cycles`, `allFIFO`, and `all`, with configurable `buffer-size`. |
| `handshake-remove-buffers` | Removes buffers from Handshake functions, useful for cleanup or experiments. |
| `handshake-lock-functions` | Adds a locking mechanism so only one control token can be active in a function at a time. |
| `handshake-legalize-memrefs` | Lowers selected memref operations into a form suitable for `lower-cf-to-handshake`. |
| `handshake-split-merges` | Decomposes merge and control-merge operations with more than two inputs into two-input merge structures plus support logic. |
| `handshake-add-ids` | Assigns per-operation IDs inside a `handshake.func` so later lowered IR can be mapped back to Handshake operations. |
| `handshake-print-dot` | Prints a DOT graph for the top-level Handshake function, with called functions shown as subgraphs. |
| `handshake-op-count` | Counts operation resources in Handshake functions. This is useful for quick design-size inspection. |
| `handshake-lower-extmem-to-hw` | Lowers `handshake.extmemory` and memref inputs to explicit hardware memory ports. The `wrap-esi` option can create an ESI wrapper. |
| `lower-handshake-to-hw` | Lowers Handshake to CIRCT `hw`, `esi`, `comb`, and `seq` operations. This is the main route toward concrete hardware IR. |
| `lower-handshake-to-dc` | Lowers Handshake to the CIRCT `dc` dialect. At the time of the source inspected here, `handshake.func` is converted into `hw.module` as a container because DC has no generic graph-function container. |

## Typical Lowering Flow

A common path starts with ordinary MLIR function and control-flow IR. Before conversion, memory operations may be prepared with `handshake-legalize-memrefs`. Then `lower-cf-to-handshake` creates `handshake.func`, branches, merges, muxes, loads, stores, and memory interfaces.

After that, the graph is normalized. `handshake-remove-block-structure` removes leftover block structure. `handshake-materialize-forks-sinks` makes fanout and discards explicit. `handshake-split-merges` can simplify large merges into smaller pieces. `handshake-insert-buffers` can make cyclic graphs implementable and tune dataflow timing.

For hardware generation, `lower-handshake-to-hw` lowers to `hw`, `esi`, `comb`, and `seq`. This step cares about the structural details that the earlier passes prepared. In particular, explicit fork and sink materialization helps satisfy the one-use form expected by the lowering.

For dataflow-channel experimentation, `lower-handshake-to-dc` sends the design into CIRCT `dc`. This can be useful when you want to study dataflow communication separately from immediate RTL emission.

## How To Read Handshake IR

Start with the function boundary. A `handshake.func` tells you the data inputs, data outputs, and often the control entry or exit structure. Then find the token sources. A `handshake.source` or incoming control argument often starts the network.

Next, trace branch and merge pairs. A conditional path usually has `handshake.cond_br` for token routing, `handshake.control_merge` to remember which predecessor arrived, and `handshake.mux` to pick the matching data value. That trio is the Handshake version of an if-then-else value.

Then look for fanout. If one result feeds many conceptual users, `handshake.fork` or `handshake.lazy_fork` explains how the token is duplicated. The choice matters: eager forks let each output proceed independently, while lazy forks wait for all consumers.

For memory, read the `handshake.load`, `handshake.store`, and `handshake.memory` or `handshake.extmemory` together. The load and store operations describe request tokens. The memory operation ties those request ports to one memory resource and emits response tokens.

Finally, inspect buffers. A `handshake.buffer` is a hint that timing, backpressure, or cycles matter. It may be inserted by a pass rather than written by the original lowering, but it changes the structure that later hardware sees.

## What It Implies

Handshake makes concurrency visible. If two operations do not depend on each other by tokens, they may proceed independently. That is powerful, but it means the IR reader must think in terms of availability and backpressure rather than a single instruction order.

Handshake also makes hardware costs visible earlier than many source-like dialects. A fork implies fanout. A buffer implies storage. A merge implies arbitration. A mux implies selection logic. A memory operation implies ports and ordering constraints. Beginners should practice translating each operation into "what hardware communication does this require?"

The dialect is also a reminder that MLIR dialects are not only about syntax. A dialect can encode a specific execution model. Here, the execution model is token-based dataflow with valid-ready communication. The operations are the vocabulary, but the important lesson is the discipline they enforce: explicit synchronization, explicit fanout, explicit memory requests, and explicit lowering boundaries.


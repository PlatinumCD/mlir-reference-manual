# `async` Dialect

## Beginner Summary

The `async` dialect is MLIR's representation for asynchronous execution.

It gives the compiler IR-level concepts for work that may complete later:

- `!async.token` says that an asynchronous action has finished.
- `!async.value<T>` says that a value of type `T` will be available later.
- `async.execute` describes a region of work that can run independently after
  its dependencies are ready.
- `async.await` waits for a token or unwraps an async value.
- `async.group` collects multiple async tokens or values so they can be waited
  on together.

Think of `async` as MLIR's "future and task" dialect. It does not by itself
guarantee parallel execution. It makes possible concurrency explicit so later
passes can lower it to runtime calls, coroutines, threads, or a target-specific
execution system.

## Why This Dialect Exists

Many compiler pipelines need to represent latency and concurrency before they
know the final runtime.

Examples include:

- Launching independent compute tasks.
- Splitting an `scf.parallel` loop into pieces that can run on worker threads.
- Starting work while the caller continues with other work.
- Waiting for several pieces of work to complete.
- Returning a value from a computation that may finish later.

Without the `async` dialect, each producer would need to invent its own token
type, future type, wait operation, and lowering path. The `async` dialect gives
MLIR a common representation for these ideas.

The important beginner point is that `async` describes dependencies, not a
specific scheduler. A legal lowering may run tasks concurrently, but a fully
sequential lowering is also semantically valid. If one async task needs another
task's result or side effect, that dependency must be explicit in the IR.

## When It Matters

The `async` dialect matters when an MLIR pipeline wants to expose independent
work before lowering to an executable runtime.

It commonly appears in pipelines shaped like this:

```text
scf.parallel / higher-level parallel work
  -> async-parallel-for
  -> async.execute, async.await, async.group
  -> async-to-async-runtime / async-func-to-async-runtime
  -> async.runtime.* and async.coro.*
  -> async-runtime-ref-counting
  -> async-runtime-ref-counting-opt
  -> convert-async-to-llvm
  -> LLVM dialect plus Async Runtime API calls
```

It is also useful when a frontend already has futures, tasks, promises, or
non-blocking function calls and wants to keep that structure visible in MLIR.

## When To Use It

Use `async` when your IR needs to represent work that can complete later.

Use it for:

- Task-like regions with explicit dependencies.
- Future-like values that are produced asynchronously.
- Waiting for individual async values or completion tokens.
- Waiting for a fixed-size group of async work items.
- Async functions that can suspend instead of blocking a worker thread.
- Lowering structured CPU parallelism to a task runtime.

Do not use it merely because an operation is expensive. If the computation is
ordinary sequential work and no concurrency or delayed availability needs to be
represented, keep it in the source dialect. Also do not use `async` as a GPU
launch model; GPU host/device execution is represented by the `gpu` dialect,
though GPU host ops have their own `!gpu.async.token`.

## Core Concepts

### Tokens

`!async.token` represents completion.

It does not carry a payload value. It says only: the asynchronous operation that
created this token is complete. Other async work can depend on the token, and
`async.await` can wait for it.

Example shape:

```mlir
%token = ... : !async.token
async.await %token : !async.token
```

### Async Values

`!async.value<T>` represents a value that will become available later.

Awaiting an async value unwraps the payload:

```mlir
%future = ... : !async.value<f32>
%x = async.await %future : !async.value<f32>
```

The result `%x` has type `f32`, not `!async.value<f32>`.

### Execute Regions

`async.execute` is the main high-level operation.

It has three pieces:

- Dependency tokens in square brackets.
- Async body operands that are unwrapped before entering the region.
- A region that ends with `async.yield`.

The operation returns a completion token plus one `!async.value<T>` for each
value yielded by the body.

The body can execute concurrently with later IR, but concurrency is not
guaranteed. The only portable way to express ordering is through tokens and
async values.

### Body Operands

An `async.execute` body may receive `!async.value<T>` operands. Inside the body,
those operands become plain `T` values.

This means the body cannot start until those async values are ready.

```mlir
%token, %result = async.execute(%input as %x: !async.value<f32>)
    -> !async.value<f32> {
  %two = arith.constant 2.000000e+00 : f32
  %y = arith.mulf %x, %two : f32
  async.yield %y : f32
}
```

Here `%input` is a future outside the body, but `%x` is a normal `f32` inside
the body.

### Awaiting

`async.await` has two forms:

- Await a token: it blocks until completion and returns no value.
- Await an async value: it blocks until the value is ready and returns the
  payload.

Awaiting is a synchronization point. It tells the compiler that the following
IR may depend on the completed work.

### Groups

`!async.group` represents a fixed-size group of async tokens or async values.

The flow is:

1. Create a group with `async.create_group`.
2. Add tokens or values with `async.add_to_group`.
3. Wait for the whole group with `async.await_all`.

The group size is chosen at creation time. `async.await_all` waits until the
group has received that many elements and those elements are ready.

### Async Functions

`async.func` is an async callable operation. It is paired with:

- `async.call` for direct calls to async functions.
- `async.return` for returning from async functions.

An async function cannot return `void`. It must return an async token, one or
more async values, or both. The token represents completion of side effects, and
the async values represent delayed payload results.

In custom assembly, `async.return` is printed as `return` inside an
`async.func`, because `async.func` uses `async` as its default dialect.

### Runtime And Coroutine Forms

The dialect also contains lower-level operations:

- `async.runtime.*` operations model the Async Runtime API.
- `async.coro.*` operations model LLVM coroutine concepts in a typed MLIR form.

Beginners should read these as compiler-generated lowering IR. You usually
author `async.execute`, `async.await`, groups, and async functions. Passes then
introduce runtime and coroutine operations before final lowering to LLVM.

## Types

| Type | Meaning |
| --- | --- |
| `!async.token` | Completion token for asynchronous work. |
| `!async.value<T>` | Future-like value that will eventually contain a `T`. |
| `!async.group` | Fixed-size group of async tokens or values. |
| `!async.coro.id` | Typed representation of a switched-resume coroutine id. |
| `!async.coro.handle` | Handle to a coroutine frame. |
| `!async.coro.state` | Saved coroutine state passed to suspension. |

The first three types are the ones users normally see in high-level async IR.
The coroutine types appear during lowering.

## Operations

The dialect has 29 operations.

### High-Level Async Operations

| Operation | Purpose |
| --- | --- |
| `async.execute` | Defines an asynchronous task region. |
| `async.yield` | Terminates an `async.execute` body and yields payload results. |
| `async.await` | Waits for a token or unwraps an async value. |
| `async.create_group` | Creates a fixed-size async group. |
| `async.add_to_group` | Adds a token or value to a group and returns its rank. |
| `async.await_all` | Waits for all elements in a group. |
| `async.func` | Defines an async function. |
| `async.call` | Calls an async function by symbol. |
| `async.return` | Terminates an async function. |

### Coroutine Lowering Operations

| Operation | Purpose |
| --- | --- |
| `async.coro.id` | Creates a switched-resume coroutine identifier. |
| `async.coro.begin` | Allocates a coroutine frame and returns a handle. |
| `async.coro.save` | Saves coroutine state before suspension. |
| `async.coro.suspend` | Suspends a coroutine with suspend, resume, and cleanup successors. |
| `async.coro.end` | Marks the end of a coroutine in the suspend block. |
| `async.coro.free` | Frees a coroutine frame. |

### Runtime Operations

| Operation | Purpose |
| --- | --- |
| `async.runtime.create` | Creates a runtime token or async value in the non-ready state. |
| `async.runtime.create_group` | Creates a runtime group. |
| `async.runtime.set_available` | Marks a token or value as available. |
| `async.runtime.set_error` | Marks a token or value as failed. |
| `async.runtime.is_error` | Tests whether a token, value, or group is in the error state. |
| `async.runtime.await` | Blocks the caller thread until a token, value, or group is ready or failed. |
| `async.runtime.await_and_resume` | Waits for an async object and resumes a coroutine on the runtime. |
| `async.runtime.resume` | Resumes a coroutine on a runtime-managed thread. |
| `async.runtime.store` | Stores a payload value into an async runtime value. |
| `async.runtime.load` | Loads a payload value from an async runtime value. |
| `async.runtime.add_to_group` | Adds a runtime token or value to a runtime group. |
| `async.runtime.add_ref` | Increments the runtime reference count of a token, value, or group. |
| `async.runtime.drop_ref` | Decrements the runtime reference count of a token, value, or group. |
| `async.runtime.num_worker_threads` | Reads the runtime worker-thread count. |

### `async.execute`

`async.execute` is the operation most beginners should learn first.

Example:

```mlir
func.func @run(%input: !async.value<f32>, %dep: !async.token) -> f32 {
  %token, %result = async.execute [%dep](%input as %x: !async.value<f32>)
      -> !async.value<f32> {
    %two = arith.constant 2.000000e+00 : f32
    %y = arith.mulf %x, %two : f32
    async.yield %y : f32
  }
  async.await %token : !async.token
  %value = async.await %result : !async.value<f32>
  return %value : f32
}
```

Read this as:

- `%dep` must be ready before the task starts.
- `%input` must be ready before the body receives `%x`.
- `%token` becomes available when the body finishes.
- `%result` becomes available when the yielded `f32` is ready.

### `async.await`

`async.await` is a synchronization operation.

For tokens:

```mlir
async.await %token : !async.token
```

For async values:

```mlir
%x = async.await %value : !async.value<i32>
```

The second form unwraps the `i32`.

### Group Operations

Groups are useful when the number of async items is known and the program wants
one wait operation for all of them.

```mlir
func.func @wait_for_many(%a: !async.token, %b: !async.value<i32>) {
  %c2 = arith.constant 2 : index
  %group = async.create_group %c2 : !async.group
  %rank0 = async.add_to_group %a, %group : !async.token
  %rank1 = async.add_to_group %b, %group : !async.value<i32>
  async.await_all %group
  return
}
```

The ranks are stable positions in the group. They are often ignored if the
program only needs to wait for completion.

### Async Function Operations

`async.func` gives the dialect a function-like async abstraction.

```mlir
async.func @produce() -> !async.value<i32> {
  %c42 = arith.constant 42 : i32
  async.return %c42 : i32
}

func.func @caller() -> i32 {
  %v = async.call @produce() : () -> !async.value<i32>
  %x = async.await %v : !async.value<i32>
  return %x : i32
}
```

This chapter writes `async.return` explicitly to make the dialect clear. MLIR's
printer may display it as `return` inside the body of `async.func`.

## Transformations

The Async dialect has a small transformation surface, but the passes are
central to how the dialect becomes executable.

| Pass | What it does |
| --- | --- |
| `async-parallel-for` | Converts `scf.parallel` operations to multiple async compute tasks over non-overlapping iteration ranges. |
| `async-to-async-runtime` | Lowers high-level async operations such as `async.execute` to explicit `async.runtime.*` and `async.coro.*` operations. |
| `async-func-to-async-runtime` | Lowers `async.func` operations to explicit runtime and coroutine operations. |
| `async-runtime-ref-counting` | Inserts automatic reference counting for async runtime values after high-level lowering. |
| `async-runtime-ref-counting-opt` | Removes redundant async runtime reference-counting operations. |
| `async-runtime-policy-based-ref-counting` | Inserts lower-overhead reference counting using a stricter usage policy. |

### `async-parallel-for`

`async-parallel-for` is a bridge from structured CPU parallelism to task-style
async IR.

It splits an `scf.parallel` operation into independent iteration ranges and
uses async compute tasks to execute those ranges. Its important options are:

| Option | Meaning |
| --- | --- |
| `async-dispatch` | If true, dispatches work using recursive async work splitting. If false, launches tasks using a simple loop in the caller thread. |
| `num-workers` | Number of workers to assume. `-1` queries the runtime. |
| `min-task-size` | Minimum task size for sharding the parallel operation. |

This pass is useful when a pipeline has already expressed parallel loop work in
`scf.parallel` and wants to lower it toward a CPU task runtime.

### Runtime Lowering

`async-to-async-runtime` lowers high-level operations to a form that looks more
like a runtime implementation.

For example, an `async.execute` task is lowered toward:

- Creating runtime async tokens or values.
- Creating coroutine state.
- Storing yielded values into async runtime storage.
- Marking tokens or values available.
- Suspending and resuming through runtime calls.

This is the stage where the convenient source-level task form becomes explicit
machinery.

### Reference Counting

Async runtime values are reference counted.

The dialect has explicit `async.runtime.add_ref` and
`async.runtime.drop_ref` operations because tokens, values, and groups can
outlive the operation that created them. The automatic reference counting pass
uses liveness information to insert these operations.

The policy-based reference counting pass is cheaper but assumes:

- An async token is awaited or added to a group only once.
- An async value or group is awaited only once.

If those assumptions are violated, policy-based reference counting can produce
incorrect lifetime behavior. Use the automatic pass unless the pipeline can
prove the policy holds.

## Conversions And Lowering Paths

The main conversion is:

| Pass | Target |
| --- | --- |
| `convert-async-to-llvm` | Converts Async dialect operations to LLVM dialect operations and Async Runtime API calls. |

The lowering path is best understood in layers:

```text
High-level async IR
  async.execute
  async.await
  async.func / async.call / async.return
  async.group operations

Runtime/coroutine async IR
  async.runtime.create
  async.runtime.await
  async.runtime.store
  async.runtime.load
  async.coro.begin
  async.coro.suspend

LLVM dialect and runtime calls
  LLVM coroutine intrinsics
  calls to the MLIR Async Runtime API
```

The Async-to-LLVM conversion uses the Async Runtime API declared by the MLIR
execution engine. Runtime-level tokens, values, and groups lower to opaque
runtime objects. Coroutine operations lower to LLVM coroutine intrinsics.

This matters because the final executable behavior depends on the runtime
linked with the program. The `async` dialect itself describes the dependency
structure and the lowering interface; the runtime decides how worker threads,
queues, and resumption actually run.

## Example IR

### One Async Task

```mlir
func.func @run(%input: !async.value<f32>, %dep: !async.token) -> f32 {
  %token, %result = async.execute [%dep](%input as %x: !async.value<f32>)
      -> !async.value<f32> {
    %two = arith.constant 2.000000e+00 : f32
    %y = arith.mulf %x, %two : f32
    async.yield %y : f32
  }
  async.await %token : !async.token
  %value = async.await %result : !async.value<f32>
  return %value : f32
}
```

### Waiting For A Group

```mlir
func.func @wait_for_many(%a: !async.token, %b: !async.value<i32>) {
  %c2 = arith.constant 2 : index
  %group = async.create_group %c2 : !async.group
  %rank0 = async.add_to_group %a, %group : !async.token
  %rank1 = async.add_to_group %b, %group : !async.value<i32>
  async.await_all %group
  return
}
```

### Async Function And Call

```mlir
async.func @produce() -> !async.value<i32> {
  %c42 = arith.constant 42 : i32
  async.return %c42 : i32
}

func.func @caller() -> i32 {
  %v = async.call @produce() : () -> !async.value<i32>
  %x = async.await %v : !async.value<i32>
  return %x : i32
}
```

## Mental Model

The simplest mental model is a dependency graph.

- `async.execute` creates a task node.
- `!async.token` is an edge for completion.
- `!async.value<T>` is an edge carrying a future payload.
- `async.await` joins the async graph back into ordinary sequential IR.
- `async.group` is a bundle of dependency edges.

The dialect is not a promise that every task runs on a separate thread. It is a
promise that the dependencies are explicit enough for a later lowering to make
that choice.

## Gotchas

### Async Does Not Imply Parallel Execution

The `async.execute` operation can be lowered sequentially. If correctness
requires one task to observe another task's effects, use explicit tokens or
async values.

### Shared State Still Needs Explicit Ordering

Do not rely on shared mutable state to create hidden dependencies between async
regions. If a later operation depends on an earlier async region, carry a token
or value and await it.

### `async.await` Can Block

High-level `async.await` is convenient, but lowering must decide whether it
blocks a thread or suspends a coroutine. The dialect has an
`async.allowed_to_block` function attribute for functions where blocking
runtime awaits are allowed. Without that permission, asyncification may convert
a function to a coroutine.

### Runtime Ops Are Not The Friendly Surface

`async.runtime.*` and `async.coro.*` operations are important for lowering, but
they are not the usual authoring surface. If you are writing examples for
beginners, start with `async.execute`, `async.await`, and groups.

### Groups Have Fixed Size

The size passed to `async.create_group` matters. `async.await_all` first waits
for the expected number of elements to be added, then waits for those elements
to become ready.

### Lifetime Is Explicit After Runtime Lowering

At the runtime level, async tokens, values, and groups are reference counted.
Dropping or retaining references incorrectly can cause leaks or invalid runtime
behavior. That is why the reference counting passes are part of the normal
lowering story.

## Source Map

Use these source files when you want to inspect or update the dialect:

| Area | File |
| --- | --- |
| Dialect definition | `mlir/include/mlir/Dialect/Async/IR/AsyncDialect.td` |
| Types | `mlir/include/mlir/Dialect/Async/IR/AsyncTypes.td` |
| Operations | `mlir/include/mlir/Dialect/Async/IR/AsyncOps.td` |
| Async passes | `mlir/include/mlir/Dialect/Async/Passes.td` |
| Async transformations | `mlir/lib/Dialect/Async/Transforms/` |
| Async-to-LLVM conversion | `mlir/lib/Conversion/AsyncToLLVM/AsyncToLLVM.cpp` |
| Dialect tests | `mlir/test/Dialect/Async/` |
| Conversion tests | `mlir/test/Conversion/AsyncToLLVM/` |

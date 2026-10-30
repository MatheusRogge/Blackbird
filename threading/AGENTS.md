# threading

A minimal task executor. Only depends on `std` (+ `thiserror`, currently unused).

## API (`src/executor.rs`, re-exported from `lib.rs`)

```rust
pub trait Executor: Send + Sync + 'static {
    fn spawn(&self, task: Box<dyn FnOnce() + Send + 'static>);
    fn spawn_named(&self, name: String, f: Box<dyn FnOnce(ShutdownSignal) + Send + 'static>);
    fn shutdown(&self);
}

pub struct ShutdownSignal; // .should_stop() -> bool, non-blocking
pub struct DefaultExecutor; // ::new(), Default, Drop calls shutdown()
```

## `DefaultExecutor` behaviour
- `spawn`: sends the task over an mpsc channel to **one** worker thread (`executor-worker`). Tasks run serially in FIFO order. One slow task (a large glTF load, say) delays every task queued after it.
- `spawn_named`: creates a dedicated OS thread with its own shutdown channel. The closure must poll `signal.should_stop()` itself.
- `shutdown`: signals every named thread and joins it (logging panics to stderr), then drops the task sender so the worker exits after draining, and joins the worker. It is idempotent.
- All state sits behind one `Mutex`. Calling `spawn` from inside a task is fine because the lock is released before tasks run.

## Usage in the engine
`engine::Engine` holds an `Arc<dyn Executor>` (default `DefaultExecutor`, or inject one with `HeadlessApplication::with_executor`). `engine`'s `AssetsPlugin` hands it to `AssetManager::init_executor`, and `AssetManager::load_async` uses `spawn`.

If you need parallel loading, implement a pool-backed `Executor` rather than changing call sites.

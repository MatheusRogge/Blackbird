# engine

The user-facing facade. Owns the `Engine` state object, the `Plugin` system, and the two application runners. Games and tools should depend only on `engine` and import subsystems through its re-exports.

## Features

| Feature | Enables |
|---------|---------|
| *(none)* | `Engine`, `Plugin`, `HeadlessApplication`, `PlayerController` |
| `assets` | `assets` crate, `Engine::assets()`, `register_asset_resolvers` |
| `gltf` | + `gltf` crate (resolver for `.gltf`/`.glb`) |
| `window` | + `assets`, `rendering` + `winit` + `pollster`, `windowed::WindowedApplication`, frame stats |
| `ui` | + `ui` crate, `WindowedApplication::add_ui_layer` |
| `full` | assets + gltf + window + ui |
| `debug` | window + ui + `rendering/debug`: inserts `rendering::DebugLayer` as the first UI layer |
| `logging` | `engine::logging`: `env_logger` to stderr + optional file + in-memory `LogHistory` |

Check that feature combos still build after you change cfg-gated code (`cargo check -p engine --no-default-features`, `--features window`, `--features full`). `window` without `ui` is what production `player` builds use, and `gltf` without `window` is what `build_tool` uses.

## Modules

### `lib.rs`: `Engine`
```rust
engine.world() -> RwLockWriteGuard<World>   // takes the write lock; drop it quickly
engine.world_arc() -> Arc<RwLock<World>>
engine.input() -> &mut InputBuffer
engine.input_and_world() -> (&mut InputBuffer, RwLockWriteGuard<World>)  // split borrow
engine.ui_wants_pointer() / ui_wants_keyboard() -> bool
engine.assets() / assets_mut()              // feature "assets"; executor + resolvers already set up
engine.executor() -> Arc<dyn Executor>      // shared background executor
engine.frame_stats() -> rendering::FrameStats // feature "window"; CPU+GPU timings per pass
```
`register_asset_resolvers(&mut AssetManager)` registers every resolver the enabled features provide. `Engine::new` calls it, and `build_tool` calls it so that a build's dependency walk matches what the game can load. Register new built-in resolvers there, not in a plugin.

Re-exports: `engine_core::{entity, input, world}`, `Entity`, `EntityKey`, `Controllable`, `InputEvent`, `Key`, `Modifiers`, `MouseButton`, `EntityHandle`, plus the crates `assets`, `gltf`, `rendering`, `ui` behind their features.

**Deadlock trap:** `engine.world()` returns a guard. Holding it while calling another `engine.world()` (or anything else that locks) deadlocks the game thread. Write `let mut world = engine.world();` inside a block, or call `engine.world().add_entity(..)` as one statement.

### `plugin.rs`
```rust
pub trait Plugin: Send + 'static {
    fn setup(&mut self, engine: &mut Engine) -> Result<(), EnginePluginError>;
    fn tick(&mut self, _engine: &mut Engine, _delta: f32) {}
}
```
Any `FnMut(&mut Engine) -> Result<(), EnginePluginError>` closure is a setup-only plugin.

### `application.rs`
- `HeadlessApplication::new()` / `with_executor(Arc<dyn Executor>)`
- `.add_plugin(p)` runs `p.setup` **immediately** (so plugins initialise in the order you add them) and **panics** on error.
- `.run()` is a fixed 60 Hz loop: `tick_all` then plugin ticks, forever. It never returns `Ok`.
- `ApplicationError { message, source }`.

### `windowed.rs`: `WindowedApplication<B: RenderGraphBuilder>`
```rust
WindowedApplication::new(render_graph_builder)
    .with_title("My Game")
    .add_plugin(MyGame::default())
    .add_ui_layer(MyLayer)                       // feature "ui"
    .run()?;
```
On winit `resumed` it creates the window, builds the `Renderer` (blocking on `pollster`), takes `Engine` and the plugins out of the inner `HeadlessApplication`, and spawns the **game** and **render** threads (see the root `AGENTS.md` for the thread model). Plugin setup already ran on the main thread at `add_plugin` time, before any window exists.

Per game tick: inputs → `InputBuffer`, `World::tick_all`, `Plugin::tick` for each plugin in order, then the input batch goes to the render thread for egui.

Winit translation: named keys map to `Key::*`. Everything else becomes `Key::Scancode(physical scancode)`. Pixel scroll deltas are divided by 20.

### `player_controller.rs`
`PlayerController::new(speed)`, `.attach(EntityKey)`, `.tick::<E: Controllable>(engine, delta)`. It consumes WASD (scancodes 17/31/30/32) and re-queues every other event. Call it from your plugin's `tick`.

### `logging.rs` (feature `logging`)
`logging::init(LogConfig { level, file, history }) -> LogHistory` installs the global logger. `RUST_LOG` still overrides `level`. Each record goes to stderr, to the file if set (parent dirs are created) and to the last `history` lines, which `LogHistory::snapshot()` returns for in-game consoles.

## Gotchas
- `src/entity.rs`, `src/input.rs`, `src/world.rs` are **not** in the module tree. They are stale copies of `engine_core`. Delete rather than edit.
- Shutdown: `CloseRequested` sends `GameMsg::Shutdown` and exits the event loop. The thread handles are stored but never joined.
- Most setup failures in `resumed` are `unwrap()`s (window, renderer creation).

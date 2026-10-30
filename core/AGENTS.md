# core (`engine_core`)

Engine-agnostic primitives shared by every other crate. No GPU, windowing or threading code. Dependencies: `slotmap`, `downcast-rs`, `ultraviolet`.

## Modules

### `entity`
- `EntityKey`: a `slotmap` key. Generational, so stale keys return `None`.
- `trait Entity: Any + Downcast + Send + Sync + 'static { fn tick(&mut self, delta: f32) {} }`
- `trait Controllable: Entity { fn translate(&mut self, Vec3); fn forward(&self) -> Vec3; fn right(&self) -> Vec3 }`. Defaults are -Z forward and +X right. Used by `engine::player_controller::PlayerController`.

### `world`
`World` is a heterogeneous store, `SlotMap<EntityKey, Box<dyn Entity>>`. It is not an ECS: each entity is one concrete type, and queries downcast by type.

| Method | Notes |
|--------|-------|
| `add_entity<E>(E) -> EntityKey` | |
| `get_entity<E>(key) -> Option<&E>` / `get_entity_mut` | `None` if the key is missing **or** the type doesn't match |
| `get_entities<E>() -> Vec<&E>` / `get_entities_mut` | Linear scan over all entities. Allocates a `Vec`. |
| `get_entity_handles<E>() -> Vec<EntityHandle<E>>` | `{ id, entity }` pairs. Use this when you need keys (e.g. GPU caches keyed by `EntityKey`). |
| `tick_all(delta)` | Calls `Entity::tick` on everything |
| `remove_entity(key) -> bool`, `clear()` | |

Iteration order is slotmap order, not insertion order guaranteed across removals. Code that takes "the first camera" (`get_entities::<Camera>().first()`) depends on this.

### `input`
- `InputEvent`: keys (`Key` enum: named keys + `Scancode(u32)`), text, modifiers, mouse buttons/move/scroll, cursor left, focus, scale factor, `FileDrop(PathBuf)`.
- `InputBuffer`: a FIFO capped at **64** events. `push` adds to the front and truncates, so under overflow the *oldest* events are dropped. `pop` takes the oldest.

Consumers drain with `std::iter::from_fn(|| input.pop())` and **re-push the events they didn't handle** (see `PlayerController`, `CameraController`). Re-push them in their original (oldest-first) order. `push` adds to the front, so reversing them would flip the queue for the next consumer. Any new input consumer must follow the same pattern, or it will starve consumers that run after it.

WASD arrives as `Key::Scancode(17/31/30/32)`, which are PC set-1 scancodes from winit.

## Gotchas
- `engine/src/{entity,input,world}.rs` are orphaned copies of these modules (not declared in `engine/src/lib.rs`). This crate is the source of truth. Don't edit the copies, delete them.
- Adding an `InputEvent` variant requires updating the exhaustive match in `ui/src/renderer.rs` and the winit translation in `engine/src/windowed.rs`.

# ui

egui (0.37) integration on top of wgpu. Re-exports `egui`. Depends only on `engine_core`, so it knows nothing about `rendering` or `engine`.

## API

```rust
pub trait UiLayer: Send + 'static {
    fn ui(&mut self, ctx: &egui::Context, world: &World);
}

pub struct EguiRenderer;
EguiRenderer::new(device, surface_format, layers: Vec<Box<dyn UiLayer>>,
                  ui_wants_pointer: Arc<AtomicBool>, ui_wants_keyboard: Arc<AtomicBool>)
renderer.on_events(&[InputEvent])          // translate engine input → egui RawInput
renderer.execute(device, queue, encoder, view, (w, h), &World)
```

## How it is wired
`engine::windowed` wraps `EguiRenderer` in an `EguiOverlay` that implements `rendering::renderer::OverlayPass`. Everything runs **on the render thread**:
1. The render thread receives the frame's `Vec<InputEvent>` from the game thread → `on_events`.
2. `Renderer::render` runs the graph, then calls the overlay with a **non-sRGB view** of the swapchain texture (egui outputs gamma-space colors) and submits it in a separate command buffer.
3. `execute` runs every layer in order, tessellates, and renders with `LoadOp::Load` on top of the frame.
4. After the pass it stores `egui_wants_pointer_input()` / `egui_wants_keyboard_input()` into the shared atomics. The game thread reads them through `Engine::ui_wants_pointer()` / `ui_wants_keyboard()`, one frame late.

Layers get `&World` under the render thread's **read** lock. Don't mutate game state from a layer. Share an `Arc<Mutex<_>>` with a plugin instead (that is how `editor` does it), and use `try_lock` so the render thread never blocks on the game thread.

## Input translation notes
- Mouse positions are divided by `pixels_per_point` (set from `InputEvent::ScaleFactorChanged`).
- `Key::Scancode(_)` is not forwarded as an egui key. Letter keys reach egui only as `TextInput`.
- `command` modifier = ctrl, and `mac_cmd` is always false.
- The match over `InputEvent` is exhaustive. New variants must be handled here.

## egui 0.37 gotchas
- Panels (`egui::Panel`, `CentralPanel`) only show inside a `Ui`. From a `UiLayer`, wrap them in `ctx.root_ui(|ui| …)`. Windows and `Area`s still take `&Context`.
- `egui` no longer enables `default_fonts` by default. It is turned on in this crate's `Cargo.toml`. Without it, no text renders.
- `TexturesDelta` debug-asserts on drop if it still holds deltas, so `execute` drains it (`set` before painting, `free` after).

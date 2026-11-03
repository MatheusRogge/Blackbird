mod camera;
mod export;
mod import;
mod lights;
mod math;
mod plugin;
mod project;
mod recent;
mod state;
mod ui;

use std::sync::{Arc, RwLock};

use engine::rendering::debug_flags::RenderDebugFlags;
use engine::rendering::pbr::RenderGraphPBRBuilder;
use engine::rendering::shader::ShaderAsset;
use engine::windowed::WindowedApplication;
use log::LevelFilter;

use plugin::EditorPlugin;
use state::EditorState;
use ui::EditorUiLayer;

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .format_source_path(true)
        .format_level(true)
        .format_timestamp_millis()
        .parse_default_env()
        .init();

    let shared_state = EditorState::new();

    let gbuffer_shader   = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/gbuffer.wgsl"));
    let cluster_shader   = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/cluster_assignment.wgsl"));
    let lighting_shader  = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/lighting.wgsl"));
    let present_shader   = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/present.wgsl"));
    let shadow_shader    = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/shadow.wgsl"));
    let probe_trace      = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/probe_trace.wgsl"));
    let probe_update     = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/probe_update.wgsl"));
    let debug_overlay    = ShaderAsset::from_raw(include_str!("../../sandbox/shaders/debug_overlay.wgsl"));

    let debug_flags = Arc::new(RwLock::new(RenderDebugFlags {
        show_grid: true,
        ..RenderDebugFlags::default()
    }));

    let render_graph = RenderGraphPBRBuilder::new(
        gbuffer_shader,
        cluster_shader,
        lighting_shader,
        present_shader,
        shadow_shader,
        probe_trace,
        probe_update,
        debug_overlay,
    ).with_debug_flags(debug_flags);

    WindowedApplication::new(render_graph)
        .add_plugin(EditorPlugin::new(shared_state.clone()))
        .add_ui_layer(EditorUiLayer::new(shared_state))
        .run()?;

    Ok(())
}

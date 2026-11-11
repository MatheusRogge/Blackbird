//! Keeps the View menu and GI settings in `EditorState` (saved in `project.json`) in step with the
//! renderer's debug flags, which the View menu, the GI window and the F1 overlay all edit.

use engine::rendering::debug_flags::RenderDebugFlags;

use crate::state::{EditorState, GiSettings, ViewSettings};

/// Pushes freshly loaded project settings into the flags once, then mirrors the flags back into
/// the state every frame.
pub fn sync(state: &mut EditorState, flags: &mut RenderDebugFlags) {
    if std::mem::take(&mut state.apply_view_settings) {
        apply(&state.view_settings, &state.gi_settings, flags);
        state.timeline.show = state.view_settings.animation;
        return;
    }
    let view = &mut state.view_settings;
    view.grid = flags.show_grid;
    view.animation = state.timeline.show;
    view.reflections = flags.reflections;
    view.ssr = flags.ssr;
    view.traced_reflections = flags.traced_reflections;
    view.reflection_probe = flags.reflection_probe;
    view.ambient_occlusion = flags.ambient_occlusion;
    view.contact_shadows = flags.contact_shadows;
    view.particles = flags.particles;
    state.gi_settings = GiSettings {
        hysteresis: flags.ddgi_hysteresis,
        normal_bias: flags.ddgi_normal_bias,
        view_bias: flags.ddgi_view_bias,
        probe_spacing: flags.ddgi_probe_spacing,
        grid_anchor: flags.ddgi_grid_anchor,
        indirect_intensity: flags.ddgi_indirect_intensity,
        bounce: flags.ddgi_bounce,
        update_interval: flags.ddgi_update_interval,
    };
}

fn apply(view: &ViewSettings, gi: &GiSettings, flags: &mut RenderDebugFlags) {
    flags.show_grid = view.grid;
    flags.reflections = view.reflections;
    flags.ssr = view.ssr;
    flags.traced_reflections = view.traced_reflections;
    flags.reflection_probe = view.reflection_probe;
    flags.ambient_occlusion = view.ambient_occlusion;
    flags.contact_shadows = view.contact_shadows;
    flags.particles = view.particles;
    flags.ddgi_hysteresis = gi.hysteresis;
    flags.ddgi_normal_bias = gi.normal_bias;
    flags.ddgi_view_bias = gi.view_bias;
    flags.ddgi_probe_spacing = gi.probe_spacing;
    flags.ddgi_grid_anchor = gi.grid_anchor;
    flags.ddgi_indirect_intensity = gi.indirect_intensity;
    flags.ddgi_bounce = gi.bounce;
    flags.ddgi_update_interval = gi.update_interval;
}

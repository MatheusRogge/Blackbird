//! The editor's animation clips. Like `engine::gltf::animation::ClipData`, except that light tracks
//! name lights by `SceneLight::id`: save writes lights in the editor's order, so their glTF indices
//! change as lights are added or deleted. Converted from glTF on load (`load_clips`) and back on
//! save and for the viewport preview (`to_clip_data`).

use std::collections::HashMap;

use engine::gltf::animation::{
    ClipData, LightProperty, ParticleProperty, Target, TrackData, read_clip_data, standard_tracks,
};
use engine_core::animation::{Curve, Interpolation, Key, Value};
use gltf_api::Document;

use crate::math::{self, euler_to_quat, quat_to_euler};
use crate::state::{EditorState, LightData, SceneLight};
use ultraviolet::Mat4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EditorTarget {
    /// Anything but a light.
    Scene(Target),
    Light(u64, LightProperty),
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditorTrack {
    pub target: EditorTarget,
    pub curve: Curve,
    /// The channel of the clip's standard glTF animation this track shows. Saved (as an override)
    /// only once `curve` differs from it.
    pub standard: Option<Curve>,
}

impl EditorTrack {
    pub fn edited(&self) -> bool {
        self.standard.as_ref() != Some(&self.curve)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditorClip {
    pub name: String,
    pub looping: bool,
    pub autoplay: bool,
    pub speed: f32,
    /// A standard glTF animation the clip also plays. Its channels are tracks with `standard` set.
    pub animation: Option<usize>,
    pub tracks: Vec<EditorTrack>,
}

impl EditorClip {
    pub fn new(name: String) -> Self {
        Self { name, looping: true, autoplay: true, speed: 1.0, animation: None, tracks: Vec::new() }
    }

    pub fn duration(&self) -> f32 {
        self.tracks.iter().map(|t| t.curve.duration()).fold(0.0, f32::max)
    }

    /// Adds a key for `target` at `time` (replacing one within a millisecond), creating the track
    /// if needed. Returns the track's index.
    pub fn set_key(&mut self, target: EditorTarget, time: f32, value: Value) -> usize {
        let index = match self.tracks.iter().position(|t| t.target == target) {
            Some(index) => index,
            None => {
                let components = components(target);
                let curve = if matches!(target, EditorTarget::Scene(Target::Rotation(_))) {
                    Curve::rotation(Vec::new())
                } else {
                    Curve::new(components, Vec::new())
                };
                self.tracks.push(EditorTrack { target, curve, standard: None });
                self.tracks.len() - 1
            }
        };
        let keys = &mut self.tracks[index].curve.keys;
        match keys.iter_mut().find(|k| (k.time - time).abs() < 1e-3) {
            Some(key) => key.value = value,
            None => {
                let interpolation = keys.iter().rev().find(|k| k.time < time).map_or(Interpolation::Linear, |k| k.interpolation);
                keys.push(Key::new(time, value, interpolation));
                keys.sort_by(|a, b| a.time.total_cmp(&b.time));
            }
        }
        index
    }
}

/// The Animation panel's selection and playhead. Not part of undo.
#[derive(Clone, Debug, Default)]
pub struct Timeline {
    pub show: bool,
    pub clip: Option<usize>,
    pub track: Option<usize>,
    pub key: Option<usize>,
    /// Seconds into the clip.
    pub time: f32,
    pub playing: bool,
    /// The viewport shows the clip at `time`. Turned on by playing or scrubbing.
    pub preview: bool,
    /// Every clip previewing together instead, as the game plays them. Takes over from `preview`.
    pub all: Option<PlayAll>,
    /// The curve view's time and (shown) value ranges; `None` fits the clip / the track.
    pub view_time: Option<(f32, f32)>,
    pub view_values: Option<(f32, f32)>,
}

/// Seconds since every clip started together, each at its own speed and wrapping or holding at
/// its end.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayAll {
    pub time: f32,
    pub playing: bool,
}

impl Timeline {
    /// Drops selections that no longer exist (after undo or a load).
    pub fn clamp(&mut self, clips: &[EditorClip]) {
        let Some(clip) = self.clip.and_then(|c| clips.get(c)) else {
            *self = Self { show: self.show, all: self.all.filter(|_| !clips.is_empty()), ..Self::default() };
            return;
        };
        let Some(track) = self.track.and_then(|t| clip.tracks.get(t)) else {
            self.track = None;
            self.key = None;
            return;
        };
        if self.key.is_some_and(|k| k >= track.curve.keys.len()) {
            self.key = None;
        }
    }

    /// Moves the playhead by `delta` seconds, wrapping (or stopping, for a clip that doesn't loop)
    /// at the clip's end.
    pub fn advance(&mut self, clip: &EditorClip, delta: f32) {
        let duration = clip.duration();
        self.time += delta * clip.speed;
        if duration <= 0.0 {
            self.time = 0.0;
        } else if clip.looping {
            self.time = self.time.rem_euclid(duration);
        } else if self.time >= duration {
            self.time = duration;
            self.playing = false;
        }
    }
}

pub fn components(target: EditorTarget) -> usize {
    match target {
        EditorTarget::Scene(target) => target.components(),
        EditorTarget::Light(_, LightProperty::Intensity) => 1,
        EditorTarget::Light(_, LightProperty::Color) => 3,
    }
}

/// Each light's index in what save writes: `KHR_lights_punctual` for everything but area lights,
/// `blackbird_lights` for those, both in the editor's order.
pub fn light_indices(lights: &[SceneLight]) -> HashMap<u64, (bool, usize)> {
    let (mut punctual, mut area) = (0, 0);
    lights
        .iter()
        .map(|light| {
            let is_area = matches!(light.data, LightData::Area { .. });
            let counter = if is_area { &mut area } else { &mut punctual };
            *counter += 1;
            (light.id, (is_area, *counter - 1))
        })
        .collect()
}

/// The scene's clips: `blackbird_animations`, then a clip for each standard animation none of
/// them includes (so it stays listed and playable). A clip's standard channels become tracks,
/// paired with the clip's own track for the same node property if it has one (an override).
pub fn load_clips(doc: &Document, buffers: &[gltf_api::buffer::Data], lights: &[SceneLight]) -> Vec<EditorClip> {
    let punctual: HashMap<usize, u64> = lights
        .iter()
        .filter_map(|l| Some((doc.nodes().nth(l.node?)?.light()?.index(), l.id)))
        .collect();
    let area: Vec<u64> = lights.iter().filter(|l| matches!(l.data, LightData::Area { .. })).map(|l| l.id).collect();

    let data = read_clip_data(doc);
    let mut clips: Vec<EditorClip> = data
        .iter()
        .map(|clip| EditorClip {
            name: clip.name.clone(),
            looping: clip.looping,
            autoplay: clip.autoplay,
            speed: clip.speed,
            animation: clip.animation,
            tracks: clip
                .tracks
                .iter()
                .filter_map(|track| {
                    let (target, curve) = track.curve()?;
                    let target = match target {
                        Target::PunctualLight(i, p) => EditorTarget::Light(*punctual.get(&i)?, p),
                        Target::AreaLight(i, p) => EditorTarget::Light(*area.get(i)?, p),
                        other => EditorTarget::Scene(other),
                    };
                    Some(EditorTrack { target, curve, standard: None })
                })
                .collect(),
        })
        .collect();
    for animation in doc.animations() {
        if data.iter().any(|c| c.animation == Some(animation.index())) {
            continue;
        }
        let name = animation.name().map_or_else(|| format!("Animation {}", animation.index()), str::to_string);
        clips.push(EditorClip { animation: Some(animation.index()), ..EditorClip::new(name) });
    }
    for clip in &mut clips {
        let Some(animation) = clip.animation.and_then(|i| doc.animations().nth(i)) else { continue };
        for (target, curve) in standard_tracks(&animation, buffers) {
            let target = EditorTarget::Scene(target);
            match clip.tracks.iter_mut().find(|t| t.target == target) {
                Some(track) => track.standard = Some(curve),
                None => clip.tracks.push(EditorTrack { target, curve: curve.clone(), standard: Some(curve) }),
            }
        }
    }
    clips
}

/// `clips` as saved, with light tracks pointing at the lights' indices in `lights`' order. Tracks
/// of deleted lights are dropped, and so are standard channels nobody edited (the glTF animation
/// still plays them).
pub fn to_clip_data(clips: &[EditorClip], lights: &[SceneLight]) -> Vec<ClipData> {
    let indices = light_indices(lights);
    clips
        .iter()
        .map(|clip| ClipData {
            name: clip.name.clone(),
            looping: clip.looping,
            autoplay: clip.autoplay,
            speed: clip.speed,
            animation: clip.animation,
            tracks: clip
                .tracks
                .iter()
                .filter(|track| track.edited())
                .filter_map(|track| {
                    let target = match track.target {
                        EditorTarget::Scene(target) => target,
                        EditorTarget::Light(id, p) => match *indices.get(&id)? {
                            (true, i) => Target::AreaLight(i, p),
                            (false, i) => Target::PunctualLight(i, p),
                        },
                    };
                    Some(TrackData::from_curve(target, &track.curve))
                })
                .collect(),
        })
        .collect()
}

/// A readable name for a track's target.
pub fn label(state: &EditorState, target: EditorTarget) -> String {
    let node_name = |n: usize| {
        let name = state.node_edits.get(&n).map(|e| e.name.clone()).filter(|n| !n.is_empty());
        name.or_else(|| state.document.as_ref()?.nodes().nth(n)?.name().map(str::to_string))
            .unwrap_or_else(|| format!("Node {n}"))
    };
    let material_name = |m: usize| {
        state
            .document
            .as_ref()
            .and_then(|d| d.materials().nth(m)?.name().map(str::to_string))
            .unwrap_or_else(|| format!("Material {m}"))
    };
    let light = |p| match p {
        LightProperty::Intensity => "intensity",
        LightProperty::Color => "color",
    };
    match target {
        EditorTarget::Scene(Target::Translation(n)) => format!("{} · position", node_name(n)),
        EditorTarget::Scene(Target::Rotation(n)) => format!("{} · rotation", node_name(n)),
        EditorTarget::Scene(Target::Scale(n)) => format!("{} · scale", node_name(n)),
        EditorTarget::Scene(Target::EmissiveFactor(m)) => format!("{} · emissive color", material_name(m)),
        EditorTarget::Scene(Target::EmissiveStrength(m)) => format!("{} · emissive strength", material_name(m)),
        EditorTarget::Scene(Target::Particles(n, p)) => format!("{} · {}", node_name(n), p.key().replace('_', " ")),
        EditorTarget::Scene(Target::CameraFov(c)) => {
            camera_nodes(state, c).first().map_or_else(|| format!("Camera {c}"), |&n| node_name(n)) + " · field of view"
        }
        EditorTarget::Scene(Target::PunctualLight(i, p) | Target::AreaLight(i, p)) => {
            format!("Light {i} · {}", light(p))
        }
        EditorTarget::Light(id, p) => {
            let name = state
                .lights
                .iter()
                .position(|l| l.id == id)
                .map_or_else(|| "Deleted light".to_string(), |i| format!("Light {}", i + 1));
            format!("{name} · {}", light(p))
        }
    }
}

/// What `target` is set to outside the animation (the inspector's value), for a new key.
pub fn current_value(state: &EditorState, target: EditorTarget) -> Option<Value> {
    let v3 = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
    let v1 = |v: f32| [v, 0.0, 0.0, 0.0];
    match target {
        EditorTarget::Scene(target) => match target {
            Target::Translation(n) => Some(v3(state.node_edits.get(&n)?.translation)),
            Target::Rotation(n) => Some(state.node_edits.get(&n)?.rotation),
            Target::Scale(n) => Some(v3(state.node_edits.get(&n)?.scale)),
            Target::EmissiveFactor(m) => {
                let edited = state.node_edits.iter().find_map(|(&n, edit)| {
                    edit.emissive_factor.filter(|_| node_materials(state, n).contains(&m))
                });
                Some(v3(edited.unwrap_or_else(|| {
                    state.document.as_ref().and_then(|d| d.materials().nth(m)).map_or([0.0; 3], |m| m.emissive_factor())
                })))
            }
            Target::EmissiveStrength(m) => Some(v1(
                state.document.as_ref()?.materials().nth(m)?.emissive_strength().unwrap_or(1.0),
            )),
            Target::Particles(n, p) => {
                let data = state.node_edits.get(&n)?.particles.as_ref()?;
                Some(match p {
                    ParticleProperty::SpawnRate => v1(data.spawn_rate),
                    ParticleProperty::Lifetime => [data.lifetime[0], data.lifetime[1], 0.0, 0.0],
                    ParticleProperty::Size => [data.size[0], data.size[1], 0.0, 0.0],
                    ParticleProperty::Velocity => v3(data.velocity),
                    ParticleProperty::Turbulence => v1(data.turbulence),
                    ParticleProperty::Color => v3(data.color),
                    ParticleProperty::Opacity => v1(data.opacity),
                })
            }
            Target::CameraFov(c) => {
                let fov = camera_nodes(state, c).iter().find_map(|n| Some(state.node_edits.get(n)?.camera.as_ref()?.fov_y_deg))?;
                Some(v1(fov.to_radians()))
            }
            Target::PunctualLight(..) | Target::AreaLight(..) => None,
        },
        EditorTarget::Light(id, p) => {
            let (color, intensity) = match state.lights.iter().find(|l| l.id == id)?.data {
                LightData::Sky { color, intensity, .. }
                | LightData::Point { color, intensity, .. }
                | LightData::Spot { color, intensity, .. }
                | LightData::Area { color, intensity, .. } => (color, intensity),
            };
            Some(match p {
                LightProperty::Intensity => v1(intensity),
                LightProperty::Color => v3(color),
            })
        }
    }
}

/// The materials of node `n`'s mesh.
pub fn node_materials(state: &EditorState, n: usize) -> Vec<usize> {
    let Some(mesh) = state.document.as_ref().and_then(|d| d.nodes().nth(n)?.mesh()) else { return Vec::new() };
    let mut materials: Vec<usize> = mesh.primitives().filter_map(|p| p.material().index()).collect();
    materials.dedup();
    materials
}

/// The glTF camera node `n` shows.
pub fn node_camera(state: &EditorState, n: usize) -> Option<usize> {
    Some(state.document.as_ref()?.nodes().nth(n)?.camera()?.index())
}

/// The nodes showing glTF camera `camera`.
fn camera_nodes(state: &EditorState, camera: usize) -> Vec<usize> {
    let Some(doc) = &state.document else { return Vec::new() };
    doc.nodes().filter(|n| n.camera().is_some_and(|c| c.index() == camera)).map(|n| n.index()).collect()
}

/// The previewed clip, if the viewport shows one, and the playhead.
fn previewed(state: &EditorState) -> Option<(&EditorClip, f32)> {
    let clip = state.timeline.clip.filter(|_| state.timeline.preview && state.timeline.all.is_none())?;
    Some((state.clips.get(clip)?, state.timeline.time))
}

/// Camera node `n`'s world transform and field of view (radians) in the previewed clip at the
/// playhead, for Assume Camera View. `None` when nothing previews.
pub fn posed_camera(state: &EditorState, n: usize) -> Option<(Mat4, Option<f32>)> {
    let (clip, time) = previewed(state)?;
    let sample = |target| clip.tracks.iter().find(|t| t.target == EditorTarget::Scene(target)).map(|t| t.curve.sample(time));
    let local = |node: usize| {
        let (mut translation, mut rotation, mut scale) = match state.node_edits.get(&node) {
            Some(edit) => (edit.translation, edit.rotation, edit.scale),
            None => state
                .document
                .as_ref()
                .and_then(|d| d.nodes().nth(node))
                .map_or(([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]), |n| n.transform().decomposed()),
        };
        if let Some(v) = sample(Target::Translation(node)) {
            translation = [v[0], v[1], v[2]];
        }
        if let Some(v) = sample(Target::Rotation(node)) {
            rotation = v;
        }
        if let Some(v) = sample(Target::Scale(node)) {
            scale = [v[0], v[1], v[2]];
        }
        math::trs(translation, rotation, scale)
    };
    let mut world = local(n);
    let mut current = state.node_parents.get(&n).copied();
    while let Some(parent) = current {
        world = local(parent) * world;
        current = state.node_parents.get(&parent).copied();
    }
    let fov = node_camera(state, n).and_then(|c| sample(Target::CameraFov(c))).map(|v| v[0]);
    Some((world, fov))
}

/// How the editor shows a track's values: rotations as Euler degrees, fields of view in degrees,
/// everything else as stored. Returns the shown value and its component count.
pub fn to_display(target: EditorTarget, value: Value) -> (Value, usize) {
    match linear_scale(target) {
        Some(scale) => (value.map(|v| v * scale), components(target)),
        None => {
            let [x, y, z] = quat_to_euler(value);
            ([x, y, z, 0.0], 3)
        }
    }
}

/// Inverse of `to_display`. A rotation stays in the hemisphere of `previous`, so blending with
/// its neighbours doesn't take the long way round.
pub fn from_display(target: EditorTarget, shown: Value, previous: Value) -> Value {
    match linear_scale(target) {
        Some(scale) => shown.map(|v| v / scale),
        None => {
            let q = euler_to_quat([shown[0], shown[1], shown[2]]);
            let dot: f32 = (0..4).map(|c| q[c] * previous[c]).sum();
            if dot < 0.0 { q.map(|c| -c) } else { q }
        }
    }
}

/// The factor from stored to shown values (and slopes), or `None` for rotations, which aren't
/// shown linearly.
pub fn linear_scale(target: EditorTarget) -> Option<f32> {
    match target {
        EditorTarget::Scene(Target::Rotation(_)) => None,
        EditorTarget::Scene(Target::CameraFov(_)) => Some(1f32.to_degrees()),
        _ => Some(1.0),
    }
}

/// Everything about the selection that can be animated.
pub fn animatable(state: &EditorState) -> Vec<EditorTarget> {
    use crate::state::Selection;
    let mut targets = Vec::new();
    match state.selected {
        Some(Selection::GltfNode(n)) => {
            targets.extend([Target::Translation(n), Target::Rotation(n), Target::Scale(n)].map(EditorTarget::Scene));
            for m in node_materials(state, n) {
                targets.extend([Target::EmissiveFactor(m), Target::EmissiveStrength(m)].map(EditorTarget::Scene));
            }
            if state.node_edits.get(&n).is_some_and(|e| e.particles.is_some()) {
                targets.extend(ParticleProperty::ALL.map(|p| EditorTarget::Scene(Target::Particles(n, p))));
            }
            if let Some(camera) = node_camera(state, n) {
                targets.push(EditorTarget::Scene(Target::CameraFov(camera)));
            }
        }
        Some(Selection::Light(i)) => {
            if let Some(light) = state.lights.get(i) {
                targets.extend([LightProperty::Intensity, LightProperty::Color].map(|p| EditorTarget::Light(light.id, p)));
            }
        }
        Some(Selection::Object(_)) | None => {}
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn light(id: u64, area: bool) -> SceneLight {
        let data = if area {
            LightData::Area { position: [0.0; 3], right: [1.0, 0.0, 0.0], up: [0.0, 1.0, 0.0], color: [1.0; 3], intensity: 5.0 }
        } else {
            LightData::Point { position: [0.0; 3], color: [1.0; 3], intensity: 5.0, radius: 3.0, source_radius: 0.0 }
        };
        SceneLight { id, data, node: None }
    }

    #[test]
    fn light_tracks_follow_the_save_order() {
        let mut clip = EditorClip::new("Pulse".into());
        clip.set_key(EditorTarget::Light(7, LightProperty::Intensity), 0.0, [1.0, 0.0, 0.0, 0.0]);
        clip.set_key(EditorTarget::Light(9, LightProperty::Color), 0.0, [1.0, 0.5, 0.0, 0.0]);
        let lights = [light(3, false), light(9, true), light(7, false)];
        let data = to_clip_data(&[clip], &lights);
        let pointers: Vec<&str> = data[0].tracks.iter().map(|t| t.pointer.as_str()).collect();
        assert_eq!(pointers, ["/extensions/KHR_lights_punctual/lights/1/intensity", "/extras/blackbird_lights/0/color"]);
        // A deleted light's track is dropped.
        assert_eq!(to_clip_data(&data_clip(), &lights[..1]).first().map(|c| c.tracks.len()), Some(0));
    }

    fn data_clip() -> Vec<EditorClip> {
        let mut clip = EditorClip::new("c".into());
        clip.set_key(EditorTarget::Light(9, LightProperty::Intensity), 0.0, [1.0; 4]);
        vec![clip]
    }

    #[test]
    fn loading_and_saving_keeps_light_pointers() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_lights_punctual"],
            "extensions": { "KHR_lights_punctual": { "lights": [
                { "type": "point", "intensity": 1.0 }, { "type": "spot", "intensity": 2.0, "spot": {} } ] } },
            "scene": 0,
            "scenes": [{ "nodes": [0, 1] }],
            "nodes": [
                { "name": "Spot", "extensions": { "KHR_lights_punctual": { "light": 1 } } },
                { "name": "Point", "extensions": { "KHR_lights_punctual": { "light": 0 } } }
            ],
            "extras": { "blackbird_animations": [{ "name": "Blink", "loop": false, "tracks": [{
                "pointer": "/extensions/KHR_lights_punctual/lights/1/intensity",
                "keys": [{ "time": 0.0, "value": 2.0, "interpolation": "ease_in" }, { "time": 1.0, "value": 0.0 }]
            }] }] }
        }"#;
        let doc = gltf_api::Gltf::from_slice(json.as_bytes()).unwrap().document;
        let lights: Vec<SceneLight> = engine::gltf::lights::read_lights(&doc)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, (node, data))| SceneLight { id: i as u64 + 10, data, node })
            .collect();
        let clips = load_clips(&doc, &[], &lights);
        let spot = lights.iter().find(|l| l.node == Some(0)).unwrap().id;
        assert_eq!(clips[0].tracks[0].target, EditorTarget::Light(spot, LightProperty::Intensity));
        assert!(!clips[0].looping);

        // Saved, the spot gets the index of its place among the editor's punctual lights.
        let saved = to_clip_data(&clips, &lights);
        let index = lights.iter().position(|l| l.id == spot).unwrap();
        assert_eq!(saved[0].tracks[0].pointer, format!("/extensions/KHR_lights_punctual/lights/{index}/intensity"));
        assert_eq!(saved[0].tracks[0].keys[0].interpolation, engine::gltf::animation::InterpolationData::EaseIn);
    }

    #[test]
    fn display_round_trips_and_keeps_the_rotation_hemisphere() {
        let fov = EditorTarget::Scene(Target::CameraFov(0));
        let (shown, n) = to_display(fov, [0.5, 0.0, 0.0, 0.0]);
        assert_eq!(n, 1);
        assert!((shown[0] - 0.5f32.to_degrees()).abs() < 1e-4);
        assert!((from_display(fov, shown, [0.0; 4])[0] - 0.5).abs() < 1e-6);

        let rotation = EditorTarget::Scene(Target::Rotation(0));
        let previous = euler_to_quat([10.0, 20.0, 30.0]).map(|c| -c);
        let (shown, n) = to_display(rotation, previous);
        assert_eq!(n, 3);
        let back = from_display(rotation, shown, previous);
        for c in 0..4 {
            assert!((back[c] - previous[c]).abs() < 1e-4, "{back:?} vs {previous:?}");
        }
    }

    /// A glTF animation moving node 0 from the origin to y = 2 over a second.
    const RISE: &str = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 32, "uri": "data:application/octet-stream;base64,AAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAA=" }],
        "bufferViews": [{ "buffer": 0, "byteLength": 8 }, { "buffer": 0, "byteOffset": 8, "byteLength": 24 }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1] },
            { "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3" }],
        "animations": [{ "name": "Rise", "channels": [{ "sampler": 0, "target": { "node": 0, "path": "translation" } }],
            "samplers": [{ "input": 0, "output": 1 }] }],
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "name": "Box" }]
    }"#;

    #[test]
    fn standard_channels_are_saved_only_once_edited() {
        let (doc, buffers, _) = gltf_api::import_slice(RISE.as_bytes()).unwrap();
        let mut clips = load_clips(&doc, &buffers, &[]);
        let target = EditorTarget::Scene(Target::Translation(0));
        assert_eq!(clips[0].tracks.len(), 1);
        assert_eq!(clips[0].tracks[0].target, target);
        assert_eq!(clips[0].tracks[0].curve.keys[1].value[1], 2.0);
        assert!(to_clip_data(&clips, &[])[0].tracks.is_empty());

        clips[0].set_key(target, 1.0, [0.0, 3.0, 0.0, 0.0]);
        let saved = to_clip_data(&clips, &[]);
        assert_eq!((saved[0].animation, saved[0].tracks.len()), (Some(0), 1));

        // Reloaded, the override pairs up with its channel instead of adding a second track.
        let reloaded = load_clips_from(&doc, &buffers, &saved);
        assert_eq!(reloaded[0].tracks.len(), 1);
        assert!(reloaded[0].tracks[0].edited());
    }

    fn load_clips_from(doc: &Document, buffers: &[gltf_api::buffer::Data], data: &[ClipData]) -> Vec<EditorClip> {
        let mut json: serde_json::Value = serde_json::from_str(RISE).unwrap();
        json["extras"] = serde_json::json!({ engine::gltf::animation::EXTRAS_KEY: data });
        let doc2 = gltf_api::Gltf::from_slice(json.to_string().as_bytes()).unwrap().document;
        assert_eq!(doc2.animations().count(), doc.animations().count());
        load_clips(&doc2, buffers, &[])
    }

    #[test]
    fn set_key_replaces_a_key_at_the_same_time() {
        let mut clip = EditorClip::new("c".into());
        let target = EditorTarget::Scene(Target::EmissiveStrength(0));
        clip.set_key(target, 1.0, [2.0, 0.0, 0.0, 0.0]);
        clip.set_key(target, 0.0, [1.0, 0.0, 0.0, 0.0]);
        clip.set_key(target, 1.0005, [3.0, 0.0, 0.0, 0.0]);
        let keys = &clip.tracks[0].curve.keys;
        assert_eq!(keys.len(), 2);
        assert_eq!((keys[0].time, keys[1].value[0]), (0.0, 3.0));
    }
}

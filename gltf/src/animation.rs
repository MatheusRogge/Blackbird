//! Keyframe animation from glTF: standard `animations` (node translation / rotation / scale) and
//! our own clips in the root `extras.blackbird_animations`, whose tracks address any animatable
//! property with a `KHR_animation_pointer`-style path and keep their keys inline. `GltfScene` plays
//! them (`animate`), writing the sampled values into the entities it spawned.

use engine_core::{
    animation::{Clip, Curve, Interpolation, Key, Playback, Value, Wrap},
    entity::EntityKey,
    world::World,
};
use rendering::{
    light::{AreaLight, PointLight, SkyLight, SpotLight},
    mesh::{Mat4, Mesh},
    particles::ParticleEmitter,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use ultraviolet::{Vec3, Vec4};

use crate::GltfScene;

pub const EXTRAS_KEY: &str = "blackbird_animations";

/// A property a track animates, parsed from its pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Translation(usize),
    Rotation(usize),
    Scale(usize),
    EmissiveFactor(usize),
    EmissiveStrength(usize),
    /// A `KHR_lights_punctual` light, by its index in the extension's `lights`.
    PunctualLight(usize, LightProperty),
    /// An area light, by its index in `extras.blackbird_lights`.
    AreaLight(usize, LightProperty),
    /// The `blackbird_particles` emitter on a node.
    Particles(usize, ParticleProperty),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LightProperty {
    Intensity,
    Color,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParticleProperty {
    SpawnRate,
    Lifetime,
    Size,
    Velocity,
    Turbulence,
    Color,
    Opacity,
}

impl ParticleProperty {
    pub const ALL: [Self; 7] =
        [Self::SpawnRate, Self::Lifetime, Self::Size, Self::Velocity, Self::Turbulence, Self::Color, Self::Opacity];

    /// The field's name in `blackbird_particles`.
    pub fn key(self) -> &'static str {
        match self {
            Self::SpawnRate => "spawn_rate",
            Self::Lifetime => "lifetime",
            Self::Size => "size",
            Self::Velocity => "velocity",
            Self::Turbulence => "turbulence",
            Self::Color => "color",
            Self::Opacity => "opacity",
        }
    }
}

impl Target {
    pub fn parse(pointer: &str) -> Option<Self> {
        let parts: Vec<&str> = pointer.strip_prefix('/')?.split('/').collect();
        let index = |s: &str| s.parse::<usize>().ok();
        let light = |s: &str| match s {
            "intensity" => Some(LightProperty::Intensity),
            "color" => Some(LightProperty::Color),
            _ => None,
        };
        Some(match parts.as_slice() {
            ["nodes", n, "translation"] => Self::Translation(index(n)?),
            ["nodes", n, "rotation"] => Self::Rotation(index(n)?),
            ["nodes", n, "scale"] => Self::Scale(index(n)?),
            ["materials", m, "emissiveFactor"] => Self::EmissiveFactor(index(m)?),
            ["materials", m, "extensions", "KHR_materials_emissive_strength", "emissiveStrength"] => {
                Self::EmissiveStrength(index(m)?)
            }
            ["extensions", "KHR_lights_punctual", "lights", l, p] => Self::PunctualLight(index(l)?, light(p)?),
            ["extras", "blackbird_lights", l, p] => Self::AreaLight(index(l)?, light(p)?),
            ["nodes", n, "extras", "blackbird_particles", p] => {
                Self::Particles(index(n)?, *ParticleProperty::ALL.iter().find(|q| q.key() == *p)?)
            }
            _ => return None,
        })
    }

    pub fn pointer(self) -> String {
        let light = |p| match p {
            LightProperty::Intensity => "intensity",
            LightProperty::Color => "color",
        };
        match self {
            Self::Translation(n) => format!("/nodes/{n}/translation"),
            Self::Rotation(n) => format!("/nodes/{n}/rotation"),
            Self::Scale(n) => format!("/nodes/{n}/scale"),
            Self::EmissiveFactor(m) => format!("/materials/{m}/emissiveFactor"),
            Self::EmissiveStrength(m) => {
                format!("/materials/{m}/extensions/KHR_materials_emissive_strength/emissiveStrength")
            }
            Self::PunctualLight(l, p) => format!("/extensions/KHR_lights_punctual/lights/{l}/{}", light(p)),
            Self::AreaLight(l, p) => format!("/extras/blackbird_lights/{l}/{}", light(p)),
            Self::Particles(n, p) => format!("/nodes/{n}/extras/blackbird_particles/{}", p.key()),
        }
    }

    pub fn components(self) -> usize {
        match self {
            Self::Rotation(_) => 4,
            Self::Translation(_) | Self::Scale(_) | Self::EmissiveFactor(_) => 3,
            Self::EmissiveStrength(_) => 1,
            Self::PunctualLight(_, p) | Self::AreaLight(_, p) => match p {
                LightProperty::Intensity => 1,
                LightProperty::Color => 3,
            },
            Self::Particles(_, p) => match p {
                ParticleProperty::SpawnRate | ParticleProperty::Turbulence | ParticleProperty::Opacity => 1,
                ParticleProperty::Lifetime | ParticleProperty::Size => 2,
                ParticleProperty::Velocity | ParticleProperty::Color => 3,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationData {
    Step,
    #[default]
    Linear,
    Smooth,
    EaseIn,
    EaseOut,
    EaseInOut,
    Cubic,
}

impl From<InterpolationData> for Interpolation {
    fn from(value: InterpolationData) -> Self {
        match value {
            InterpolationData::Step => Self::Step,
            InterpolationData::Linear => Self::Linear,
            InterpolationData::Smooth => Self::Smooth,
            InterpolationData::EaseIn => Self::EaseIn,
            InterpolationData::EaseOut => Self::EaseOut,
            InterpolationData::EaseInOut => Self::EaseInOut,
            InterpolationData::Cubic => Self::Cubic,
        }
    }
}

impl From<Interpolation> for InterpolationData {
    fn from(value: Interpolation) -> Self {
        match value {
            Interpolation::Step => Self::Step,
            Interpolation::Linear => Self::Linear,
            Interpolation::Smooth => Self::Smooth,
            Interpolation::EaseIn => Self::EaseIn,
            Interpolation::EaseOut => Self::EaseOut,
            Interpolation::EaseInOut => Self::EaseInOut,
            Interpolation::Cubic => Self::Cubic,
        }
    }
}

/// A number, or a vector for multi-component properties.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Values {
    Scalar(f32),
    Vector(Vec<f32>),
}

impl Values {
    fn value(&self) -> Value {
        match self {
            Self::Scalar(v) => [*v, 0.0, 0.0, 0.0],
            Self::Vector(v) => std::array::from_fn(|c| v.get(c).copied().unwrap_or(0.0)),
        }
    }

    fn from_value(value: Value, components: usize) -> Self {
        if components == 1 { Self::Scalar(value[0]) } else { Self::Vector(value[..components].to_vec()) }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeyData {
    pub time: f32,
    pub value: Values,
    #[serde(default)]
    pub interpolation: InterpolationData,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_tangent: Option<Values>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_tangent: Option<Values>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackData {
    pub pointer: String,
    pub keys: Vec<KeyData>,
}

impl TrackData {
    /// The track's target and curve, or `None` if the pointer isn't animatable.
    pub fn curve(&self) -> Option<(Target, Curve)> {
        let target = Target::parse(&self.pointer)?;
        let keys = self
            .keys
            .iter()
            .map(|k| Key {
                time: k.time,
                value: k.value.value(),
                interpolation: k.interpolation.into(),
                in_tangent: k.in_tangent.as_ref().map_or([0.0; 4], Values::value),
                out_tangent: k.out_tangent.as_ref().map_or([0.0; 4], Values::value),
            })
            .collect();
        let curve = match target {
            Target::Rotation(_) => Curve::rotation(keys),
            _ => Curve::new(target.components(), keys),
        };
        Some((target, curve))
    }

    pub fn from_curve(target: Target, curve: &Curve) -> Self {
        let n = target.components();
        let tangent = |t: Value| (t != [0.0; 4]).then(|| Values::from_value(t, n));
        Self {
            pointer: target.pointer(),
            keys: curve
                .keys
                .iter()
                .map(|k| KeyData {
                    time: k.time,
                    value: Values::from_value(k.value, n),
                    interpolation: k.interpolation.into(),
                    in_tangent: tangent(k.in_tangent),
                    out_tangent: tangent(k.out_tangent),
                })
                .collect(),
        }
    }
}

fn yes() -> bool {
    true
}

fn one() -> f32 {
    1.0
}

/// One entry of `blackbird_animations`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipData {
    pub name: String,
    #[serde(rename = "loop", default = "yes")]
    pub looping: bool,
    #[serde(default = "yes")]
    pub autoplay: bool,
    #[serde(default = "one")]
    pub speed: f32,
    /// A standard glTF animation whose channels this clip also plays.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation: Option<usize>,
    #[serde(default)]
    pub tracks: Vec<TrackData>,
}

/// The clips in `doc`'s root `extras`. A malformed list is logged and skipped.
pub fn read_clip_data(doc: &gltf::Document) -> Vec<ClipData> {
    let Some(raw) = &doc.as_json().extras else { return Vec::new() };
    let Ok(Json::Object(mut extras)) = serde_json::from_str::<Json>(raw.get()) else { return Vec::new() };
    let Some(value) = extras.remove(EXTRAS_KEY) else { return Vec::new() };
    serde_json::from_value(value).unwrap_or_else(|err| {
        log::warn!("invalid {EXTRAS_KEY}: {err}");
        Vec::new()
    })
}

/// A clip ready to play.
pub struct PlayingClip {
    pub clip: Clip<Target>,
    pub playback: Playback,
}

/// Every clip of `doc`: our `blackbird_animations` (plus the standard channels they include), and
/// each standard animation no clip includes, looping.
pub fn load_clips(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<PlayingClip> {
    let data = read_clip_data(doc);
    let mut clips = Vec::new();
    for clip in &data {
        let mut tracks: Vec<(Target, Curve)> = clip.tracks.iter().filter_map(TrackData::curve).collect();
        if let Some(animation) = clip.animation.and_then(|i| doc.animations().nth(i)) {
            tracks.extend(standard_tracks(&animation, buffers));
        }
        clips.push(PlayingClip {
            clip: Clip { name: clip.name.clone(), tracks, wrap: if clip.looping { Wrap::Loop } else { Wrap::Once } },
            playback: Playback { elapsed: 0.0, speed: clip.speed, playing: clip.autoplay },
        });
    }
    for animation in doc.animations() {
        if data.iter().any(|c| c.animation == Some(animation.index())) {
            continue;
        }
        clips.push(PlayingClip {
            clip: Clip {
                name: animation.name().map_or_else(|| format!("Animation {}", animation.index()), str::to_string),
                tracks: standard_tracks(&animation, buffers),
                wrap: Wrap::Loop,
            },
            playback: Playback::default(),
        });
    }
    clips
}

/// The translation / rotation / scale channels of a standard glTF animation.
fn standard_tracks(animation: &gltf::Animation<'_>, buffers: &[gltf::buffer::Data]) -> Vec<(Target, Curve)> {
    use gltf::animation::{Interpolation as Gltf, util::ReadOutputs};
    let mut tracks = Vec::new();
    for channel in animation.channels() {
        let reader = channel.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
        let (Some(inputs), Some(outputs)) = (reader.read_inputs(), reader.read_outputs()) else { continue };
        let times: Vec<f32> = inputs.collect();
        let node = channel.target().node().index();
        let (target, values): (Target, Vec<Value>) = match outputs {
            ReadOutputs::Translations(it) => (Target::Translation(node), it.map(|[x, y, z]| [x, y, z, 0.0]).collect()),
            ReadOutputs::Rotations(it) => (Target::Rotation(node), it.into_f32().collect()),
            ReadOutputs::Scales(it) => (Target::Scale(node), it.map(|[x, y, z]| [x, y, z, 0.0]).collect()),
            ReadOutputs::MorphTargetWeights(_) => continue,
        };
        let interpolation = channel.sampler().interpolation();
        let keys: Vec<Key> = match interpolation {
            Gltf::CubicSpline => times
                .iter()
                .zip(values.as_chunks::<3>().0)
                .map(|(&time, v)| Key { time, value: v[1], interpolation: Interpolation::Cubic, in_tangent: v[0], out_tangent: v[2] })
                .collect(),
            Gltf::Step | Gltf::Linear => {
                let preset = if interpolation == Gltf::Step { Interpolation::Step } else { Interpolation::Linear };
                times.iter().zip(&values).map(|(&time, &value)| Key::new(time, value, preset)).collect()
            }
        };
        let curve = match target {
            Target::Rotation(_) => Curve::rotation(keys),
            _ => Curve::new(3, keys),
        };
        tracks.push((target, curve));
    }
    tracks
}

/// A node's translation, rotation (xyzw) and scale.
pub type Trs = ([f32; 3], [f32; 4], [f32; 3]);

impl GltfScene {
    /// The scene's clips, in `blackbird_animations` order, then the standard animations no clip includes.
    pub fn clips(&self) -> &[PlayingClip] {
        &self.clips
    }

    pub fn clips_mut(&mut self) -> &mut [PlayingClip] {
        &mut self.clips
    }

    /// Hands over the light entities spawned from `lights::read_lights`, in its order (each with its
    /// node, `None` for `blackbird_lights`), so clips can animate them and their nodes can move them.
    pub fn bind_lights(&mut self, lights: impl IntoIterator<Item = (Option<usize>, EntityKey)>) {
        let mut area = 0;
        for (node, key) in lights {
            match node {
                Some(node) => {
                    if let Some(light) = self.asset.document.nodes().nth(node).and_then(|n| n.light()) {
                        self.punctual_lights.insert(light.index(), key);
                    }
                    self.node_lights.insert(node, key);
                }
                None => {
                    self.area_lights.insert(area, key);
                    area += 1;
                }
            }
        }
    }

    /// Advances every playing clip by `delta` seconds and writes the animated values into the
    /// scene's entities. Clips that are neither playing nor moved from their start are left alone.
    pub fn animate(&mut self, world: &mut World, delta: f32) {
        let mut samples = Vec::new();
        for playing in &mut self.clips {
            playing.playback.advance(delta);
            if playing.playback.playing || playing.playback.elapsed != 0.0 {
                let time = playing.clip.local_time(playing.playback.elapsed);
                samples.extend(playing.clip.sample(time).map(|(target, value)| (*target, value)));
            }
        }
        if samples.is_empty() {
            return;
        }
        let mut moved = false;
        for (target, value) in samples {
            moved |= self.apply(world, target, value);
        }
        if moved {
            self.update_transforms(world);
        }
    }

    /// Writes one sampled value. Returns whether it moved a node.
    fn apply(&mut self, world: &mut World, target: Target, v: Value) -> bool {
        let vec3 = Vec3::new(v[0], v[1], v[2]);
        match target {
            Target::Translation(node) | Target::Rotation(node) | Target::Scale(node) => {
                let Some(gltf_node) = self.asset.document.nodes().nth(node) else { return false };
                let trs = self.animated_nodes.entry(node).or_insert_with(|| gltf_node.transform().decomposed());
                match target {
                    Target::Translation(_) => trs.0 = [v[0], v[1], v[2]],
                    Target::Rotation(_) => trs.1 = v,
                    _ => trs.2 = [v[0], v[1], v[2]],
                }
                return true;
            }
            Target::EmissiveFactor(material) | Target::EmissiveStrength(material) => {
                for &key in self.material_meshes.get(&material).into_iter().flatten() {
                    let Some(mesh) = world.get_entity_mut::<Mesh>(key) else { continue };
                    match target {
                        Target::EmissiveFactor(_) => mesh.material.emissive = [v[0], v[1], v[2]],
                        _ => mesh.material.emissive_strength = v[0].max(0.0),
                    }
                }
            }
            Target::PunctualLight(index, property) | Target::AreaLight(index, property) => {
                let lights = if matches!(target, Target::PunctualLight(..)) { &self.punctual_lights } else { &self.area_lights };
                if let Some(&key) = lights.get(&index) {
                    set_light(world, key, property, v[0].max(0.0), vec3);
                }
            }
            Target::Particles(node, property) => {
                let Some(emitter) = self.node_emitters.get(&node).and_then(|&k| world.get_entity_mut::<ParticleEmitter>(k))
                else {
                    return false;
                };
                let sorted = |a: f32, b: f32| [a.min(b).max(0.0), a.max(b).max(0.0)];
                match property {
                    ParticleProperty::SpawnRate => emitter.spawn_rate = v[0].max(0.0),
                    ParticleProperty::Lifetime => emitter.lifetime = sorted(v[0], v[1]),
                    ParticleProperty::Size => emitter.size = sorted(v[0], v[1]),
                    ParticleProperty::Velocity => emitter.velocity = vec3,
                    ParticleProperty::Turbulence => emitter.turbulence = v[0].max(0.0),
                    ParticleProperty::Color => emitter.color = vec3,
                    ParticleProperty::Opacity => emitter.opacity = v[0].max(0.0),
                }
            }
        }
        false
    }

    /// Re-derives the world transform of every node at or under an animated one and moves its
    /// meshes, emitters and lights.
    fn update_transforms(&self, world: &mut World) {
        let document = &self.asset.document;
        let Some(scene) = document.default_scene().or_else(|| document.scenes().next()) else { return };
        let mut stack: Vec<(gltf::Node<'_>, Mat4, bool)> = scene.nodes().map(|n| (n, self.root, false)).collect();
        while let Some((node, parent, parent_moved)) = stack.pop() {
            let animated = self.animated_nodes.get(&node.index());
            let local = match animated {
                Some(&(translation, rotation, scale)) => {
                    gltf::scene::Transform::Decomposed { translation, rotation, scale }.matrix()
                }
                None => node.transform().matrix(),
            };
            let transform = parent * Mat4::from(local);
            let moved = parent_moved || animated.is_some();
            stack.extend(node.children().map(|child| (child, transform, moved)));
            if moved {
                self.place_node(world, node.index(), transform);
            }
        }
    }

    fn place_node(&self, world: &mut World, node: usize, transform: Mat4) {
        for &key in self.node_entities.get(&node).into_iter().flatten() {
            if let Some(mesh) = world.get_entity_mut::<Mesh>(key) {
                mesh.transform = transform;
            }
        }
        if let Some(emitter) = self.node_emitters.get(&node).and_then(|&k| world.get_entity_mut::<ParticleEmitter>(k)) {
            emitter.transform = transform;
        }
        let Some(&light) = self.node_lights.get(&node) else { return };
        let position = transform.cols[3].xyz();
        let direction = (transform * Vec4::new(0.0, 0.0, -1.0, 0.0)).xyz().normalized();
        if let Some(l) = world.get_entity_mut::<PointLight>(light) {
            l.position = position;
        } else if let Some(l) = world.get_entity_mut::<SpotLight>(light) {
            l.position = position;
            l.direction = direction;
        } else if let Some(l) = world.get_entity_mut::<SkyLight>(light) {
            l.direction = direction;
        }
    }
}

fn set_light(world: &mut World, key: EntityKey, property: LightProperty, intensity: f32, color: Vec3) {
    macro_rules! set {
        ($light:expr) => {{
            match property {
                LightProperty::Intensity => $light.intensity = intensity,
                LightProperty::Color => $light.color = color,
            }
            return;
        }};
    }
    if let Some(l) = world.get_entity_mut::<PointLight>(key) {
        set!(l)
    }
    if let Some(l) = world.get_entity_mut::<SpotLight>(key) {
        set!(l)
    }
    if let Some(l) = world.get_entity_mut::<AreaLight>(key) {
        set!(l)
    }
    if let Some(l) = world.get_entity_mut::<SkyLight>(key) {
        set!(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointers_round_trip() {
        for target in [
            Target::Translation(3),
            Target::Rotation(1),
            Target::EmissiveStrength(6),
            Target::EmissiveFactor(2),
            Target::PunctualLight(0, LightProperty::Color),
            Target::AreaLight(4, LightProperty::Intensity),
            Target::Particles(29, ParticleProperty::SpawnRate),
        ] {
            assert_eq!(Target::parse(&target.pointer()), Some(target));
        }
        assert_eq!(Target::parse("/materials/1/pbrMetallicRoughness/baseColorFactor"), None);
    }

    #[test]
    fn reads_a_flicker_clip() {
        let json = r#"[{ "name": "Flicker", "tracks": [{
            "pointer": "/materials/6/extensions/KHR_materials_emissive_strength/emissiveStrength",
            "keys": [
                { "time": 0.0, "value": 40.0, "interpolation": "smooth" },
                { "time": 1.0, "value": 2.0, "interpolation": "step" },
                { "time": 1.2, "value": 40.0 }
            ] }] }]"#;
        let clips: Vec<ClipData> = serde_json::from_str(json).expect("valid clips");
        assert!(clips[0].looping && clips[0].autoplay);
        let (target, curve) = clips[0].tracks[0].curve().expect("animatable");
        assert_eq!(target, Target::EmissiveStrength(6));
        assert_eq!(curve.sample(1.1)[0], 2.0);
        assert_eq!(TrackData::from_curve(target, &curve).keys[1].interpolation, InterpolationData::Step);
    }
}

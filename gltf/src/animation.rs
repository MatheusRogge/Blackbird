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
    camera::Camera,
    light::{AreaLight, PointLight, SkyLight, SpotLight},
    mesh::{Mat4, Mesh},
    particles::ParticleEmitter,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::collections::{HashMap, HashSet};
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
    /// A perspective camera's vertical field of view, in radians, by its index in `cameras`.
    CameraFov(usize),
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
            ["cameras", c, "perspective", "yfov"] => Self::CameraFov(index(c)?),
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
            Self::CameraFov(c) => format!("/cameras/{c}/perspective/yfov"),
        }
    }

    pub fn components(self) -> usize {
        match self {
            Self::Rotation(_) => 4,
            Self::Translation(_) | Self::Scale(_) | Self::EmissiveFactor(_) => 3,
            Self::EmissiveStrength(_) | Self::CameraFov(_) => 1,
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
    build_clips(doc, buffers, &read_clip_data(doc))
}

/// `data`'s clips (plus the standard channels they include, except those a track of the clip
/// overrides), then each standard animation of `doc` none of them includes.
pub fn build_clips(doc: &gltf::Document, buffers: &[gltf::buffer::Data], data: &[ClipData]) -> Vec<PlayingClip> {
    let mut clips = Vec::new();
    for clip in data {
        let mut tracks: Vec<(Target, Curve)> = clip.tracks.iter().filter_map(TrackData::curve).collect();
        if let Some(animation) = clip.animation.and_then(|i| doc.animations().nth(i)) {
            let overridden: HashSet<Target> = tracks.iter().map(|(t, _)| *t).collect();
            tracks.extend(standard_tracks(&animation, buffers).into_iter().filter(|(t, _)| !overridden.contains(t)));
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
pub fn standard_tracks(animation: &gltf::Animation<'_>, buffers: &[gltf::buffer::Data]) -> Vec<(Target, Curve)> {
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

    /// Replaces the clips with `data`'s (`build_clips`), e.g. the editor's edited ones.
    pub fn set_clip_data(&mut self, data: &[ClipData]) {
        self.clips = build_clips(&self.asset.document, &self.asset.buffers, data);
    }

    /// Sets the light entities light targets resolve to, by `KHR_lights_punctual` index and by
    /// `blackbird_lights` index, and the punctual lights that follow each node, replacing what
    /// `bind_lights` set.
    pub fn set_light_bindings(
        &mut self,
        punctual: HashMap<usize, EntityKey>,
        area: HashMap<usize, EntityKey>,
        nodes: HashMap<usize, EntityKey>,
    ) {
        self.punctual_lights = punctual;
        self.area_lights = area;
        self.node_lights = nodes;
    }

    /// Applies clip `index` at clip-local `time`, without touching any clip's playback.
    pub fn pose(&mut self, world: &mut World, index: usize, time: f32) {
        let Some(playing) = self.clips.get(index) else { return };
        let samples: Vec<(Target, Value)> = playing.clip.sample(time).map(|(t, v)| (*t, v)).collect();
        let mut moved = false;
        for (target, value) in samples {
            moved |= self.apply(world, target, value);
        }
        if moved {
            self.update_transforms(world, &HashSet::new());
        }
    }

    /// Writes back every value animation has changed since the last reset, and puts animated
    /// nodes back at their glTF transforms.
    pub fn reset_animation(&mut self, world: &mut World) {
        for (target, value) in std::mem::take(&mut self.originals) {
            self.write(world, target, value);
        }
        if !self.animated_nodes.is_empty() {
            let moved: HashSet<usize> = self.animated_nodes.drain().map(|(node, _)| node).collect();
            self.update_transforms(world, &moved);
        }
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
            self.update_transforms(world, &HashSet::new());
        }
    }

    /// Writes one sampled value, first remembering what it replaces (`reset_animation`). Returns
    /// whether it moved a node.
    fn apply(&mut self, world: &mut World, target: Target, v: Value) -> bool {
        if !matches!(target, Target::Translation(_) | Target::Rotation(_) | Target::Scale(_))
            && !self.originals.contains_key(&target)
            && let Some(current) = self.current(world, target)
        {
            self.originals.insert(target, current);
        }
        self.write(world, target, v)
    }

    /// The value `target` has now, for the properties `write` sets on entities.
    fn current(&self, world: &World, target: Target) -> Option<Value> {
        let vec3 = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let scalar = |v: f32| [v, 0.0, 0.0, 0.0];
        match target {
            Target::Translation(_) | Target::Rotation(_) | Target::Scale(_) => None,
            Target::EmissiveFactor(m) | Target::EmissiveStrength(m) => {
                let mesh = self.material_meshes.get(&m)?.iter().find_map(|&k| world.get_entity::<Mesh>(k))?;
                Some(match target {
                    Target::EmissiveFactor(_) => {
                        let [r, g, b] = mesh.material.emissive;
                        [r, g, b, 0.0]
                    }
                    _ => scalar(mesh.material.emissive_strength),
                })
            }
            Target::PunctualLight(index, property) | Target::AreaLight(index, property) => {
                let lights = if matches!(target, Target::PunctualLight(..)) { &self.punctual_lights } else { &self.area_lights };
                let key = *lights.get(&index)?;
                let (intensity, color) = if let Some(l) = world.get_entity::<PointLight>(key) {
                    (l.intensity, l.color)
                } else if let Some(l) = world.get_entity::<SpotLight>(key) {
                    (l.intensity, l.color)
                } else if let Some(l) = world.get_entity::<AreaLight>(key) {
                    (l.intensity, l.color)
                } else {
                    let l = world.get_entity::<SkyLight>(key)?;
                    (l.intensity, l.color)
                };
                Some(match property {
                    LightProperty::Intensity => scalar(intensity),
                    LightProperty::Color => vec3(color),
                })
            }
            Target::CameraFov(camera) => {
                let key = self.camera_entities(camera).next()?;
                Some(scalar(world.get_entity::<Camera>(key)?.fovy))
            }
            Target::Particles(node, property) => {
                let e = world.get_entity::<ParticleEmitter>(*self.node_emitters.get(&node)?)?;
                Some(match property {
                    ParticleProperty::SpawnRate => scalar(e.spawn_rate),
                    ParticleProperty::Lifetime => [e.lifetime[0], e.lifetime[1], 0.0, 0.0],
                    ParticleProperty::Size => [e.size[0], e.size[1], 0.0, 0.0],
                    ParticleProperty::Velocity => vec3(e.velocity),
                    ParticleProperty::Turbulence => scalar(e.turbulence),
                    ParticleProperty::Color => vec3(e.color),
                    ParticleProperty::Opacity => scalar(e.opacity),
                })
            }
        }
    }

    /// Writes one value. Returns whether it moved a node.
    fn write(&mut self, world: &mut World, target: Target, v: Value) -> bool {
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
            Target::CameraFov(camera) => {
                let keys: Vec<EntityKey> = self.camera_entities(camera).collect();
                for key in keys {
                    if let Some(c) = world.get_entity_mut::<Camera>(key) {
                        c.fovy = v[0].clamp(0.01, 3.1);
                    }
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

    /// Re-derives the world transform of every node at or under an animated one (or one in
    /// `also`) and moves its meshes, emitters and lights.
    fn update_transforms(&self, world: &mut World, also: &HashSet<usize>) {
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
            let moved = parent_moved || animated.is_some() || also.contains(&node.index());
            stack.extend(node.children().map(|child| (child, transform, moved)));
            if moved {
                self.place_node(world, node.index(), transform);
            }
        }
    }

    /// The entities of the nodes showing glTF camera `camera`.
    fn camera_entities(&self, camera: usize) -> impl Iterator<Item = EntityKey> + '_ {
        self.asset
            .document
            .nodes()
            .filter(move |n| n.camera().is_some_and(|c| c.index() == camera))
            .filter_map(|n| self.node_cameras.get(&n.index()).copied())
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
        if let Some(camera) = self.node_cameras.get(&node).and_then(|&k| world.get_entity_mut::<Camera>(k)) {
            camera.eye = transform.cols[3].xyz();
            camera.target = camera.eye + (transform * Vec4::new(0.0, 0.0, -1.0, 0.0)).xyz().normalized();
            camera.up = (transform * Vec4::new(0.0, 1.0, 0.0, 0.0)).xyz().normalized();
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
    fn pose_and_reset_drive_a_material() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_materials_emissive_strength"],
            "buffers": [{ "byteLength": 36, "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA" }],
            "bufferViews": [{ "buffer": 0, "byteLength": 36 }],
            "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                "min": [0, 0, 0], "max": [1, 1, 0] }],
            "materials": [{ "name": "Lens", "emissiveFactor": [1, 1, 1],
                "extensions": { "KHR_materials_emissive_strength": { "emissiveStrength": 40.0 } } }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "material": 0 }] }],
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "extras": { "blackbird_animations": [{ "name": "Dim", "autoplay": false, "tracks": [{
                "pointer": "/materials/0/extensions/KHR_materials_emissive_strength/emissiveStrength",
                "keys": [{ "time": 0.0, "value": 40.0 }, { "time": 1.0, "value": 0.0 }] }] }] }
        }"#;
        let (document, buffers, images) = gltf::import_slice(json.as_bytes()).expect("valid glTF");
        let mut scene = GltfScene::new(std::sync::Arc::new(crate::GLTFAsset { document, buffers, images }));
        let mut world = World::default();
        while scene.load_batch(&mut world, 10) {}
        let strength = |world: &World| world.get_entities::<Mesh>()[0].material.emissive_strength;

        // Not autoplaying: ticking leaves it alone.
        scene.animate(&mut world, 0.5);
        assert_eq!(strength(&world), 40.0);

        scene.pose(&mut world, 0, 0.75);
        assert!((strength(&world) - 10.0).abs() < 1e-4);
        scene.reset_animation(&mut world);
        assert_eq!(strength(&world), 40.0);

        scene.clips_mut()[0].playback.playing = true;
        scene.animate(&mut world, 0.25);
        assert!((strength(&world) - 30.0).abs() < 1e-4);
    }

    #[test]
    fn a_camera_follows_its_animated_node_and_fov() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "cameras": [{ "type": "perspective", "perspective": { "yfov": 1.0, "znear": 0.1 } }],
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "camera": 0 }],
            "extras": { "blackbird_animations": [{ "name": "Dolly", "tracks": [
                { "pointer": "/nodes/0/translation", "keys": [
                    { "time": 0.0, "value": [0, 0, 0] }, { "time": 2.0, "value": [0, 0, -4] }] },
                { "pointer": "/cameras/0/perspective/yfov", "keys": [
                    { "time": 0.0, "value": 1.0 }, { "time": 2.0, "value": 0.5 }] }] }] }
        }"#;
        let (document, buffers, images) = gltf::import_slice(json.as_bytes()).expect("valid glTF");
        let mut scene = GltfScene::new(std::sync::Arc::new(crate::GLTFAsset { document, buffers, images }));
        let mut world = World::default();
        while scene.load_batch(&mut world, 10) {}

        scene.pose(&mut world, 0, 1.0);
        let camera = world.get_entities::<Camera>()[0];
        assert!((camera.eye.z + 2.0).abs() < 1e-5);
        assert!((camera.target.z + 3.0).abs() < 1e-5, "still looks down -Z");
        assert!((camera.fovy - 0.75).abs() < 1e-5);

        scene.reset_animation(&mut world);
        let camera = world.get_entities::<Camera>()[0];
        assert_eq!((camera.eye.z, camera.fovy), (0.0, 1.0));
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
    fn a_track_overrides_the_standard_channel_it_targets() {
        let (doc, buffers, _) = gltf::import_slice(RISE.as_bytes()).expect("valid glTF");
        let standard = build_clips(&doc, &buffers, &[]);
        assert_eq!(standard[0].clip.sample(0.5).next().map(|(_, v)| v[1]), Some(1.0));

        let lowered = TrackData::from_curve(
            Target::Translation(0),
            &Curve::new(3, vec![Key::new(0.0, [0.0; 4], Interpolation::Linear), Key::new(1.0, [0.0, -2.0, 0.0, 0.0], Interpolation::Linear)]),
        );
        let data = ClipData { animation: Some(0), tracks: vec![lowered], ..serde_json::from_str(r#"{ "name": "Rise" }"#).unwrap() };
        let clips = build_clips(&doc, &buffers, &[data]);
        assert_eq!(clips.len(), 1, "the clip includes the animation, so it isn't listed twice");
        let values: Vec<f32> = clips[0].clip.sample(0.5).map(|(_, v)| v[1]).collect();
        assert_eq!(values, [-1.0]);
    }

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
            Target::CameraFov(1),
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

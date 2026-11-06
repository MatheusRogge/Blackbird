use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::{Arc, RwLock};

use engine_core::{entity::EntityKey, world::World};
use ultraviolet::Vec4;

use crate::{
    camera::Camera,
    debug_flags::RenderDebugFlags,
    graph::{NodeId, RenderGraph},
    light::{
        emissive::{self, EmissiveRect},
        AreaLight, GpuAreaLight, GpuPointLight, GpuSkyLight, GpuSpotLight, LightCounts,
        MAX_AREA_LIGHTS, MAX_POINT_LIGHTS, MAX_SKY_LIGHTS, MAX_SPOT_LIGHTS, PointLight, SkyLight,
        SpotLight, SHADOW_KIND_AREA, SHADOW_KIND_POINT, SHADOW_KIND_SPOT, SHADOW_SLOT_NONE, ShadowedLights,
        SharedShadowCasters, SpotShadow,
        shadow_slots::{ShadowCandidate, ShadowSlots},
    },
    mesh::Mesh,
    pass::{PassContext, Pass, PassDesc},
    resource::{ResourceDescriptor, ResourceId},
};

/// Emissive meshes dimmer than this (in lumens, emitting from one side) don't become lights.
const MIN_EMISSIVE_FLUX: f32 = 0.01;
/// An area light's shadow range ends where its illuminance drops below this (lux), up to the max.
const MIN_SHADOW_ILLUMINANCE: f32 = 1e-3;
const MAX_AREA_SHADOW_RANGE: f32 = 50.0;
/// How far in front of a one-sided area light its shadow map is centred.
const AREA_SHADOW_OFFSET: f32 = 0.02;
/// Spot lights up to this outer angle get one atlas tile instead of a cube.
const MAX_SPOT_SHADOW_ANGLE: f32 = std::f32::consts::FRAC_PI_3;
/// Extra field of view around a spot's outer cone, so PCF and the normal offset near the cone's
/// edge still land on the map.
const SPOT_SHADOW_MARGIN: f32 = 0.1;

pub struct LightBuffers {
    pub point_buffer_id: ResourceId,
    pub spot_buffer_id: ResourceId,
    pub area_buffer_id: ResourceId,
    pub sky_buffer_id: ResourceId,
    pub counts_buffer_id: ResourceId,
    /// Uniform `ShadowedLights`.
    pub shadowed_buffer_id: ResourceId,
}

pub struct LightUploadPass {
    node_id: Option<NodeId>,
    pub buffers: LightBuffers,
    /// `emissive::fit` per mesh entity (vertices never change in place), pruned to live meshes.
    emissive_rects: HashMap<EntityKey, Option<EmissiveRect>>,
    /// Emissive texture means per `TextureAsset::id`.
    emissive_means: HashMap<u64, [f32; 3]>,
    debug_flags: Arc<RwLock<RenderDebugFlags>>,
    /// The slots picked this frame, for `ShadowPass` to render.
    shadow_casters: SharedShadowCasters,
    shadow_slots: ShadowSlots,
}

impl LightUploadPass {
    pub fn new(
        graph: &mut RenderGraph,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
        shadow_casters: SharedShadowCasters,
    ) -> (Self, LightBuffers) {
        let point_buffer_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: (MAX_POINT_LIGHTS * size_of::<GpuPointLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let spot_buffer_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: (MAX_SPOT_LIGHTS * size_of::<GpuSpotLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let area_buffer_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: (MAX_AREA_LIGHTS * size_of::<GpuAreaLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sky_buffer_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: (MAX_SKY_LIGHTS * size_of::<GpuSkyLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let counts_buffer_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: size_of::<LightCounts>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shadowed_buffer_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: size_of::<ShadowedLights>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buffers = LightBuffers {
            point_buffer_id,
            spot_buffer_id,
            area_buffer_id,
            sky_buffer_id,
            counts_buffer_id,
            shadowed_buffer_id,
        };

        let pass = Self {
            node_id: None,
            emissive_rects: HashMap::new(),
            emissive_means: HashMap::new(),
            debug_flags,
            shadow_casters,
            shadow_slots: ShadowSlots::default(),
            buffers: LightBuffers {
                point_buffer_id,
                spot_buffer_id,
                area_buffer_id,
                sky_buffer_id,
                counts_buffer_id,
                shadowed_buffer_id,
            },
        };

        (pass, buffers)
    }
}

impl PassDesc for LightUploadPass {
    fn name(&self) -> &'static str {
        "light_upload"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![
            self.buffers.point_buffer_id,
            self.buffers.spot_buffer_id,
            self.buffers.area_buffer_id,
            self.buffers.sky_buffer_id,
            self.buffers.counts_buffer_id,
            self.buffers.shadowed_buffer_id,
        ]
    }
}

impl LightUploadPass {
    /// One one-sided area light per flat emissive mesh, brightest first. Luminance is the mesh's
    /// emissive factor × strength × its emissive texture's mean, the same nits `gbuffer` writes.
    fn emissive_lights(&mut self, world: &World) -> Vec<AreaLight> {
        let meshes = world.get_entity_handles::<Mesh>();
        let live: HashSet<EntityKey> = meshes.iter().map(|h| h.id).collect();
        self.emissive_rects.retain(|key, _| live.contains(key));

        let mut lights: Vec<(f32, AreaLight)> = Vec::new();
        for handle in &meshes {
            let mesh = handle.entity;
            let factors = &mesh.material;
            if factors.emissive_strength <= 0.0 || factors.emissive.iter().all(|&c| c <= 0.0) {
                continue;
            }
            let Some(rect) = *self.emissive_rects.entry(handle.id).or_insert_with(|| emissive::fit(mesh)) else {
                continue;
            };
            let mean = mesh.emissive_texture.as_ref().map_or([1.0; 3], |t| {
                *self.emissive_means.entry(t.id()).or_insert_with(|| {
                    let [r, g, b, _] = t.mean(true);
                    [r, g, b]
                })
            });
            let radiance: [f32; 3] = std::array::from_fn(|c| factors.emissive[c] * factors.emissive_strength * mean[c]);
            let intensity = radiance.iter().copied().fold(0.0, f32::max);
            let rect = rect.transformed(mesh.transform);
            let flux = std::f32::consts::PI * intensity * rect.area();
            if flux < MIN_EMISSIVE_FLUX {
                continue;
            }
            lights.push((
                flux,
                AreaLight {
                    position: rect.center,
                    right: rect.right,
                    up: rect.up,
                    color: radiance.map(|c| c / intensity).into(),
                    intensity,
                    two_sided: false,
                },
            ));
        }
        lights.sort_by(|a, b| b.0.total_cmp(&a.0));
        lights.into_iter().map(|(_, light)| light).collect()
    }
}

impl Pass for LightUploadPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        world: &World,
    ) {
        let cameras = world.get_entities::<Camera>();
        let Some(camera) = cameras.first() else {
            return;
        };
        let view = camera.view_matrix();

        let to_vs = |p: ultraviolet::Vec3| -> [f32; 3] {
            let v = view * Vec4::new(p.x, p.y, p.z, 1.0);
            [v.x, v.y, v.z]
        };

        let dir_to_vs = |d: ultraviolet::Vec3| -> [f32; 3] {
            let v = view * Vec4::new(d.x, d.y, d.z, 0.0);
            [v.x, v.y, v.z]
        };

        let points = world.get_entities::<PointLight>();
        let spots = world.get_entities::<SpotLight>();
        let mut areas: Vec<AreaLight> = world
            .get_entities::<AreaLight>()
            .into_iter()
            .take(MAX_AREA_LIGHTS)
            .map(|l| AreaLight { ..*l })
            .collect();
        let (emissive_lights, shadows) = {
            let flags = self.debug_flags.read().expect("debug_flags poisoned");
            (flags.emissive_lights, flags.shadows)
        };
        if emissive_lights {
            let free = MAX_AREA_LIGHTS - areas.len();
            areas.extend(self.emissive_lights(world).into_iter().take(free));
        }

        let mut point_lights: Vec<GpuPointLight> = points
            .iter()
            .take(MAX_POINT_LIGHTS)
            .map(|l| GpuPointLight {
                position_vs: to_vs(l.position),
                radius: l.radius,
                color: l.color.into(),
                intensity: l.intensity,
                source_radius: l.source_radius,
                shadow_slot: SHADOW_SLOT_NONE,
                _pad: [0.0; 2],
            })
            .collect();

        let mut spot_lights: Vec<GpuSpotLight> = spots
            .iter()
            .take(MAX_SPOT_LIGHTS)
            .map(|l| GpuSpotLight {
                position_vs: to_vs(l.position),
                radius: l.radius,
                direction_vs: dir_to_vs(l.direction),
                inner_cos: l.inner_angle.cos(),
                color: l.color.into(),
                outer_cos: l.outer_angle.cos(),
                intensity: l.intensity,
                source_radius: l.source_radius,
                shadow_slot: SHADOW_SLOT_NONE,
                _pad: 0.0,
            })
            .collect();

        let to_gpu = |l: &AreaLight| GpuAreaLight {
            position_vs: to_vs(l.position),
            intensity: l.intensity,
            right_vs: dir_to_vs(l.right),
            two_sided: if l.two_sided { 1.0 } else { 0.0 },
            up_vs: dir_to_vs(l.up),
            shadow_slot: SHADOW_SLOT_NONE,
            color: l.color.into(),
            _pad2: 0.0,
        };
        let mut area_lights: Vec<GpuAreaLight> = areas.iter().map(to_gpu).collect();

        // The lights most likely to matter on screen: flux over squared distance to the camera.
        let candidate = |kind: u32, i: usize, position: ultraviolet::Vec3, origin, range: f32, flux: f32, half_size, spot| {
            let distance = (position - camera.eye).mag();
            ShadowCandidate {
                score: flux / (distance * distance + 1.0),
                entry: kind << 16 | i as u32,
                origin,
                range,
                spot,
                half_size,
                coverage: range / distance.max(1e-3),
            }
        };
        let four_pi = 4.0 * std::f32::consts::PI;
        let mut candidates: Vec<ShadowCandidate> = points
            .iter()
            .take(MAX_POINT_LIGHTS)
            .enumerate()
            .map(|(i, l)| {
                candidate(SHADOW_KIND_POINT, i, l.position, l.position, l.radius, four_pi * l.intensity, l.source_radius, None)
            })
            .chain(spots.iter().take(MAX_SPOT_LIGHTS).enumerate().map(|(i, l)| {
                let spot = (l.outer_angle <= MAX_SPOT_SHADOW_ANGLE).then(|| SpotShadow {
                    direction: l.direction.normalized(),
                    tan_half_fov: (l.outer_angle + SPOT_SHADOW_MARGIN).tan(),
                });
                let flux = four_pi * l.intensity;
                candidate(SHADOW_KIND_SPOT, i, l.position, l.position, l.radius, flux, l.source_radius, spot)
            }))
            .chain(areas.iter().enumerate().map(|(i, l)| {
                let sides = if l.two_sided { 2.0 } else { 1.0 };
                let area = 4.0 * l.right.mag() * l.up.mag();
                let flux = sides * std::f32::consts::PI * l.intensity * area;
                // Where its illuminance (≈ L·A / d²) falls below MIN_SHADOW_ILLUMINANCE.
                let range = (l.intensity * area / MIN_SHADOW_ILLUMINANCE).sqrt().clamp(1.0, MAX_AREA_SHADOW_RANGE);
                // One-sided lights shadow from just in front, so their own emitter mesh isn't a caster.
                let origin = if l.two_sided {
                    l.position
                } else {
                    l.position + l.right.cross(l.up).normalized() * AREA_SHADOW_OFFSET
                };
                let half_size = l.right.mag().max(l.up.mag());
                candidate(SHADOW_KIND_AREA, i, l.position, origin, range, flux, half_size, None)
            }))
            .filter(|c| c.score > 0.0)
            .collect();
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        if !shadows {
            candidates.clear();
        }

        let slots = self.shadow_slots.update(&candidates);
        let mut shadowed = ShadowedLights::default();
        for (slot, caster) in slots.iter().enumerate() {
            let Some(caster) = caster else { continue };
            let index = (caster.entry & 0xffff) as usize;
            let slot_id = slot as u32;
            match caster.entry >> 16 {
                SHADOW_KIND_POINT => point_lights[index].shadow_slot = slot_id,
                SHADOW_KIND_SPOT => spot_lights[index].shadow_slot = slot_id,
                _ => area_lights[index].shadow_slot = slot_id,
            }
            shadowed.count = slot_id + 1;
        }
        *self.shadow_casters.lock().expect("shadow_casters poisoned") = slots;

        let sky_lights: Vec<GpuSkyLight> = world
            .get_entities::<SkyLight>()
            .iter()
            .take(MAX_SKY_LIGHTS)
            .map(|l| GpuSkyLight {
                direction_vs: dir_to_vs(l.direction.normalized()),
                intensity: l.intensity,
                color: l.color.into(),
                _pad: 0.0,
            })
            .collect();

        let counts = LightCounts {
            num_point: point_lights.len() as u32,
            num_spot: spot_lights.len() as u32,
            num_area: area_lights.len() as u32,
            num_sky: sky_lights.len() as u32,
        };

        if !point_lights.is_empty() {
            ctx.buffers[&self.buffers.point_buffer_id].write(queue, bytemuck::cast_slice(&point_lights),
            );
        }

        if !spot_lights.is_empty() {
            ctx.buffers[&self.buffers.spot_buffer_id].write(queue, bytemuck::cast_slice(&spot_lights),
            );
        }

        if !area_lights.is_empty() {
            ctx.buffers[&self.buffers.area_buffer_id].write(queue, bytemuck::cast_slice(&area_lights),
            );
        }

        if !sky_lights.is_empty() {
            ctx.buffers[&self.buffers.sky_buffer_id].write(queue, bytemuck::cast_slice(&sky_lights),
            );
        }

        ctx.buffers[&self.buffers.counts_buffer_id].write(queue, bytemuck::bytes_of(&counts),
        );
        ctx.buffers[&self.buffers.shadowed_buffer_id].write(queue, bytemuck::bytes_of(&shadowed));
    }
}

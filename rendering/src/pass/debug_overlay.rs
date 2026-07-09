use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};
use engine_core::world::World;

use crate::{
    debug_flags::RenderDebugFlags,
    graph::{NodeId, RenderGraph},
    pass::{Pass, PassContext, PassDesc},
    pass::probe_atlas::PROBE_GRID_DIM,
    resource::{ResourceDescriptor, ResourceId},
    shader::ShaderAsset,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayParamsGpu {
    bvh_leaves_only: u32,
    _pad: [u32; 3],
}

pub struct DebugOverlayOutputs {
    pub debug_overlay_id: ResourceId,
}

pub struct DebugOverlayPass {
    node_id: Option<NodeId>,

    bvh_nodes_id:    ResourceId,
    bvh_info_id:     ResourceId,
    camera_id:       ResourceId,
    probe_params_id: ResourceId,
    debug_overlay_id: ResourceId,

    bvh_pipeline:   Option<wgpu::RenderPipeline>,
    probe_pipeline: Option<wgpu::RenderPipeline>,
    grid_pipeline:  Option<wgpu::RenderPipeline>,
    bind_group:     Option<wgpu::BindGroup>,
    overlay_params_buf: Option<wgpu::Buffer>,

    bvh_node_count: Arc<AtomicU32>,
    debug_flags:    Arc<RwLock<RenderDebugFlags>>,
    shader:         ShaderAsset,
}

impl DebugOverlayPass {
    pub fn new(
        graph: &mut RenderGraph,
        bvh_nodes_id: ResourceId,
        bvh_info_id: ResourceId,
        camera_id: ResourceId,
        probe_params_id: ResourceId,
        bvh_node_count: Arc<AtomicU32>,
        debug_flags: Arc<RwLock<RenderDebugFlags>>,
        shader: ShaderAsset,
    ) -> (Self, DebugOverlayOutputs) {
        let debug_overlay_id = graph.alloc_resource_id(ResourceDescriptor::ScreenTexture {
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let outputs = DebugOverlayOutputs { debug_overlay_id };

        (
            Self {
                node_id: None,
                bvh_nodes_id,
                bvh_info_id,
                camera_id,
                probe_params_id,
                debug_overlay_id,
                bvh_pipeline: None,
                probe_pipeline: None,
                grid_pipeline: None,
                bind_group: None,
                overlay_params_buf: None,
                bvh_node_count,
                debug_flags,
                shader,
            },
            outputs,
        )
    }

    fn build_pipelines(
        &mut self,
        device: &wgpu::Device,
        bvh_nodes_buf: &wgpu::Buffer,
        bvh_info_buf: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
        probe_params_buf: &wgpu::Buffer,
    ) {
        let module = self.shader.compile(device).expect("debug_overlay shader compile failed");

        let overlay_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_params"),
            size: std::mem::size_of::<OverlayParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("debug_overlay_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug_overlay_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let color_target = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };

        self.bvh_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("debug_overlay_bvh"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_bvh"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_bvh"),
                targets: &[Some(color_target.clone())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }));

        self.probe_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("debug_overlay_probe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_probe"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_probe"),
                targets: &[Some(color_target.clone())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }));

        self.grid_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("debug_overlay_grid"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_grid"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_grid"),
                targets: &[Some(color_target.clone())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }));

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug_overlay_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bvh_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bvh_info_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: probe_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: overlay_params_buf.as_entire_binding() },
            ],
        }));

        self.overlay_params_buf = Some(overlay_params_buf);
    }
}

impl PassDesc for DebugOverlayPass {
    fn name(&self) -> &'static str {
        "debug_overlay"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.bvh_nodes_id, self.bvh_info_id, self.camera_id, self.probe_params_id]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.debug_overlay_id]
    }
}

impl Pass for DebugOverlayPass {
    fn bind_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }

    fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassContext<'_>,
        _world: &World,
    ) {
        let Some(&overlay_view) = ctx.views.get(&self.debug_overlay_id) else { return };
        let Some(&bvh_nodes_buf) = ctx.buffers.get(&self.bvh_nodes_id) else { return };
        let Some(&bvh_info_buf) = ctx.buffers.get(&self.bvh_info_id) else { return };
        let Some(&camera_buf) = ctx.buffers.get(&self.camera_id) else { return };
        let Some(&probe_params_buf) = ctx.buffers.get(&self.probe_params_id) else { return };

        if self.bvh_pipeline.is_none() {
            self.build_pipelines(device, bvh_nodes_buf, bvh_info_buf, camera_buf, probe_params_buf);
        }

        let (show_bvh, bvh_leaves_only, show_probes, show_grid) = self
            .debug_flags
            .read()
            .map(|f| (f.show_bvh, f.bvh_leaves_only, f.show_probes, f.show_grid))
            .unwrap_or((false, false, false, false));

        let overlay_data = OverlayParamsGpu {
            bvh_leaves_only: bvh_leaves_only as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(
            self.overlay_params_buf.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&overlay_data),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("debug_overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: overlay_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        let bg = self.bind_group.as_ref().unwrap();
        pass.set_bind_group(0, bg, &[]);

        if show_grid {
            pass.set_pipeline(self.grid_pipeline.as_ref().unwrap());
            pass.draw(0..3, 0..1);
        }

        if show_bvh {
            let node_count = self.bvh_node_count.load(Ordering::Relaxed);
            if node_count > 0 {
                pass.set_pipeline(self.bvh_pipeline.as_ref().unwrap());
                pass.draw(0..24, 0..node_count);
            }
        }

        if show_probes {
            let probe_count = PROBE_GRID_DIM[0] * PROBE_GRID_DIM[1] * PROBE_GRID_DIM[2];
            pass.set_pipeline(self.probe_pipeline.as_ref().unwrap());
            pass.draw(0..6, 0..probe_count);
        }
    }
}

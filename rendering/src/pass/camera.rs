use crate::{
    camera::Camera,
    graph::{NodeId, RenderGraph},
    pass::{PassContext, Pass, PassDesc},
    resource::{BindingResource, ResourceDescriptor, ResourceId},
};
use engine_core::world::World;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    /// near, far, frame index (mod `FRAME_PERIOD`, for temporal jitter), 0. Near and far are for
    /// linearizing reversed-Z depth.
    pub depth_params: [f32; 4],
    /// Last frame's `view_proj`, for reprojecting history (`ssr_temporal`).
    pub prev_view_proj: [[f32; 4]; 4],
}

/// The frame index in `depth_params.z` wraps here, keeping it exact in f32.
const FRAME_PERIOD: u32 = 1024;

impl CameraUniform {
    fn from_camera(camera: &Camera, viewport_aspect: f32, prev_view_proj: Option<[[f32; 4]; 4]>, frame: u32) -> Self {
        let view_proj = camera.view_proj_matrix(viewport_aspect);
        Self {
            view_proj: view_proj.into(),
            inv_view_proj: view_proj.inversed().into(),
            inv_view: camera.view_matrix().inversed().into(),
            depth_params: [camera.near, camera.far, (frame % FRAME_PERIOD) as f32, 0.0],
            prev_view_proj: prev_view_proj.unwrap_or(view_proj.into()),
        }
    }
}

pub struct CameraPass {
    node_id: Option<NodeId>,
    pub camera_buffer_id: ResourceId,
    prev_view_proj: Option<[[f32; 4]; 4]>,
    frame: u32,
}

impl CameraPass {
    pub fn new(graph: &mut RenderGraph) -> (Self, ResourceId) {
        let camera_buffer_id = graph.alloc_resource_id(ResourceDescriptor::UploadBuffer {
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        (
            Self {
                node_id: None,
                camera_buffer_id,
                prev_view_proj: None,
                frame: 0,
            },
            camera_buffer_id,
        )
    }
}

impl PassDesc for CameraPass {
    fn name(&self) -> &'static str {
        "camera"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.camera_buffer_id]
    }

    fn layout_entries(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            count: None,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        }]
    }

    fn binding_resources(&self) -> Vec<BindingResource> {
        vec![BindingResource {
            slot: 0,
            resource_id: self.camera_buffer_id,
            descriptor: ResourceDescriptor::UploadBuffer {
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        }]
    }
}

impl Pass for CameraPass {
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
        let Some(main_camera) = cameras.first() else {
            return;
        };

        let (w, h) = ctx.surface_size;
        let uniform = CameraUniform::from_camera(main_camera, w as f32 / h as f32, self.prev_view_proj, self.frame);
        self.prev_view_proj = Some(uniform.view_proj);
        self.frame = self.frame.wrapping_add(1);
        let data = bytemuck::bytes_of(&uniform);

        ctx.buffers[&self.camera_buffer_id].write(queue, data);
    }
}

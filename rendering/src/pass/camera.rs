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
    /// near, far, 0, 0. For linearizing reversed-Z depth.
    pub depth_params: [f32; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera, viewport_aspect: f32) -> Self {
        let view_proj = camera.view_proj_matrix(viewport_aspect);
        Self {
            view_proj: view_proj.into(),
            inv_view_proj: view_proj.inversed().into(),
            inv_view: camera.view_matrix().inversed().into(),
            depth_params: [camera.near, camera.far, 0.0, 0.0],
        }
    }
}

pub struct CameraPass {
    node_id: Option<NodeId>,
    pub camera_buffer_id: ResourceId,
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
        let uniform = CameraUniform::from_camera(main_camera, w as f32 / h as f32);
        let data = bytemuck::bytes_of(&uniform);

        ctx.buffers[&self.camera_buffer_id].write(queue, data);
    }
}

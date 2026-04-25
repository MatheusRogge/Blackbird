use crate::{
    camera::Camera,
    graph::{NodeId, RenderGraph},
    pass::{PassContext, RenderPass, RenderPassDesc},
    resource::{BindingResource, ResourceDescriptor, ResourceId},
};
use engine_core::world::World;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_proj_matrix().into(),
        }
    }
}

pub struct CameraPass {
    node_id: Option<NodeId>,
    pub camera_buffer_id: ResourceId,
}

impl CameraPass {
    pub fn new(graph: &mut RenderGraph) -> (Self, ResourceId) {
        let camera_buffer_id = graph.alloc_resource_id(ResourceDescriptor::Buffer {
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

impl RenderPassDesc for CameraPass {
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
            descriptor: ResourceDescriptor::Buffer {
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        }]
    }
}

impl RenderPass for CameraPass {
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

        let uniform = CameraUniform::from_camera(main_camera);
        let data = bytemuck::bytes_of(&uniform);

        queue.write_buffer(ctx.buffers[&self.camera_buffer_id], 0, data);
    }
}

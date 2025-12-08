use engine::entity::Entity;
use ultraviolet::{Mat4, Vec3, projection};
use wgpu::{Buffer, BufferDescriptor, Device};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub field_of_view: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Entity for Camera {}

impl Camera {
    pub fn get_camera_uniform(&self) -> CameraUniform {
        let view = Mat4::look_at(self.eye, self.target, self.up);

        let projection = projection::perspective_wgpu_dx(
            self.field_of_view.to_radians(),
            self.aspect,
            self.znear,
            self.zfar,
        );

        CameraUniform {
            view_proj: (projection * view).into(),
        }
    }

    pub fn get_uniform_buffer(&self, device: &Device) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            mapped_at_creation: false,
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
}

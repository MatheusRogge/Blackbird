use std::io;

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, ColorTargetState, Device, FragmentState,
    PipelineCompilationOptions, PipelineLayoutDescriptor, RenderPipeline, RenderPipelineDescriptor,
    SurfaceConfiguration, VertexState,
};

use crate::{mesh::Vertex, shader::ShaderAsset};
use thiserror::Error;

pub struct StageShaderDescriptor<'a> {
    pub entrypoint: &'a str,
    pub asset: &'a ShaderAsset,
}

impl<'a> StageShaderDescriptor<'a> {
    pub(crate) fn bind_group_layout(&self, device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &self.asset.bindings,
        })
    }
}

pub struct RenderingPipelineDescriptor<'a> {
    pub vertex: StageShaderDescriptor<'a>,
    pub fragment: StageShaderDescriptor<'a>,
}

#[derive(Error, Debug)]
pub enum RenderingPipelineError {
    #[error("Shader error")]
    IO(#[from] io::Error),
}

impl<'a> RenderingPipelineDescriptor<'a> {
    pub(crate) fn create_pipeline(
        &self,
        device: &Device,
        surface_config: &SurfaceConfiguration,
        extra_layouts: &[BindGroupLayout],
    ) -> Result<RenderPipeline, RenderingPipelineError> {
        let vertex_shader_module = self.vertex.asset.compile(device)?;
        let fragment_shader_module = self.fragment.asset.compile(device)?;

        let mut layouts = Vec::new();

        let vertex_layout = self.vertex.bind_group_layout(device);
        layouts.push(vertex_layout);

        let fragment_layout = self.fragment.bind_group_layout(device);
        layouts.push(fragment_layout);

        let mut bind_group_layouts: Vec<&BindGroupLayout> = layouts.iter().skip(2).collect();
        for layout in extra_layouts.iter() {
            bind_group_layouts.push(layout);
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            push_constant_ranges: &[],
            bind_group_layouts: &bind_group_layouts,
        });

        let vertex_state = VertexState {
            module: &vertex_shader_module,
            buffers: &[Vertex::buffer_descriptor()],
            entry_point: Some(self.vertex.entrypoint),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        };

        let fragment_state = FragmentState {
            module: &fragment_shader_module,
            entry_point: Some(self.fragment.entrypoint),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: surface_config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: vertex_state,
            fragment: Some(fragment_state),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            cache: None,
            multiview: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        Ok(pipeline)
    }
}

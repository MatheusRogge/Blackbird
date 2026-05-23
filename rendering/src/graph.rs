use engine_core::world::World;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use wgpu::{BindGroup, BindGroupLayout, Device, util::DeviceExt};

use crate::{
    pass::{Pass, PassContext},
    profiler::{FrameProfiler, FrameStats, GpuProfiler},
    resource::{AllocatedResource, GraphResource, ResourceDescriptor, ResourceId},
};

pub type NodeId = u32;

/// Global allocator and resource pool shared by the main graph and all subgraphs.
/// All `NodeId`s and `ResourceId`s within a graph hierarchy are unique.
#[derive(Default)]
pub struct GraphContext {
    next_node_id: NodeId,
    next_resource_id: ResourceId,
    resources: HashMap<ResourceId, GraphResource>,
    passes: HashMap<NodeId, Box<dyn Pass>>,
    bind_group_layouts: HashMap<NodeId, BindGroupLayout>,
    bind_group_resources: HashMap<NodeId, BindGroup>,
}

impl GraphContext {
    pub fn alloc_resource_id(&mut self, desc: ResourceDescriptor) -> ResourceId {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        self.resources.insert(
            id,
            GraphResource {
                version: 0,
                desc,
                resource: None,
            },
        );
        id
    }

    fn alloc_node_id(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    fn register_pass<P: Pass>(&mut self, device: &Device, mut pass: P) -> NodeId {
        let node_id = self.alloc_node_id();
        pass.bind_node_id(node_id);

        for br in pass.binding_resources() {
            self.resources
                .entry(br.resource_id)
                .or_insert(GraphResource {
                    version: 0,
                    desc: br.descriptor,
                    resource: None,
                });
        }

        let layout_entries = pass.layout_entries();
        if !layout_entries.is_empty() {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(pass.name()),
                entries: &layout_entries,
            });
            self.bind_group_layouts.insert(node_id, layout);
        }

        self.passes.insert(node_id, Box::new(pass));
        node_id
    }
}

/// A node in the graph tree: either a single pass or a grouped subgraph.
/// Subgraphs are flattened into a single topological sort at compile time.
pub enum GraphNode {
    Pass(NodeId),
    SubGraph(Vec<GraphNode>),
}

impl GraphNode {
    fn flatten(&self) -> Vec<NodeId> {
        match self {
            GraphNode::Pass(id) => vec![*id],
            GraphNode::SubGraph(nodes) => nodes.iter().flat_map(|n| n.flatten()).collect(),
        }
    }
}

/// Builder for a subgraph. Borrows the parent's `GraphContext` so all
/// resource and node IDs are globally unique. Call `.finish()` and pass
/// the result to `RenderGraph::add_subgraph`.
pub struct SubGraphBuilder<'a> {
    ctx: &'a mut GraphContext,
    nodes: Vec<GraphNode>,
}

impl<'a> SubGraphBuilder<'a> {
    pub fn alloc_resource_id(&mut self, desc: ResourceDescriptor) -> ResourceId {
        self.ctx.alloc_resource_id(desc)
    }

    pub fn create_external_resource(&mut self) -> ResourceId {
        self.ctx.alloc_resource_id(ResourceDescriptor::ExternalView)
    }

    pub fn add_pass<P: Pass>(&mut self, device: &Device, pass: P) -> NodeId {
        let node_id = self.ctx.register_pass(device, pass);
        self.nodes.push(GraphNode::Pass(node_id));
        node_id
    }

    pub fn finish(self) -> Vec<GraphNode> {
        self.nodes
    }
}

#[derive(Default)]
pub struct RenderGraph {
    ctx: GraphContext,
    nodes: Vec<GraphNode>,
    profiler: FrameProfiler,
    gpu_profiler: Option<GpuProfiler>,
    gpu_resolved: bool,
    surface_size: (u32, u32),
}

impl RenderGraph {
    pub fn alloc_resource_id(&mut self, desc: ResourceDescriptor) -> ResourceId {
        self.ctx.alloc_resource_id(desc)
    }

    pub fn create_external_resource(&mut self) -> ResourceId {
        self.ctx.alloc_resource_id(ResourceDescriptor::ExternalView)
    }

    pub fn import_texture(&mut self, view: wgpu::TextureView) -> ResourceId {
        let id = self.ctx.alloc_resource_id(ResourceDescriptor::ExternalView);

        if let Some(res) = self.ctx.resources.get_mut(&id) {
            res.resource = Some(AllocatedResource::ExternalView(view));
            res.version = 1;
        }

        id
    }

    pub fn update_imported(&mut self, resource_id: ResourceId, view: wgpu::TextureView) -> bool {
        let Some(resource) = self.ctx.resources.get_mut(&resource_id) else {
            return false;
        };

        resource.resource = Some(AllocatedResource::ExternalView(view));
        resource.version += 1;

        true
    }

    pub fn add_pass<P: Pass>(&mut self, device: &Device, pass: P) -> NodeId {
        let node_id = self.ctx.register_pass(device, pass);
        self.nodes.push(GraphNode::Pass(node_id));

        node_id
    }

    pub fn add_subgraph(&mut self, nodes: Vec<GraphNode>) {
        self.nodes.push(GraphNode::SubGraph(nodes));
    }

    pub fn subgraph(&mut self) -> SubGraphBuilder<'_> {
        SubGraphBuilder {
            ctx: &mut self.ctx,
            nodes: Vec::new(),
        }
    }

    pub fn on_resize(&mut self, width: u32, height: u32) {
        self.surface_size = (width, height);
        for res in self.ctx.resources.values_mut() {
            match &mut res.desc {
                ResourceDescriptor::ScreenTexture { .. } => {
                    res.resource = None;
                    res.version += 1;
                }
                ResourceDescriptor::Texture { size, .. } => {
                    if size.width != width || size.height != height {
                        size.width = width;
                        size.height = height;
                        res.resource = None;
                        res.version += 1;
                    }
                }
                _ => {}
            }
        }

        for pass in self.ctx.passes.values_mut() {
            pass.on_resize(width, height);
        }

        self.ctx.bind_group_resources.clear();
    }

    fn flat_node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().flat_map(|n| n.flatten()).collect()
    }

    fn compile(&self, output: ResourceId) -> Vec<NodeId> {
        let all_ids = self.flat_node_ids();

        let writer_of: HashMap<ResourceId, NodeId> = all_ids
            .iter()
            .flat_map(|&id| {
                self.ctx.passes[&id]
                    .writes()
                    .into_iter()
                    .map(move |r| (r, id))
            })
            .collect();

        // Backwards reachability from the output resource
        let mut stack = vec![output];
        let mut needed = HashSet::new();

        while let Some(res) = stack.pop() {
            let Some(&writer) = writer_of.get(&res) else {
                continue;
            };

            if needed.insert(writer) {
                stack.extend(self.ctx.passes[&writer].reads());
            }
        }

        // Kahn's topological sort over the needed nodes
        let mut in_degree: HashMap<NodeId, usize> = needed.iter().map(|&k| (k, 0)).collect();
        let mut edges: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for &id in &needed {
            for res in self.ctx.passes[&id].reads() {
                let Some(&writer) = writer_of.get(&res) else {
                    continue;
                };
                if needed.contains(&writer) {
                    edges.entry(writer).or_default().push(id);
                    *in_degree.entry(id).or_insert(0) += 1;
                }
            }
        }

        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut order = Vec::new();
        while let Some(id) = queue.pop() {
            order.push(id);
            if let Some(deps) = edges.get(&id) {
                for &dep in deps {
                    let d = in_degree.get_mut(&dep).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        queue.push(dep);
                    }
                }
            }
        }

        order
    }

    pub fn frame_stats(&self) -> &FrameStats {
        self.profiler.stats()
    }

    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: ResourceId,
        surface_view: &wgpu::TextureView,
        surface_size: (u32, u32),
        world: &World,
    ) -> wgpu::CommandBuffer {
        if let Some(gpu) = &mut self.gpu_profiler
            && gpu.try_read_results()
        {
            let times = gpu.last_gpu_times_ms.clone();
            self.profiler.apply_gpu_times(&times);
        }

        self.profiler.begin_frame();
        let order = self.compile(output);

        let timing_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if self.gpu_profiler.is_none() && device.features().contains(timing_features) {
            self.gpu_profiler = Some(GpuProfiler::new(device, queue, order.len() as u32));
        }

        self.surface_size = surface_size;
        let (sw, sh) = surface_size;
        for res in self.ctx.resources.values_mut() {
            let skip = matches!(res.desc, ResourceDescriptor::ExternalView);
            if res.resource.is_none() && !skip {
                res.resource = Some(Self::allocate_resource(device, &res.desc, sw, sh));
            }
        }

        let all_ids = self.flat_node_ids();
        let writer_of: HashMap<ResourceId, NodeId> = all_ids
            .iter()
            .flat_map(|&id| {
                self.ctx.passes[&id]
                    .writes()
                    .into_iter()
                    .map(move |r| (r, id))
            })
            .collect();

        for &node_id in &all_ids {
            let Some(layout) = self.ctx.bind_group_layouts.get(&node_id) else {
                continue;
            };

            let binding_resources = self.ctx.passes[&node_id].binding_resources();
            if binding_resources.is_empty() {
                continue;
            }

            let sampler_descs = self.ctx.passes[&node_id].samplers();
            let samplers: HashMap<u32, wgpu::Sampler> = sampler_descs
                .into_iter()
                .map(|(slot, desc)| (slot, device.create_sampler(&desc)))
                .collect();

            let entries: Vec<wgpu::BindGroupEntry> = binding_resources
                .iter()
                .filter_map(|br| {
                    let res = self.ctx.resources.get(&br.resource_id)?.resource.as_ref()?;
                    Some(match res {
                        AllocatedResource::Buffer(buf) => wgpu::BindGroupEntry {
                            binding: br.slot,
                            resource: buf.as_entire_binding(),
                        },
                        AllocatedResource::Texture(_, view)
                        | AllocatedResource::ExternalView(view) => wgpu::BindGroupEntry {
                            binding: br.slot,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                    })
                })
                .chain(samplers.iter().map(|(slot, sampler)| wgpu::BindGroupEntry {
                    binding: *slot,
                    resource: wgpu::BindingResource::Sampler(sampler),
                }))
                .collect();

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            });

            self.ctx.bind_group_resources.insert(node_id, bg);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RenderGraph"),
        });

        for (pass_idx, &node_id) in order.iter().enumerate() {
            let reads = self.ctx.passes[&node_id].reads();
            let writes = self.ctx.passes[&node_id].writes();

            let mut views: HashMap<ResourceId, &wgpu::TextureView> = HashMap::new();
            let mut buffers: HashMap<ResourceId, &wgpu::Buffer> = HashMap::new();

            for id in reads.iter().chain(writes.iter()) {
                if *id == output {
                    views.insert(*id, surface_view);
                    continue;
                }
                match self.ctx.resources.get(id).and_then(|r| r.resource.as_ref()) {
                    Some(AllocatedResource::Texture(_, view)) => {
                        views.insert(*id, view);
                    }
                    Some(AllocatedResource::ExternalView(view)) => {
                        views.insert(*id, view);
                    }
                    Some(AllocatedResource::Buffer(buf)) => {
                        buffers.insert(*id, buf);
                    }
                    None => {}
                }
            }

            let bind_group = self
                .ctx
                .bind_group_resources
                .get(&node_id)
                .map(|bg| bg as &wgpu::BindGroup);

            let mut upstream: HashMap<NodeId, &wgpu::BindGroup> = HashMap::new();
            for res_id in &reads {
                let Some(&writer_node) = writer_of.get(res_id) else {
                    continue;
                };

                if let Some(bg) = self.ctx.bind_group_resources.get(&writer_node) {
                    upstream.insert(writer_node, bg);
                }
            }

            let ctx = PassContext {
                views,
                buffers,
                bind_group,
                upstream,
                surface_size,
            };

            if let Some(gpu) = &self.gpu_profiler {
                gpu.write_begin(&mut encoder, pass_idx as u32);
            }

            let pass = self.ctx.passes.get_mut(&node_id).unwrap();
            let pass_name = pass.name();
            let t0 = Instant::now();

            pass.execute(device, queue, &mut encoder, &ctx, world);
            self.profiler.record_pass(pass_name, t0.elapsed());

            if let Some(gpu) = &self.gpu_profiler {
                gpu.write_end(&mut encoder, pass_idx as u32);
            }

            for res_id in &writes {
                if let Some(r) = self.ctx.resources.get_mut(res_id) {
                    r.version += 1;
                }
            }
        }

        self.gpu_resolved = self
            .gpu_profiler
            .as_ref()
            .is_some_and(|gpu| gpu.resolve(&mut encoder));

        self.profiler.end_frame();
        encoder.finish()
    }

    pub fn schedule_gpu_readback(&mut self, device: &wgpu::Device) {
        if !self.gpu_resolved {
            return;
        }

        if let Some(gpu) = &mut self.gpu_profiler {
            gpu.schedule_readback(device);
        }
    }

    fn allocate_resource(
        device: &wgpu::Device,
        desc: &ResourceDescriptor,
        sw: u32,
        sh: u32,
    ) -> AllocatedResource {
        let make_texture =
            |size: wgpu::Extent3d, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                });

                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                AllocatedResource::Texture(tex, view)
            };

        match desc {
            ResourceDescriptor::ExternalView => {
                panic!("ExternalView must be populated via update_imported, not allocated")
            }
            ResourceDescriptor::Texture {
                size,
                format,
                usage,
            } => make_texture(*size, *format, *usage),
            ResourceDescriptor::FixedTexture { size, format, usage } => {
                make_texture(*size, *format, *usage)
            }
            ResourceDescriptor::Fixed3DTexture { size, format, usage } => {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: size[2],
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: *format,
                    usage: *usage,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D3),
                    ..Default::default()
                });
                AllocatedResource::Texture(tex, view)
            }
            ResourceDescriptor::ScreenTexture { format, usage } => {
                let size = wgpu::Extent3d {
                    width: sw,
                    height: sh,
                    depth_or_array_layers: 1,
                };
                make_texture(size, *format, *usage)
            }
            ResourceDescriptor::Buffer { size, usage } => {
                AllocatedResource::Buffer(device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: *size,
                    usage: *usage,
                    mapped_at_creation: false,
                }))
            }
            ResourceDescriptor::BufferInit { data, usage } => AllocatedResource::Buffer(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data,
                    usage: *usage,
                }),
            ),
        }
    }
}

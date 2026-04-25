use engine_core::world::World;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use wgpu::{BindGroup, BindGroupLayout, Device, util::DeviceExt};

use crate::{
    pass::{PassContext, RenderPass},
    resource::{AllocatedResource, GraphResource, ResourceDescriptor, ResourceId},
};

pub type NodeId = u32;

#[derive(Default)]
pub struct RenderGraph {
    next_node_id: NodeId,
    next_resource_id: ResourceId,

    nodes: Vec<(NodeId, Box<dyn RenderPass>)>,
    resources: HashMap<ResourceId, GraphResource>,
    bind_group_layouts: HashMap<NodeId, BindGroupLayout>,
    bind_group_resources: HashMap<NodeId, Arc<BindGroup>>,
}

impl RenderGraph {
    /// Allocate a new resource slot with the given descriptor.
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

    /// Register an external resource slot (e.g. the swapchain surface).
    /// The actual view is injected each frame via `update_imported`.
    pub fn create_external_resource(&mut self) -> ResourceId {
        self.alloc_resource_id(ResourceDescriptor::ExternalView)
    }

    /// Import an external texture view (e.g. swapchain) into the graph.
    /// Returns the `ResourceId` for later updates via `update_imported`.
    pub fn import_texture(&mut self, view: wgpu::TextureView) -> ResourceId {
        let resource_id = self.next_resource_id;
        self.next_resource_id += 1;

        self.resources.insert(
            resource_id,
            GraphResource {
                version: 1,
                desc: ResourceDescriptor::ExternalView,
                resource: Some(AllocatedResource::ExternalView(view)),
            },
        );

        resource_id
    }

    /// Update the texture view for an imported external resource.
    pub fn update_imported(&mut self, resource_id: ResourceId, view: wgpu::TextureView) -> bool {
        let Some(resource) = self.resources.get_mut(&resource_id) else {
            return false;
        };

        resource.resource = Some(AllocatedResource::ExternalView(view));
        resource.version += 1;

        true
    }

    /// Add a render pass to the graph. Returns the assigned `NodeId`.
    pub fn add_pass<P>(&mut self, device: &Device, mut pass: P) -> NodeId
    where
        P: RenderPass,
    {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        pass.bind_node_id(node_id);

        // Register resources declared by the pass (if not already present)
        for br in pass.binding_resources() {
            self.resources
                .entry(br.resource_id)
                .or_insert(GraphResource {
                    version: 0,
                    desc: br.descriptor,
                    resource: None,
                });
        }

        // Create and cache bind group layout from the pass's layout entries
        let layout_entries = pass.layout_entries();
        if !layout_entries.is_empty() {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(pass.name()),
                entries: &layout_entries,
            });

            self.bind_group_layouts.insert(node_id, layout);
        }

        self.nodes.push((node_id, Box::new(pass)));
        node_id
    }

    /// Invalidate sized resources on surface resize.
    pub fn on_resize(&mut self, width: u32, height: u32) {
        for res in self.resources.values_mut() {
            let ResourceDescriptor::Texture { ref mut size, .. } = res.desc else {
                continue;
            };

            if size.width != width || size.height != height {
                size.width = width;
                size.height = height;

                res.resource = None;
                res.version += 1;
            }
        }

        self.bind_group_resources.clear();
    }

    fn compile(&self, output: ResourceId) -> Vec<NodeId> {
        let writer_of: HashMap<ResourceId, NodeId> = self
            .nodes
            .iter()
            .flat_map(|(id, pass)| pass.writes().iter().map(|r| (*r, *id)).collect::<Vec<_>>())
            .collect();

        // Backwards reachability from output
        let mut stack = vec![output];
        let mut needed = HashSet::new();

        while let Some(res) = stack.pop() {
            let Some(&writer) = writer_of.get(&res) else {
                continue;
            };

            if needed.insert(writer) {
                let pass = self.pass_by_id(&writer);
                stack.extend(pass.reads());
            }
        }

        // Kahn's sort over needed nodes
        let mut in_degree: HashMap<NodeId, usize> = needed.iter().map(|&k| (k, 0)).collect();
        let mut edges: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for id in needed.iter() {
            let pass = self.pass_by_id(id);

            for res in pass.reads() {
                let Some(&writer) = writer_of.get(&res) else {
                    continue;
                };

                if needed.contains(&writer) {
                    edges.entry(writer).or_default().push(*id);
                    *in_degree.entry(*id).or_insert(0) += 1;
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

    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: ResourceId,
        surface_view: &wgpu::TextureView,
        world: &World,
    ) -> wgpu::CommandBuffer {
        let order = self.compile(output);

        // Allocate any missing resources
        for res in self.resources.values_mut() {
            if res.resource.is_none() {
                if matches!(res.desc, ResourceDescriptor::ExternalView) {
                    continue;
                }

                res.resource = Some(Self::allocate_resource(device, &res.desc));
            }
        }

        // Build writer map for upstream bind group lookup
        let writer_of: HashMap<ResourceId, NodeId> = self
            .nodes
            .iter()
            .flat_map(|(id, pass)| pass.writes().iter().map(|r| (*r, *id)).collect::<Vec<_>>())
            .collect();

        // Create/update bind groups for nodes that have layouts
        for (node_id, pass) in self.nodes.iter() {
            let Some(layout) = self.bind_group_layouts.get(node_id) else {
                continue;
            };

            let binding_resources = pass.binding_resources();
            if binding_resources.is_empty() {
                continue;
            }

            // Build sampler descriptors for this pass
            let sampler_descs = pass.samplers();
            let samplers: HashMap<u32, wgpu::Sampler> = sampler_descs
                .into_iter()
                .map(|(slot, desc)| (slot, device.create_sampler(&desc)))
                .collect();

            let entries: Vec<wgpu::BindGroupEntry> = binding_resources
                .iter()
                .filter_map(|br| {
                    let res = self.resources.get(&br.resource_id)?.resource.as_ref()?;
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

            let bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            }));
            self.bind_group_resources.insert(*node_id, bg);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RenderGraph"),
        });

        for node_id in order.iter() {
            // Build PassContext for this node
            let pass = self
                .nodes
                .iter()
                .find(|(id, _)| *id == *node_id)
                .map(|(_, pass)| pass)
                .unwrap();

            let mut views = HashMap::new();
            let mut buffers = HashMap::new();

            for id in pass.reads().iter().chain(pass.writes().iter()) {
                // Use the directly-passed surface view for the output resource
                if *id == output {
                    views.insert(*id, surface_view);
                    continue;
                }
                match self.resources.get(id).and_then(|r| r.resource.as_ref()) {
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

            // Current node's bind group
            let bind_group = self.bind_group_resources.get(node_id).map(|bg| bg.as_ref());

            // Upstream bind groups: for each read resource, find the writer node's bind group
            let mut upstream = HashMap::new();
            for res_id in pass.reads() {
                let Some(&writer_node) = writer_of.get(&res_id) else {
                    continue;
                };

                if let Some(bg) = self.bind_group_resources.get(&writer_node) {
                    upstream.insert(writer_node, bg.as_ref());
                }
            }

            let ctx = PassContext {
                views,
                buffers,
                bind_group,
                upstream,
            };

            // Get mutable reference to the pass for execution
            let pass = self
                .nodes
                .iter_mut()
                .find(|(id, _)| *id == *node_id)
                .map(|(_, pass)| pass)
                .unwrap();

            pass.execute(device, queue, &mut encoder, &ctx, world);

            for res_id in pass.writes() {
                if let Some(r) = self.resources.get_mut(&res_id) {
                    r.version += 1;
                }
            }
        }

        encoder.finish()
    }

    fn pass_by_id(&self, id: &NodeId) -> &dyn RenderPass {
        self.nodes
            .iter()
            .find(|(nid, _)| *nid == *id)
            .map(|(_, p)| p.as_ref())
            .unwrap()
    }

    fn allocate_resource(device: &wgpu::Device, desc: &ResourceDescriptor) -> AllocatedResource {
        match desc {
            ResourceDescriptor::ExternalView => {
                panic!("ExternalView must be populated via update_imported, not allocated")
            }
            ResourceDescriptor::Texture {
                size,
                format,
                usage,
            } => {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: *size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: *format,
                    usage: *usage,
                    view_formats: &[],
                });

                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                AllocatedResource::Texture(tex, view)
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

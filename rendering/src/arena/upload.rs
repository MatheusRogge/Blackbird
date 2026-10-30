/// Packs every fixed-size, CPU-written, GPU-read-only buffer of a graph into a
/// few large backing buffers. wgpu tracks usage per buffer, so anything the GPU
/// writes must stay out of here.
pub struct UploadArena {
    pages: Vec<wgpu::Buffer>,
    used_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UploadSlot {
    pub page: usize,
    pub offset: u64,
}

impl UploadArena {
    pub fn alignment(limits: &wgpu::Limits) -> u64 {
        limits
            .min_uniform_buffer_offset_alignment
            .max(limits.min_storage_buffer_offset_alignment) as u64
    }

    /// Returns the arena and, for each entry in `entries` (same order), the
    /// backing buffer and byte offset it lives at.
    pub fn build(
        device: &wgpu::Device,
        entries: &[(u64, wgpu::BufferUsages)],
    ) -> (Self, Vec<(wgpu::Buffer, u64)>) {
        let limits = device.limits();
        let sizes: Vec<u64> = entries.iter().map(|(size, _)| *size).collect();
        let slots = layout(&sizes, Self::alignment(&limits), limits.max_buffer_size);

        let page_count = slots.iter().map(|s| s.page + 1).max().unwrap_or(0);
        let mut page_sizes = vec![0u64; page_count];
        let mut page_usages = vec![wgpu::BufferUsages::COPY_DST; page_count];
        for (slot, (size, usage)) in slots.iter().zip(entries) {
            page_sizes[slot.page] = page_sizes[slot.page].max(slot.offset + size);
            page_usages[slot.page] |= *usage;
        }

        let pages: Vec<wgpu::Buffer> = page_sizes
            .iter()
            .zip(&page_usages)
            .map(|(&size, &usage)| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upload_arena"),
                    size: size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT),
                    usage,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let placed = slots
            .iter()
            .map(|slot| (pages[slot.page].clone(), slot.offset))
            .collect();

        let arena = Self {
            used_bytes: page_sizes.iter().sum(),
            pages,
        };
        (arena, placed)
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }
}

/// Bump-allocates `sizes` at `align`, starting a new page whenever the next
/// entry would cross `max_page_size`.
pub fn layout(sizes: &[u64], align: u64, max_page_size: u64) -> Vec<UploadSlot> {
    let mut page = 0;
    let mut cursor = 0u64;

    sizes
        .iter()
        .map(|&size| {
            let mut offset = cursor.next_multiple_of(align);
            if offset > 0 && offset + size > max_page_size {
                page += 1;
                offset = 0;
            }
            cursor = offset + size;
            UploadSlot { page, offset }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_every_entry() {
        let slots = layout(&[64, 4, 300, 16], 256, u64::MAX);
        let offsets: Vec<u64> = slots.iter().map(|s| s.offset).collect();
        assert_eq!(offsets, [0, 256, 512, 1024]);
        assert!(slots.iter().all(|s| s.page == 0));
    }

    #[test]
    fn spills_to_new_page() {
        let slots = layout(&[600, 600, 100], 256, 1024);
        assert_eq!(slots[0], UploadSlot { page: 0, offset: 0 });
        assert_eq!(slots[1], UploadSlot { page: 1, offset: 0 });
        assert_eq!(slots[2], UploadSlot { page: 1, offset: 768 });
    }

    #[test]
    fn oversized_entry_gets_its_own_page() {
        let slots = layout(&[2048, 16], 256, 1024);
        assert_eq!(slots[0], UploadSlot { page: 0, offset: 0 });
        assert_eq!(slots[1], UploadSlot { page: 1, offset: 0 });
    }
}

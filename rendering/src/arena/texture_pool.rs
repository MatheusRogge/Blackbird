use crate::resource::ResourceId;

/// One logical screen texture and the physical slot it was assigned to.
#[derive(Debug, Clone)]
pub struct AliasedTexture {
    pub id: ResourceId,
    pub writer: &'static str,
    pub format: wgpu::TextureFormat,
    pub first: usize,
    pub last: usize,
    pub slot: usize,
    pub bytes: u64,
}

/// Result of the last screen texture assignment, for debug views.
#[derive(Debug, Clone, Default)]
pub struct AliasingReport {
    /// Pass names in execution order; `first` / `last` index into this.
    pub passes: Vec<&'static str>,
    pub textures: Vec<AliasedTexture>,
    pub slots: usize,
    /// Screen textures allocated outside aliasing (read before written, or
    /// not used by any pass in the order).
    pub unaliased: usize,
}

impl AliasingReport {
    pub fn logical(&self) -> usize {
        self.textures.len() + self.unaliased
    }

    pub fn physical(&self) -> usize {
        self.slots + self.unaliased
    }

    pub fn logical_bytes(&self) -> u64 {
        self.textures.iter().map(|t| t.bytes).sum()
    }

    pub fn physical_bytes(&self) -> u64 {
        let mut slot_bytes = vec![0; self.slots];
        for t in &self.textures {
            slot_bytes[t.slot] = slot_bytes[t.slot].max(t.bytes);
        }
        slot_bytes.iter().sum()
    }
}

/// Span of pass indices (inclusive) during which a transient texture is live.
#[derive(Clone, Debug)]
pub struct Lifetime<K> {
    pub key: K,
    pub first: usize,
    pub last: usize,
}

/// Maps each lifetime to a physical slot. Two lifetimes share a slot only if
/// their keys match and one ends strictly before the other starts, so a pass
/// never reads and writes the same physical texture through different ids.
/// Returns the slot per lifetime (input order) and the number of slots.
pub fn assign<K: PartialEq>(lifetimes: &[Lifetime<K>]) -> (Vec<usize>, usize) {
    let mut by_start: Vec<usize> = (0..lifetimes.len()).collect();
    by_start.sort_by_key(|&i| lifetimes[i].first);

    let mut slot_last_use: Vec<(usize, usize)> = Vec::new();
    let mut slots = vec![0; lifetimes.len()];

    for i in by_start {
        let lifetime = &lifetimes[i];
        let free = slot_last_use.iter().position(|&(owner, last)| {
            last < lifetime.first && lifetimes[owner].key == lifetime.key
        });

        slots[i] = match free {
            Some(slot) => {
                slot_last_use[slot] = (i, lifetime.last);
                slot
            }
            None => {
                slot_last_use.push((i, lifetime.last));
                slot_last_use.len() - 1
            }
        };
    }

    (slots, slot_last_use.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lt(key: u32, first: usize, last: usize) -> Lifetime<u32> {
        Lifetime { key, first, last }
    }

    #[test]
    fn disjoint_same_key_share() {
        let (slots, count) = assign(&[lt(0, 0, 1), lt(0, 2, 3)]);
        assert_eq!(slots, [0, 0]);
        assert_eq!(count, 1);
    }

    #[test]
    fn touching_lifetimes_do_not_share() {
        let (slots, count) = assign(&[lt(0, 0, 2), lt(0, 2, 3)]);
        assert_eq!(slots, [0, 1]);
        assert_eq!(count, 2);
    }

    #[test]
    fn different_keys_never_share() {
        let (_, count) = assign(&[lt(0, 0, 0), lt(1, 1, 1)]);
        assert_eq!(count, 2);
    }

    #[test]
    fn report_counts_shared_slot_once() {
        let texture = |id, slot, bytes| AliasedTexture {
            id,
            writer: "pass",
            format: wgpu::TextureFormat::Rgba8Unorm,
            first: 0,
            last: 0,
            slot,
            bytes,
        };
        let report = AliasingReport {
            passes: vec!["pass"],
            textures: vec![texture(0, 0, 100), texture(1, 0, 100), texture(2, 1, 40)],
            slots: 2,
            unaliased: 1,
        };
        assert_eq!(report.logical(), 4);
        assert_eq!(report.physical(), 3);
        assert_eq!(report.logical_bytes(), 240);
        assert_eq!(report.physical_bytes(), 140);
    }

    #[test]
    fn reuses_slots_regardless_of_input_order() {
        let (slots, count) = assign(&[lt(0, 4, 5), lt(0, 0, 1), lt(0, 2, 3)]);
        assert_eq!(slots, [0, 0, 0]);
        assert_eq!(count, 1);
    }
}

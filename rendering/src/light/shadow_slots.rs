//! Which local lights get shadow maps, and in which slot.

use ultraviolet::Vec3;

use super::{MAX_SHADOWED_LIGHTS, ShadowCaster, SpotShadow};

/// A light that could cast a shadow this frame.
#[derive(Copy, Clone, Debug)]
pub struct ShadowCandidate {
    /// Higher is more important; candidates come sorted by it, highest first.
    pub score: f32,
    pub entry: u32,
    pub origin: Vec3,
    pub range: f32,
    pub spot: Option<SpotShadow>,
    /// Light half size (`ShadowCaster::half_size`).
    pub half_size: f32,
    /// The emitting side of a one-sided light (`ShadowCaster::front`).
    pub front: Option<Vec3>,
}

/// The shadow far plane for a light reaching `range`: the next power of two, and never smaller
/// than `current` while the light keeps its slot. A light's range follows its intensity, which
/// animations change every frame (a flickering lamp, dropping out to a few percent); a far plane
/// that followed it would drop the light's pages at every change. A far plane beyond the light's
/// reach costs nothing.
fn shadow_range(range: f32, current: Option<f32>) -> f32 {
    let far = 2f32.powf(range.max(1e-3).log2().ceil());
    current.map_or(far, |current| current.max(far))
}

#[derive(Default)]
pub struct ShadowSlots {
    slots: [Option<ShadowCaster>; MAX_SHADOWED_LIGHTS],
}

impl ShadowSlots {
    /// Shadows the first `MAX_SHADOWED_LIGHTS` of `candidates` (sorted by score). A light keeps
    /// its slot while it stays selected with the same shape, so its pages stay cached. Returns
    /// each slot's caster.
    pub fn update(&mut self, candidates: &[ShadowCandidate]) -> [Option<ShadowCaster>; MAX_SHADOWED_LIGHTS] {
        let selected = &candidates[..candidates.len().min(MAX_SHADOWED_LIGHTS)];
        let caster = |c: &ShadowCandidate, rank: usize, current: Option<&ShadowCaster>| ShadowCaster {
            entry: c.entry,
            origin: c.origin,
            range: shadow_range(c.range, current.map(|s| s.range)),
            spot: c.spot,
            front: c.front,
            half_size: c.half_size,
            rank: rank as u32,
        };

        let mut placed = [false; MAX_SHADOWED_LIGHTS];
        for slot in &mut self.slots {
            let Some(current) = *slot else { continue };
            match selected.iter().position(|c| c.entry == current.entry && c.spot.is_some() == current.spot.is_some()) {
                Some(r) => {
                    *slot = Some(caster(&selected[r], r, Some(&current)));
                    placed[r] = true;
                }
                None => *slot = None,
            }
        }
        for (r, c) in selected.iter().enumerate().filter(|&(r, _)| !placed[r]) {
            let Some(slot) = self.slots.iter_mut().find(|s| s.is_none()) else { break };
            *slot = Some(caster(c, r, None));
        }
        self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(entry: u32, score: f32, range: f32) -> ShadowCandidate {
        ShadowCandidate { score, entry, origin: Vec3::zero(), range, spot: None, half_size: 0.0, front: None }
    }

    #[test]
    fn keeps_slots_while_selected() {
        let mut slots = ShadowSlots::default();
        let first = slots.update(&[candidate(1, 3.0, 5.0), candidate(2, 2.0, 5.0)]);
        // Ranking flips: same slots, same casters, only the ranks swap.
        let second = slots.update(&[candidate(2, 3.0, 5.0), candidate(1, 2.0, 5.0)]);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.map(|c| c.entry), b.map(|c| c.entry));
            if let (Some(a), Some(b)) = (a, b) {
                assert!(a.renders_like(b));
            }
        }
        assert_eq!(second[0].map(|c| c.rank), Some(1));
    }

    #[test]
    fn a_dropped_light_frees_its_slot() {
        let mut slots = ShadowSlots::default();
        let candidates: Vec<_> = (0..MAX_SHADOWED_LIGHTS as u32 + 4).map(|i| candidate(i, 100.0 - i as f32, 5.0)).collect();
        let result = slots.update(&candidates);
        assert!(result.iter().all(Option::is_some));
        let shifted = slots.update(&candidates[1..]);
        assert!(shifted.iter().flatten().any(|c| c.entry == MAX_SHADOWED_LIGHTS as u32));
        assert!(shifted.iter().flatten().all(|c| c.entry != 0));
    }

    #[test]
    fn flickering_range_keeps_the_far_plane() {
        assert_eq!(shadow_range(5.0, None), 8.0);
        assert_eq!(shadow_range(7.9, Some(8.0)), 8.0);
        assert_eq!(shadow_range(1.2, Some(8.0)), 8.0, "a dropout keeps the far plane");
        assert_eq!(shadow_range(8.5, Some(8.0)), 16.0);
    }
}

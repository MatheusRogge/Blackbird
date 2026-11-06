//! Which local lights get shadow maps, in which slot, and at what tile size in the atlas.

use ultraviolet::Vec3;

use super::{
    MAX_SHADOWED_LIGHTS, ShadowCaster, SpotShadow,
    shadow_atlas::{MAX_TILE, MIN_TILE, ShadowAtlas, Tile},
};

/// A light that could cast a shadow this frame.
#[derive(Copy, Clone, Debug)]
pub struct ShadowCandidate {
    /// Higher is more important; candidates come sorted by it, highest first.
    pub score: f32,
    pub entry: u32,
    pub origin: Vec3,
    pub range: f32,
    pub spot: Option<SpotShadow>,
    /// Light half size (`ShadowedLights::lights` w).
    pub half_size: f32,
    /// How much of the view the light's range can fill: range / distance to the camera.
    pub coverage: f32,
}

/// A size change needs the coverage this far past the tier boundary, so lights near a boundary
/// don't flip between sizes (and re-render) as the camera moves.
const HYSTERESIS: f32 = 1.25;

/// Tile size for `coverage`: `MAX_TILE` from coverage 1 (the camera inside the range) and halving
/// per halving of coverage, down to `MIN_TILE`. `current` sticks until the coverage leaves its tier
/// by more than `HYSTERESIS`.
fn tile_size(coverage: f32, current: Option<u32>) -> u32 {
    let tier_floor = |size: u32| size as f32 / MAX_TILE as f32;
    if let Some(size) = current {
        let low = if size == MIN_TILE { 0.0 } else { tier_floor(size) / HYSTERESIS };
        let high = if size == MAX_TILE { f32::INFINITY } else { tier_floor(size * 2) * HYSTERESIS };
        if (low..high).contains(&coverage) {
            return size;
        }
    }
    let mut size = MAX_TILE;
    while size > MIN_TILE && coverage < tier_floor(size) {
        size /= 2;
    }
    size
}

#[derive(Copy, Clone)]
struct Slot {
    caster: ShadowCaster,
    half_size: f32,
}

#[derive(Default)]
pub struct ShadowSlots {
    atlas: ShadowAtlas,
    slots: [Option<Slot>; MAX_SHADOWED_LIGHTS],
}

impl ShadowSlots {
    /// Shadows the first `MAX_SHADOWED_LIGHTS` of `candidates` (sorted by score). A light keeps
    /// its slot and tiles while it stays selected with the same shape and tile size, so its map
    /// stays cached. Returns each slot's caster and half size.
    pub fn update(&mut self, candidates: &[ShadowCandidate]) -> [Option<(ShadowCaster, f32)>; MAX_SHADOWED_LIGHTS] {
        let selected = &candidates[..candidates.len().min(MAX_SHADOWED_LIGHTS)];
        let rank = |entry: u32| selected.iter().position(|c| c.entry == entry);

        let mut pending: Vec<(usize, usize, u32)> = Vec::new();
        for slot in 0..MAX_SHADOWED_LIGHTS {
            let Some(current) = self.slots[slot] else { continue };
            let Some(r) = rank(current.caster.entry) else {
                self.release(slot);
                continue;
            };
            let c = &selected[r];
            let size = tile_size(c.coverage, Some(current.caster.tiles[0].size));
            if size != current.caster.tiles[0].size || c.spot.is_some() != current.caster.spot.is_some() {
                self.release(slot);
                pending.push((r, slot, size));
            } else {
                let caster = ShadowCaster { origin: c.origin, range: c.range, spot: c.spot, ..current.caster };
                self.slots[slot] = Some(Slot { caster, half_size: c.half_size });
            }
        }
        for (r, c) in selected.iter().enumerate() {
            let placed = self.slots.iter().flatten().any(|s| s.caster.entry == c.entry)
                || pending.iter().any(|&(pr, ..)| pr == r);
            if placed {
                continue;
            }
            let Some(slot) = (0..MAX_SHADOWED_LIGHTS)
                .find(|&s| self.slots[s].is_none() && pending.iter().all(|&(_, ps, _)| ps != s))
            else {
                break;
            };
            pending.push((r, slot, tile_size(c.coverage, None)));
        }

        pending.sort_by_key(|&(r, ..)| r);
        for (r, slot, size) in pending {
            let c = &selected[r];
            let faces = if c.spot.is_some() { 1 } else { 6 };
            if let Some(tiles) = self.alloc_faces(faces, size, r, &rank) {
                let caster = ShadowCaster { entry: c.entry, origin: c.origin, range: c.range, spot: c.spot, tiles };
                self.slots[slot] = Some(Slot { caster, half_size: c.half_size });
            }
        }
        self.slots.map(|s| s.map(|s| (s.caster, s.half_size)))
    }

    /// `faces` tiles of `size`, or of a smaller size if the atlas is too full. When even
    /// `MIN_TILE` doesn't fit, evicts the lowest-ranked slot ranked below `rank` and retries.
    fn alloc_faces(&mut self, faces: usize, size: u32, rank: usize, rank_of: &impl Fn(u32) -> Option<usize>) -> Option<[Tile; 6]> {
        loop {
            let mut size = size;
            while size >= MIN_TILE {
                let mut tiles = Vec::with_capacity(faces);
                while tiles.len() < faces {
                    let Some(tile) = self.atlas.alloc(size) else { break };
                    tiles.push(tile);
                }
                if tiles.len() == faces {
                    let mut out = [tiles[0]; 6];
                    out[..faces].copy_from_slice(&tiles);
                    return Some(out);
                }
                for tile in tiles {
                    self.atlas.free(tile);
                }
                size /= 2;
            }
            let victim = (0..MAX_SHADOWED_LIGHTS)
                .filter_map(|s| Some((s, rank_of(self.slots[s]?.caster.entry)?)))
                .filter(|&(_, r)| r > rank)
                .max_by_key(|&(_, r)| r)?;
            self.release(victim.0);
        }
    }

    fn release(&mut self, slot: usize) {
        if let Some(s) = self.slots[slot].take() {
            for &tile in &s.caster.tiles[..s.caster.faces()] {
                self.atlas.free(tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(entry: u32, score: f32, coverage: f32) -> ShadowCandidate {
        ShadowCandidate { score, entry, origin: Vec3::zero(), range: 5.0, spot: None, half_size: 0.0, coverage }
    }

    #[test]
    fn tile_size_tiers_with_hysteresis() {
        assert_eq!(tile_size(2.0, None), 512);
        assert_eq!(tile_size(0.6, None), 256);
        assert_eq!(tile_size(0.01, None), 128);
        // Just below the 512 tier: a 512 tile sticks, a 256 one doesn't grow.
        assert_eq!(tile_size(0.9, Some(512)), 512);
        assert_eq!(tile_size(1.1, Some(256)), 256);
        assert_eq!(tile_size(1.3, Some(256)), 512);
    }

    #[test]
    fn keeps_slots_and_tiles_while_selected() {
        let mut slots = ShadowSlots::default();
        let first = slots.update(&[candidate(1, 3.0, 2.0), candidate(2, 2.0, 2.0)]);
        // Ranking flips: same slots, same tiles.
        let second = slots.update(&[candidate(2, 3.0, 2.0), candidate(1, 2.0, 2.0)]);
        assert_eq!(first, second);
    }

    #[test]
    fn lights_spread_over_a_scene_all_fit() {
        let mut slots = ShadowSlots::default();
        let candidates: Vec<_> =
            (0..MAX_SHADOWED_LIGHTS as u32).map(|i| candidate(i, 100.0 - i as f32, 2.0 / (1.0 + i as f32))).collect();
        assert!(slots.update(&candidates).iter().all(Option::is_some));
    }

    #[test]
    fn a_full_atlas_squeezes_out_the_least_important() {
        let mut slots = ShadowSlots::default();
        // 32 cube lights all wanting 512² faces: 192 faces against the 64 the atlas holds.
        let candidates: Vec<_> = (0..MAX_SHADOWED_LIGHTS as u32).map(|i| candidate(i, 100.0 - i as f32, 2.0)).collect();
        let result = slots.update(&candidates);
        let shadowed: Vec<_> = result.iter().flatten().map(|(c, _)| c.entry).collect();
        assert!(shadowed.len() >= 10, "only {} lights got tiles", shadowed.len());
        let first_missing = (0..MAX_SHADOWED_LIGHTS as u32).find(|e| !shadowed.contains(e)).expect("some light misses out");
        assert!(shadowed.iter().all(|&e| e < first_missing), "a lower-ranked light kept tiles over {first_missing}");

        // A light dropping out frees its tiles for the next one.
        let mut fewer = candidates.clone();
        fewer.remove(0);
        let after = slots.update(&fewer);
        assert!(after.iter().flatten().any(|(c, _)| c.entry == first_missing));
    }
}

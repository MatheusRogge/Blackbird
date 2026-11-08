//! Which local lights get shadow maps, in which slot, and at what tile size per face in the atlas.

use ultraviolet::Vec3;

use super::{
    MAX_SHADOWED_LIGHTS, ShadowCaster, SpotShadow,
    shadow_atlas::{ATLAS_SIZE, MAX_TILE, MIN_TILE, ShadowAtlas, Tile},
};

/// Most atlas texels one light's faces take (half the atlas: one `MAX_TILE` face and four at
/// half), so the light nearest the camera can't crowd out every other.
const MAX_LIGHT_TEXELS: u64 = (ATLAS_SIZE as u64).pow(2) / 2;
/// Most texels all lights' faces plan for together; the rest is slack for fragmentation.
const PLAN_TEXELS: u64 = (ATLAS_SIZE as u64).pow(2) * 3 / 4;

fn texels(sizes: &[u32; 6]) -> u64 {
    sizes.iter().map(|&s| u64::from(s).pow(2)).sum()
}

/// `wanted` (by rank) shrunk to fit: each light under `MAX_LIGHT_TEXELS`, then all of them under
/// `PLAN_TEXELS`, the least important light first. Each step halves that light's largest face,
/// the one that needs it least (`coverage`) on ties. Depends only on its inputs, so the plan,
/// and the tiles, stay put while they do.
fn plan_sizes(wanted: &[[u32; 6]], coverage: &[[f32; 6]]) -> Vec<[u32; 6]> {
    let mut plan = wanted.to_vec();
    // The face to halve among `faces` (light, face): the last light's, then the largest, then
    // the least needed.
    let halve = |plan: &mut Vec<[u32; 6]>, faces: &mut dyn Iterator<Item = (usize, usize)>| {
        let pick = faces.filter(|&(l, f)| plan[l][f] > MIN_TILE).fold(None::<(usize, usize)>, |best, (l, f)| match best {
            Some((bl, bf)) if (bl, plan[bl][bf], coverage[l][f]) > (l, plan[l][f], coverage[bl][bf]) => Some((bl, bf)),
            _ => Some((l, f)),
        });
        pick.map(|(l, f)| plan[l][f] /= 2).is_some()
    };
    for l in 0..plan.len() {
        while texels(&plan[l]) > MAX_LIGHT_TEXELS && halve(&mut plan, &mut (0..6).map(|f| (l, f))) {}
    }
    let lights = plan.len();
    while plan.iter().map(texels).sum::<u64>() > PLAN_TEXELS
        && halve(&mut plan, &mut (0..lights).flat_map(|l| (0..6).map(move |f| (l, f))))
    {}
    plan
}

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
    /// The emitting side of a one-sided light (`shadow_lod::FaceShape::front`).
    pub front: Option<Vec3>,
    /// Per face, the wanted tile size as a fraction of `MAX_TILE` (`shadow_lod::face_coverage`);
    /// 0 for a face that needs no map.
    pub coverage: [f32; 6],
}

/// A size change needs the coverage this far past the tier boundary, so faces near a boundary
/// don't flip between sizes (and re-render) as the camera moves.
const HYSTERESIS: f32 = 1.25;

/// Tile size for `coverage`: `MAX_TILE` from coverage 1 and halving per halving of coverage, down
/// to `MIN_TILE`; 0 (no tile) for coverage 0. `current` sticks until the coverage leaves its tier
/// by more than `HYSTERESIS`.
fn tile_size(coverage: f32, current: Option<u32>) -> u32 {
    if coverage <= 0.0 {
        return 0;
    }
    let tier_floor = |size: u32| size as f32 / MAX_TILE as f32;
    if let Some(size) = current.filter(|&s| s > 0) {
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

fn tile_sizes(coverage: [f32; 6], faces: usize, current: Option<[u32; 6]>) -> [u32; 6] {
    std::array::from_fn(|f| if f < faces { tile_size(coverage[f], current.map(|c| c[f])) } else { 0 })
}

#[derive(Default)]
pub struct ShadowSlots {
    atlas: ShadowAtlas,
    slots: [Option<ShadowCaster>; MAX_SHADOWED_LIGHTS],
    /// The sizes each slot's faces asked for (`tile_sizes`, whose hysteresis starts from these).
    wanted: [[u32; 6]; MAX_SHADOWED_LIGHTS],
    /// Those sizes after `plan_sizes`. A slot re-allocates when its plan changes, not when a
    /// fragmented atlas gave it less, so a squeezed face doesn't re-allocate every frame.
    planned: [[u32; 6]; MAX_SHADOWED_LIGHTS],
}

impl ShadowSlots {
    /// Shadows the first `MAX_SHADOWED_LIGHTS` of `candidates` (sorted by score). A light keeps
    /// its slot and tiles while it stays selected with the same shape and tile size, so its map
    /// stays cached. Returns each slot's caster.
    pub fn update(&mut self, candidates: &[ShadowCandidate]) -> [Option<ShadowCaster>; MAX_SHADOWED_LIGHTS] {
        let selected = &candidates[..candidates.len().min(MAX_SHADOWED_LIGHTS)];
        let rank = |entry: u32| selected.iter().position(|c| c.entry == entry);
        let same_light = |slot: &Option<ShadowCaster>, c: &ShadowCandidate| {
            slot.is_some_and(|s| s.entry == c.entry && s.spot.is_some() == c.spot.is_some())
        };

        let wanted: Vec<[u32; 6]> = selected
            .iter()
            .map(|c| {
                let current = self.slots.iter().position(|s| same_light(s, c)).map(|slot| self.wanted[slot]);
                tile_sizes(c.coverage, if c.spot.is_some() { 1 } else { 6 }, current)
            })
            .collect();
        let coverage: Vec<[f32; 6]> = selected.iter().map(|c| c.coverage).collect();
        let plan = plan_sizes(&wanted, &coverage);

        let mut pending: Vec<(usize, usize)> = Vec::new();
        for slot in 0..MAX_SHADOWED_LIGHTS {
            let Some(current) = self.slots[slot] else { continue };
            let Some(r) = rank(current.entry) else {
                self.release(slot);
                continue;
            };
            let c = &selected[r];
            if plan[r] != self.planned[slot] || !same_light(&self.slots[slot], c) {
                self.release(slot);
                pending.push((r, slot));
                continue;
            }
            self.wanted[slot] = wanted[r];
            self.slots[slot] = Some(ShadowCaster {
                origin: c.origin,
                range: c.range,
                spot: c.spot,
                half_size: c.half_size,
                rank: r as u32,
                ..current
            });
        }
        for (r, c) in selected.iter().enumerate() {
            let placed = self.slots.iter().flatten().any(|s| s.entry == c.entry) || pending.iter().any(|&(pr, _)| pr == r);
            if placed {
                continue;
            }
            let Some(slot) =
                (0..MAX_SHADOWED_LIGHTS).find(|&s| self.slots[s].is_none() && pending.iter().all(|&(_, ps)| ps != s))
            else {
                break;
            };
            pending.push((r, slot));
        }

        pending.sort_by_key(|&(r, _)| r);
        for (r, slot) in pending {
            let c = &selected[r];
            if let Some(tiles) = self.alloc_faces(plan[r], r, &rank) {
                self.wanted[slot] = wanted[r];
                self.planned[slot] = plan[r];
                self.slots[slot] = Some(ShadowCaster {
                    entry: c.entry,
                    origin: c.origin,
                    range: c.range,
                    spot: c.spot,
                    tiles,
                    half_size: c.half_size,
                    rank: r as u32,
                });
            }
        }
        self.slots
    }

    /// A tile per face of `sizes` (none where it is 0), halving the largest face until they fit
    /// when the atlas is too fragmented. When even `MIN_TILE` faces don't fit, evicts the
    /// lowest-ranked slot ranked below `rank` and retries.
    fn alloc_faces(&mut self, sizes: [u32; 6], rank: usize, rank_of: &impl Fn(u32) -> Option<usize>) -> Option<[Tile; 6]> {
        loop {
            let mut sizes = sizes;
            loop {
                if let Some(tiles) = self.alloc_all(sizes) {
                    return Some(tiles);
                }
                let Some(largest) = sizes.iter_mut().filter(|s| **s > MIN_TILE).max_by_key(|s| **s) else { break };
                *largest /= 2;
            }
            let victim = (0..MAX_SHADOWED_LIGHTS)
                .filter_map(|s| Some((s, rank_of(self.slots[s]?.entry)?)))
                .filter(|&(_, r)| r > rank)
                .max_by_key(|&(_, r)| r)?;
            self.release(victim.0);
        }
    }

    fn alloc_all(&mut self, sizes: [u32; 6]) -> Option<[Tile; 6]> {
        let mut tiles = [Tile { x: 0, y: 0, size: 0 }; 6];
        // Largest first, so small tiles don't fragment the blocks big ones need.
        let mut order: Vec<usize> = (0..6).filter(|&f| sizes[f] > 0).collect();
        order.sort_by_key(|&f| std::cmp::Reverse(sizes[f]));
        for (n, &f) in order.iter().enumerate() {
            match self.atlas.alloc(sizes[f]) {
                Some(tile) => tiles[f] = tile,
                None => {
                    for &g in &order[..n] {
                        self.atlas.free(tiles[g]);
                    }
                    return None;
                }
            }
        }
        Some(tiles)
    }

    fn release(&mut self, slot: usize) {
        if let Some(s) = self.slots[slot].take() {
            for &tile in s.tiles.iter().filter(|t| t.size > 0) {
                self.atlas.free(tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(entry: u32, score: f32, coverage: f32) -> ShadowCandidate {
        ShadowCandidate { score, entry, origin: Vec3::zero(), range: 5.0, spot: None, half_size: 0.0, front: None, coverage: [coverage; 6] }
    }

    #[test]
    fn tile_size_tiers_with_hysteresis() {
        assert_eq!(tile_size(2.0, None), MAX_TILE);
        assert_eq!(tile_size(0.6, None), MAX_TILE / 2);
        assert_eq!(tile_size(0.001, None), MIN_TILE);
        assert_eq!(tile_size(0.0, None), 0);
        // Just below the top tier: a top tile sticks, a half one doesn't grow.
        assert_eq!(tile_size(0.9, Some(MAX_TILE)), MAX_TILE);
        assert_eq!(tile_size(1.1, Some(MAX_TILE / 2)), MAX_TILE / 2);
        assert_eq!(tile_size(1.3, Some(MAX_TILE / 2)), MAX_TILE);
    }

    #[test]
    fn keeps_slots_and_tiles_while_selected() {
        let mut slots = ShadowSlots::default();
        let first = slots.update(&[candidate(1, 3.0, 2.0), candidate(2, 2.0, 2.0)]);
        // Ranking flips: same slots, same tiles, only the ranks swap.
        let second = slots.update(&[candidate(2, 3.0, 2.0), candidate(1, 2.0, 2.0)]);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(a), Some(b)) = (a, b) {
                assert!(a.renders_like(b));
            }
        }
        assert_eq!(second[0].map(|c| c.rank), Some(1));
    }

    #[test]
    fn lights_spread_over_a_scene_all_fit() {
        let mut slots = ShadowSlots::default();
        let candidates: Vec<_> =
            (0..MAX_SHADOWED_LIGHTS as u32).map(|i| candidate(i, 100.0 - i as f32, 2.0 / (1.0 + i as f32))).collect();
        assert!(slots.update(&candidates).iter().all(Option::is_some));
    }

    fn caster_texels(caster: &ShadowCaster) -> u64 {
        texels(&caster.tiles.map(|t| t.size))
    }

    #[test]
    fn a_full_atlas_favours_the_most_important() {
        let mut slots = ShadowSlots::default();
        // 32 cube lights all wanting the largest faces: far more than the atlas holds.
        let candidates: Vec<_> = (0..MAX_SHADOWED_LIGHTS as u32).map(|i| candidate(i, 100.0 - i as f32, 2.0)).collect();
        let result = slots.update(&candidates);
        let by_rank: Vec<u64> = (0..MAX_SHADOWED_LIGHTS as u32)
            .map(|e| result.iter().flatten().find(|c| c.entry == e).map_or(0, caster_texels))
            .collect();
        assert!(by_rank[0] <= MAX_LIGHT_TEXELS, "one light took {} texels", by_rank[0]);
        assert!(by_rank.windows(2).all(|w| w[0] >= w[1]), "{by_rank:?}");
        assert!(by_rank.iter().all(|&t| t > 0), "{by_rank:?}");
        assert!(by_rank.iter().sum::<u64>() <= PLAN_TEXELS);
    }

    #[test]
    fn faces_get_their_own_sizes() {
        let mut slots = ShadowSlots::default();
        let mut c = candidate(1, 1.0, 0.0);
        c.coverage = [2.0, 0.0, 0.3, 0.001, 0.6, 0.3];
        let caster = slots.update(&[c])[0].expect("shadowed");
        let sizes = caster.tiles.map(|t| t.size);
        assert_eq!(sizes, [MAX_TILE, 0, MAX_TILE / 4, MIN_TILE, MAX_TILE / 2, MAX_TILE / 4]);
        // Unchanged coverage keeps the same tiles, so the maps stay cached.
        let again = slots.update(&[c])[0].expect("shadowed");
        assert_eq!(again.tiles, caster.tiles);
    }
}

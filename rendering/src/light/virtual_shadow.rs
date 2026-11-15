//! Virtual shadow maps for local lights (a scaled-down take on Unreal 5's): every face of a
//! shadowed light is a `VIRTUAL_SIZE`² map with a mip chain, split into `PAGE_SIZE`² pages, and
//! only the pages visible pixels ask for are backed by one of the `POOL_PAGES` physical pages of
//! the atlas. The coarsest mip of each face (one page: the whole face) is always resident, so a
//! lookup can always fall back to it. This module is the CPU side: page indexing, the physical
//! pool (least recently requested pages are evicted first) and the page table the shader reads.

use std::collections::HashMap;

use ultraviolet::{Mat4, Vec3, Vec4};

use super::MAX_SHADOWED_LIGHTS;

/// Side of the atlas the physical pages tile, in texels.
pub const ATLAS_SIZE: u32 = 4096;
pub const PAGE_SIZE: u32 = 128;
pub const VIRTUAL_SIZE: u32 = 8192;
/// Mips 0 (`VIRTUAL_SIZE`) to `MIPS - 1` (one page).
pub const MIPS: u32 = (VIRTUAL_SIZE / PAGE_SIZE).ilog2() + 1;
/// Pages of every mip of one face.
pub const PAGES_PER_FACE: u32 = mip_offset(MIPS);
/// Physical pages per atlas side, and in all.
pub const POOL_SIDE: u32 = ATLAS_SIZE / PAGE_SIZE;
pub const POOL_PAGES: u32 = POOL_SIDE * POOL_SIDE;
/// Virtual pages of every face of every slot: the page table's length.
pub const TABLE_LEN: usize = MAX_SHADOWED_LIGHTS * 6 * PAGES_PER_FACE as usize;
/// `u32` words of the page request bit mask (one bit per table entry).
pub const REQUEST_WORDS: usize = TABLE_LEN.div_ceil(32);

/// Cells per side of a face's invalidation grid (`VirtualShadows::invalidate`): mip 2's pages.
pub const DIRTY_GRID: u32 = mip_pages(DIRTY_MIP);
const DIRTY_MIP: u32 = 2;
/// One face's invalidation grid.
pub type DirtyCells = [bool; (DIRTY_GRID * DIRTY_GRID) as usize];

const NOT_RESIDENT: u16 = u16::MAX;

/// The latest page request mask read back from the GPU (`pass::shadow_pages`), `REQUEST_WORDS`
/// long; the shadow pass takes it.
pub type SharedPageRequests = std::sync::Arc<std::sync::Mutex<Option<Vec<u32>>>>;

/// Pages per side at `mip`.
pub const fn mip_pages(mip: u32) -> u32 {
    (VIRTUAL_SIZE / PAGE_SIZE) >> mip
}

/// Index of `mip`'s first page within a face.
pub const fn mip_offset(mip: u32) -> u32 {
    let mut offset = 0;
    let mut m = 0;
    while m < mip {
        offset += mip_pages(m) * mip_pages(m);
        m += 1;
    }
    offset
}

/// Calls `f` with the index of every set bit of `mask`.
fn for_each_bit(mask: &[u32], mut f: impl FnMut(u32)) {
    for (word, &bits) in mask.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            f((word * 32) as u32 + bits.trailing_zeros());
            bits &= bits - 1;
        }
    }
}

/// The mip of the virtual page at `index` (`PageKey::index`).
fn mip_of(index: u32) -> u32 {
    let rest = index % PAGES_PER_FACE;
    (1..MIPS).take_while(|&mip| rest >= mip_offset(mip)).count() as u32
}

/// One virtual page: `slot * 6 + face` its face, `x` / `y` its page within `mip`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PageKey {
    pub face: u32,
    pub mip: u32,
    pub x: u32,
    pub y: u32,
}

impl PageKey {
    pub fn index(self) -> u32 {
        self.face * PAGES_PER_FACE + mip_offset(self.mip) + self.y * mip_pages(self.mip) + self.x
    }

    pub fn from_index(index: u32) -> Self {
        let face = index / PAGES_PER_FACE;
        let mut rest = index % PAGES_PER_FACE;
        let mut mip = 0;
        while rest >= mip_pages(mip) * mip_pages(mip) {
            rest -= mip_pages(mip) * mip_pages(mip);
            mip += 1;
        }
        Self { face, mip, x: rest % mip_pages(mip), y: rest / mip_pages(mip) }
    }

    /// The coarsest page of `face`, which covers all of it.
    pub fn whole(face: u32) -> Self {
        Self { face, mip: MIPS - 1, x: 0, y: 0 }
    }

    /// `face_view_proj` narrowed to this page: the page's part of the face's clip square stretched
    /// over all of it. Depth (clip z and w) is unchanged, so pages compare against the face's.
    pub fn view_proj(self, face_view_proj: Mat4) -> Mat4 {
        let n = mip_pages(self.mip) as f32;
        let scale = n;
        // Page centre in NDC; texture v runs down while NDC y runs up.
        let cx = (self.x as f32 + 0.5) / n * 2.0 - 1.0;
        let cy = 1.0 - (self.y as f32 + 0.5) / n * 2.0;
        let narrow = Mat4::new(
            Vec4::new(scale, 0.0, 0.0, 0.0),
            Vec4::new(0.0, scale, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(-scale * cx, -scale * cy, 0.0, 1.0),
        );
        narrow * face_view_proj
    }
}

/// Top-left texel of physical page `page` in the atlas.
pub fn page_origin(page: u16) -> (u32, u32) {
    (u32::from(page) % POOL_SIDE * PAGE_SIZE, u32::from(page) / POOL_SIDE * PAGE_SIZE)
}

/// Whether a cube face's whole frustum lies behind the plane facing `front` (the dark side of a
/// one-sided light), so it needs no map. Faces in `lighting.wgsl`'s order: +x, −x, +y, −y, +z, −z.
pub fn face_behind(face: usize, front: Vec3) -> bool {
    let axes = [Vec3::unit_x(), Vec3::unit_y(), Vec3::unit_z()];
    let axis = axes[face / 2] * if face.is_multiple_of(2) { 1.0 } else { -1.0 };
    let (a, b) = (axes[(face / 2 + 1) % 3], axes[(face / 2 + 2) % 3]);
    [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)].iter().all(|&(s, t)| (axis + a * s + b * t).dot(front) <= 0.0)
}

#[derive(Copy, Clone)]
struct Physical {
    /// The virtual page it backs (`PageKey::index`).
    owner: Option<u32>,
    last_requested: u64,
    /// Holds its owner's current map. A new page's table entry waits for it; an invalidated one
    /// keeps pointing at the old map meanwhile.
    rendered: bool,
    /// When its map was last rendered.
    rendered_frame: u64,
    /// A face's coarsest page, never evicted while its slot is shadowed.
    pinned: bool,
}

pub struct VirtualShadows {
    /// Per virtual page: physical page + 1 once rendered, else 0. What the shader reads.
    table: Vec<u32>,
    /// Per virtual page: its physical page, rendered or not, or `NOT_RESIDENT`.
    resident: Vec<u16>,
    pages: Vec<Physical>,
    free: Vec<u16>,
    /// Faces (`slot * 6 + face`) whose table entries changed since the last `take_dirty_faces`.
    dirty: Vec<bool>,
    enabled: [[bool; 6]; MAX_SHADOWED_LIGHTS],
}

impl Default for VirtualShadows {
    fn default() -> Self {
        Self {
            table: vec![0; TABLE_LEN],
            resident: vec![NOT_RESIDENT; TABLE_LEN],
            pages: vec![Physical { owner: None, last_requested: 0, rendered: false, rendered_frame: 0, pinned: false }; POOL_PAGES as usize],
            free: (0..POOL_PAGES as u16).rev().collect(),
            dirty: vec![false; MAX_SHADOWED_LIGHTS * 6],
            enabled: [[false; 6]; MAX_SHADOWED_LIGHTS],
        }
    }
}

impl VirtualShadows {
    pub fn table(&self) -> &[u32] {
        &self.table
    }

    pub fn face_enabled(&self, face: u32) -> bool {
        self.enabled[face as usize / 6][face as usize % 6]
    }

    /// Starts `slot` over for a new or changed light: drops its pages and makes the coarsest page
    /// of each of `faces` resident (pinned, to be rendered). Faces left out get no pages.
    pub fn reset_slot(&mut self, slot: usize, faces: [bool; 6], frame: u64) {
        self.release_slot(slot);
        self.enabled[slot] = faces;
        for face in (0..6).filter(|&f| faces[f]) {
            let index = PageKey::whole((slot * 6 + face) as u32).index();
            if let Some(page) = self.allocate(index, frame, true) {
                self.pages[page as usize].pinned = true;
            }
        }
    }

    /// Frees every page of `slot` and disables its faces.
    pub fn release_slot(&mut self, slot: usize) {
        self.enabled[slot] = [false; 6];
        for page in 0..POOL_PAGES as usize {
            if self.pages[page].owner.is_some_and(|o| (o / PAGES_PER_FACE) as usize / 6 == slot) {
                self.evict(page as u16);
            }
        }
    }

    /// Marks the resident pages overlapping a set cell of their face's grid in `dirty` (per face
    /// `slot * 6 + face`: `DIRTY_GRID`² cells, row-major, v down) to render again. Until they do,
    /// the page table keeps their old map.
    pub fn invalidate(&mut self, dirty: &HashMap<u32, DirtyCells>) {
        for p in &mut self.pages {
            let Some(owner) = p.owner else { continue };
            let Some(cells) = dirty.get(&(owner / PAGES_PER_FACE)) else { continue };
            let key = PageKey::from_index(owner);
            let (lo, hi) = if key.mip <= DIRTY_MIP {
                let shift = DIRTY_MIP - key.mip;
                ((key.x >> shift, key.y >> shift), (key.x >> shift, key.y >> shift))
            } else {
                let shift = key.mip - DIRTY_MIP;
                ((key.x << shift, key.y << shift), (((key.x + 1) << shift) - 1, ((key.y + 1) << shift) - 1))
            };
            if (lo.1..=hi.1).any(|y| (lo.0..=hi.0).any(|x| cells[(y * DIRTY_GRID + x) as usize])) {
                p.rendered = false;
            }
        }
    }

    /// Applies a page request mask (a bit per table entry): requested resident pages are marked
    /// used at `frame`, and missing ones of enabled faces are allocated, coarsest mip first so a
    /// full pool still serves the fallbacks, evicting the least recently requested page when the
    /// pool is full. Requests the pool can't serve stay unbacked; lookups fall back to coarser
    /// pages.
    pub fn request(&mut self, mask: &[u32], frame: u64) {
        let mut missing = false;
        for_each_bit(mask, |index| {
            if index as usize >= TABLE_LEN || !self.face_enabled(index / PAGES_PER_FACE) {
                return;
            }
            match self.resident[index as usize] {
                NOT_RESIDENT => missing = true,
                page => self.pages[page as usize].last_requested = frame,
            }
        });
        let mut victims: Vec<u16> = (0..POOL_PAGES as u16)
            .filter(|&page| {
                let p = &self.pages[page as usize];
                p.owner.is_some() && !p.pinned && p.last_requested < frame
            })
            .collect();
        // With a full pool, far more pages are asked for than can ever be served (the corridor
        // with 24 lights: ~17 000 a frame); sorting them all was most of the pass's CPU time.
        if !missing || self.free.is_empty() && victims.is_empty() {
            return;
        }
        let mut by_mip: [Vec<u32>; MIPS as usize] = Default::default();
        for_each_bit(mask, |index| {
            if (index as usize) < TABLE_LEN
                && self.face_enabled(index / PAGES_PER_FACE)
                && self.resident[index as usize] == NOT_RESIDENT
            {
                by_mip[mip_of(index) as usize].push(index);
            }
        });
        victims.sort_by_key(|&page| self.pages[page as usize].last_requested);
        let mut victims = victims.into_iter();
        for &index in by_mip.iter().rev().flatten() {
            let page = match self.free.pop() {
                Some(page) => page,
                None => {
                    let Some(victim) = victims.next() else { break };
                    self.evict(victim);
                    self.free.pop().expect("an evicted page is free")
                }
            };
            self.assign(page, index, frame, false);
        }
    }

    /// Resident pages still to render: pinned ones (all of them: a face without its coarsest
    /// page has no shadow at all), then up to `budget` others. Pages without a map yet go first,
    /// coarsest mip first; then invalidated ones, which still show their old map, longest
    /// unrendered first, so a backlog of them (Osaka: ~500 under swaying curtains) takes turns.
    pub fn pending(&self, budget: usize) -> Vec<(PageKey, u16)> {
        let mut pinned = Vec::new();
        let mut others = Vec::new();
        for (page, p) in self.pages.iter().enumerate() {
            let Some(owner) = p.owner.filter(|_| !p.rendered) else { continue };
            let entry = (PageKey::from_index(owner), page as u16);
            if p.pinned { pinned.push(entry) } else { others.push(entry) }
        }
        others.sort_by_key(|&(key, page)| {
            let stale = self.table[key.index() as usize] != 0;
            (stale, if stale { self.pages[page as usize].rendered_frame } else { u64::from(MIPS - key.mip) }, key.index())
        });
        others.truncate(budget);
        pinned.extend(others);
        pinned
    }

    /// Marks `page` rendered at `frame`: the page table now points its owner at it.
    pub fn rendered(&mut self, page: u16, frame: u64) {
        let p = &mut self.pages[page as usize];
        if let Some(owner) = p.owner {
            p.rendered = true;
            p.rendered_frame = frame;
            self.table[owner as usize] = u32::from(page) + 1;
            self.dirty[(owner / PAGES_PER_FACE) as usize] = true;
        }
    }

    /// Faces whose page table entries changed since the last call.
    pub fn take_dirty_faces(&mut self) -> Vec<u32> {
        let faces = (0..self.dirty.len() as u32).filter(|&f| self.dirty[f as usize]).collect();
        self.dirty.fill(false);
        faces
    }

    /// (pinned, other) resident pages, for stats.
    pub fn resident_counts(&self) -> (usize, usize) {
        let pinned = self.pages.iter().filter(|p| p.owner.is_some() && p.pinned).count();
        let all = self.pages.iter().filter(|p| p.owner.is_some()).count();
        (pinned, all - pinned)
    }

    fn allocate(&mut self, index: u32, frame: u64, pinned: bool) -> Option<u16> {
        let page = match self.free.pop() {
            Some(page) => page,
            None => {
                // Least recently requested unpinned page not asked for this round.
                let (page, _) = self
                    .pages
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| !p.pinned && (pinned || p.last_requested < frame))
                    .min_by_key(|(_, p)| p.last_requested)?;
                self.evict(page as u16);
                self.free.pop()?
            }
        };
        self.assign(page, index, frame, pinned);
        Some(page)
    }

    fn assign(&mut self, page: u16, index: u32, frame: u64, pinned: bool) {
        self.pages[page as usize] = Physical { owner: Some(index), last_requested: frame, rendered: false, rendered_frame: 0, pinned };
        self.resident[index as usize] = page;
    }

    fn evict(&mut self, page: u16) {
        let p = &mut self.pages[page as usize];
        if let Some(owner) = p.owner.take() {
            self.resident[owner as usize] = NOT_RESIDENT;
            if self.table[owner as usize] != 0 {
                self.table[owner as usize] = 0;
                self.dirty[(owner / PAGES_PER_FACE) as usize] = true;
            }
        }
        *p = Physical { owner: None, last_requested: 0, rendered: false, rendered_frame: 0, pinned: false };
        self.free.push(page);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(indices: &[u32]) -> Vec<u32> {
        let mut mask = vec![0; REQUEST_WORDS];
        for &i in indices {
            mask[i as usize / 32] |= 1 << (i % 32);
        }
        mask
    }

    fn render_all(v: &mut VirtualShadows) {
        for (_, page) in v.pending(usize::MAX) {
            v.rendered(page, 0);
        }
    }

    #[test]
    fn layout_constants() {
        assert_eq!(MIPS, 7);
        assert_eq!(PAGES_PER_FACE, 4096 + 1024 + 256 + 64 + 16 + 4 + 1);
        assert_eq!(POOL_PAGES, 1024);
    }

    #[test]
    fn page_keys_round_trip() {
        for key in [
            PageKey { face: 0, mip: 0, x: 0, y: 0 },
            PageKey { face: 7, mip: 0, x: 63, y: 12 },
            PageKey { face: 191, mip: 3, x: 5, y: 7 },
            PageKey::whole(42),
        ] {
            assert_eq!(PageKey::from_index(key.index()), key);
            assert_eq!(mip_of(key.index()), key.mip);
        }
        assert_eq!(PageKey::whole(0).index(), PAGES_PER_FACE - 1);
    }

    #[test]
    fn a_page_maps_its_part_of_the_face_to_the_whole_clip_square() {
        let face = Mat4::identity();
        let key = PageKey { face: 0, mip: 5, x: 1, y: 0 };
        // mip 5 has 2×2 pages: page (1, 0) is the top right quarter of NDC.
        let narrow = key.view_proj(face);
        let corner = narrow * Vec4::new(1.0, 1.0, 0.3, 1.0);
        let centre = narrow * Vec4::new(0.5, 0.5, 0.3, 1.0);
        assert!((corner.x - 1.0).abs() < 1e-5 && (corner.y - 1.0).abs() < 1e-5);
        assert!(centre.x.abs() < 1e-5 && centre.y.abs() < 1e-5 && (centre.z - 0.3).abs() < 1e-6);
    }

    #[test]
    fn one_sided_lights_skip_the_face_behind() {
        let behind: Vec<bool> = (0..6).map(|f| face_behind(f, Vec3::unit_x())).collect();
        assert_eq!(behind, [false, true, false, false, false, false]);
    }

    #[test]
    fn requests_become_rendered_table_entries() {
        let mut v = VirtualShadows::default();
        v.reset_slot(0, [true; 6], 1);
        assert_eq!(v.pending(0).len(), 6, "every face's coarsest page renders regardless of budget");
        render_all(&mut v);
        let fine = PageKey { face: 2, mip: 0, x: 10, y: 20 }.index();
        v.request(&mask(&[fine]), 2);
        assert_eq!(v.table()[fine as usize], 0, "not rendered yet");
        render_all(&mut v);
        assert_ne!(v.table()[fine as usize], 0);
        assert!(v.take_dirty_faces().contains(&2));
    }

    #[test]
    fn disabled_faces_ignore_requests() {
        let mut v = VirtualShadows::default();
        v.reset_slot(0, [true, false, true, true, true, true], 1);
        v.request(&mask(&[PageKey { face: 1, mip: 0, x: 0, y: 0 }.index()]), 2);
        assert_eq!(v.resident_counts(), (5, 0));
    }

    #[test]
    fn a_full_pool_evicts_the_least_recently_requested() {
        let mut v = VirtualShadows::default();
        v.reset_slot(0, [true; 6], 1);
        let fill: Vec<u32> = (0..POOL_PAGES - 6).map(|i| PageKey { face: 0, mip: 0, x: i % 64, y: i / 64 }.index()).collect();
        v.request(&mask(&fill), 2);
        assert_eq!(v.resident_counts(), (6, (POOL_PAGES - 6) as usize));
        // Frame 3 asks for everything but the first page again, plus a new one: the first goes.
        let mut again = fill[1..].to_vec();
        let new = PageKey { face: 3, mip: 0, x: 1, y: 1 }.index();
        again.push(new);
        v.request(&mask(&again), 3);
        assert_eq!(v.resident[fill[0] as usize], NOT_RESIDENT);
        assert_ne!(v.resident[new as usize], NOT_RESIDENT);
        // Everything is wanted this frame: a further request finds nothing to evict.
        let another = PageKey { face: 4, mip: 0, x: 2, y: 2 }.index();
        again.push(another);
        v.request(&mask(&again), 3);
        assert_eq!(v.resident[another as usize], NOT_RESIDENT);
        // Pinned pages survive it all.
        assert_eq!(v.resident_counts().0, 6);
    }

    #[test]
    fn invalidating_re_renders_only_the_pages_under_the_cells() {
        let mut v = VirtualShadows::default();
        v.reset_slot(0, [true; 6], 1);
        let under = PageKey { face: 1, mip: 0, x: 9, y: 13 }.index();
        let beside = PageKey { face: 1, mip: 0, x: 20, y: 13 }.index();
        let other_face = PageKey { face: 2, mip: 0, x: 9, y: 13 }.index();
        v.request(&mask(&[under, beside, other_face]), 2);
        render_all(&mut v);
        let mut cells = [false; (DIRTY_GRID * DIRTY_GRID) as usize];
        cells[(3 * DIRTY_GRID + 2) as usize] = true; // mip 0 pages 8–11 × 12–15
        v.invalidate(&HashMap::from([(1, cells)]));
        let pending: Vec<u32> = v.pending(usize::MAX).iter().map(|(k, _)| k.index()).collect();
        assert!(pending.contains(&under) && pending.contains(&PageKey::whole(1).index()));
        assert!(!pending.contains(&beside) && !pending.contains(&other_face));
        assert_ne!(v.table()[under as usize], 0, "keeps its old map until it renders");
    }

    #[test]
    fn unmapped_pages_go_before_stale_ones_which_take_turns() {
        let mut v = VirtualShadows::default();
        v.reset_slot(0, [true; 6], 1);
        let old = PageKey { face: 0, mip: 0, x: 0, y: 0 }.index();
        let newer = PageKey { face: 0, mip: 0, x: 63, y: 63 }.index();
        v.request(&mask(&[old]), 2);
        for (_, page) in v.pending(usize::MAX) {
            v.rendered(page, 2);
        }
        v.request(&mask(&[old, newer]), 3);
        for (_, page) in v.pending(usize::MAX) {
            v.rendered(page, 3);
        }
        let mut all = [false; (DIRTY_GRID * DIRTY_GRID) as usize];
        all.fill(true);
        v.invalidate(&HashMap::from([(0, all)]));
        let fresh = PageKey { face: 0, mip: 3, x: 1, y: 1 }.index();
        v.request(&mask(&[old, newer, fresh]), 4);
        let order: Vec<u32> = v.pending(3).iter().map(|(k, _)| k.index()).collect();
        assert_eq!(order, [PageKey::whole(0).index(), fresh, old, newer], "face 0's pinned page, then the rest");
    }

    #[test]
    fn resetting_a_slot_drops_its_pages() {
        let mut v = VirtualShadows::default();
        v.reset_slot(3, [true; 6], 1);
        render_all(&mut v);
        let fine = PageKey { face: 3 * 6, mip: 1, x: 3, y: 3 }.index();
        v.request(&mask(&[fine]), 2);
        render_all(&mut v);
        v.reset_slot(3, [true; 6], 3);
        assert_eq!(v.table()[fine as usize], 0);
        assert_eq!(v.table()[PageKey::whole(18).index() as usize], 0, "the coarsest page re-renders");
        assert_eq!(v.resident_counts(), (6, 0));
    }
}

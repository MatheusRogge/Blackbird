//! Virtual shadow maps for local lights (a scaled-down take on Unreal 5's): every face of a
//! shadowed light is a `VIRTUAL_SIZE`² map with a mip chain, split into `PAGE_SIZE`² pages, and
//! only the pages visible pixels ask for are backed by one of the `POOL_PAGES` physical pages of
//! the atlas. The coarsest mip of each face (one page: the whole face) is always resident, so a
//! lookup can always fall back to it. This module is the CPU side: page indexing, the physical
//! pool (least recently requested pages are evicted first) and the page table the shader reads.

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
    /// Holds its owner's map; until then the page table doesn't point at it.
    rendered: bool,
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
            pages: vec![Physical { owner: None, last_requested: 0, rendered: false, pinned: false }; POOL_PAGES as usize],
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

    /// Applies a page request mask (a bit per table entry): requested resident pages are marked
    /// used at `frame`, and missing ones of enabled faces are allocated, coarsest mip first so a
    /// full pool still serves the fallbacks, evicting the least recently requested page when the
    /// pool is full. Requests the pool can't serve stay unbacked; lookups fall back to coarser
    /// pages.
    pub fn request(&mut self, mask: &[u32], frame: u64) {
        let mut missing = Vec::new();
        for (word, &bits) in mask.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let index = (word * 32) as u32 + bits.trailing_zeros();
                bits &= bits - 1;
                if index as usize >= TABLE_LEN || !self.face_enabled(index / PAGES_PER_FACE) {
                    continue;
                }
                match self.resident[index as usize] {
                    NOT_RESIDENT => missing.push(index),
                    page => self.pages[page as usize].last_requested = frame,
                }
            }
        }
        missing.sort_by_key(|&i| std::cmp::Reverse(PageKey::from_index(i).mip));
        for index in missing {
            if self.allocate(index, frame, false).is_none() {
                break;
            }
        }
    }

    /// Resident pages still to render: pinned ones (all of them: a face without its coarsest
    /// page has no shadow at all), then up to `budget` others, coarsest mip first.
    pub fn pending(&self, budget: usize) -> Vec<(PageKey, u16)> {
        let mut pinned = Vec::new();
        let mut others = Vec::new();
        for (page, p) in self.pages.iter().enumerate() {
            let Some(owner) = p.owner.filter(|_| !p.rendered) else { continue };
            let entry = (PageKey::from_index(owner), page as u16);
            if p.pinned { pinned.push(entry) } else { others.push(entry) }
        }
        others.sort_by_key(|(key, _)| (std::cmp::Reverse(key.mip), key.index()));
        others.truncate(budget);
        pinned.extend(others);
        pinned
    }

    /// Marks `page` rendered: the page table now points its owner at it.
    pub fn rendered(&mut self, page: u16) {
        let p = &mut self.pages[page as usize];
        if let Some(owner) = p.owner {
            p.rendered = true;
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
        self.pages[page as usize] = Physical { owner: Some(index), last_requested: frame, rendered: false, pinned };
        self.resident[index as usize] = page;
        Some(page)
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
        *p = Physical { owner: None, last_requested: 0, rendered: false, pinned: false };
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
            v.rendered(page);
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

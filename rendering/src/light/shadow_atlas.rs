//! Tiles of the local-light shadow atlas: square, power-of-two sized, allocated from a quadtree
//! (a block splits into four children; four free siblings merge back into their parent).

/// Side of the atlas in texels. The depth atlas and the glass tint atlas share this layout.
pub const ATLAS_SIZE: u32 = 4096;
/// Largest tile a light face gets.
pub const MAX_TILE: u32 = 512;
/// Smallest tile a light face gets.
pub const MIN_TILE: u32 = 128;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub size: u32,
}

impl Tile {
    /// Offset and scale in atlas UV, then the size in texels: `LocalShadowFace::rect`.
    pub fn rect(self) -> [f32; 4] {
        let atlas = ATLAS_SIZE as f32;
        [self.x as f32 / atlas, self.y as f32 / atlas, self.size as f32 / atlas, self.size as f32]
    }
}

pub struct ShadowAtlas {
    /// Free block corners per level; level `l` blocks are `ATLAS_SIZE >> l` wide.
    free: Vec<Vec<(u32, u32)>>,
}

impl Default for ShadowAtlas {
    fn default() -> Self {
        let levels = level(MIN_TILE) + 1;
        let mut free = vec![Vec::new(); levels];
        free[0].push((0, 0));
        Self { free }
    }
}

fn level(size: u32) -> usize {
    debug_assert!(size.is_power_of_two() && (MIN_TILE..=ATLAS_SIZE).contains(&size), "bad tile size {size}");
    (ATLAS_SIZE / size).ilog2() as usize
}

impl ShadowAtlas {
    /// A free tile of `size` (a power of two in `MIN_TILE..=ATLAS_SIZE`), splitting the smallest
    /// free block that fits. `None` when the atlas has no room.
    pub fn alloc(&mut self, size: u32) -> Option<Tile> {
        let target = level(size);
        let from = (0..=target).rev().find(|&l| !self.free[l].is_empty())?;
        let (x, y) = self.free[from].pop()?;
        for l in from..target {
            let half = ATLAS_SIZE >> (l + 1);
            self.free[l + 1].extend([(x + half, y), (x, y + half), (x + half, y + half)]);
        }
        Some(Tile { x, y, size })
    }

    pub fn free(&mut self, tile: Tile) {
        let (mut x, mut y, mut level) = (tile.x, tile.y, level(tile.size));
        while level > 0 {
            let size = ATLAS_SIZE >> level;
            let (px, py) = (x / (2 * size) * (2 * size), y / (2 * size) * (2 * size));
            let siblings = [(px, py), (px + size, py), (px, py + size), (px + size, py + size)];
            let free = &mut self.free[level];
            if !siblings.iter().filter(|&&s| s != (x, y)).all(|s| free.contains(s)) {
                break;
            }
            free.retain(|s| !siblings.contains(s));
            (x, y, level) = (px, py, level - 1);
        }
        self.free[level].push((x, y));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_and_merges_back() {
        let mut atlas = ShadowAtlas::default();
        let count = (ATLAS_SIZE / MAX_TILE).pow(2);
        let tiles: Vec<_> = (0..count).map(|_| atlas.alloc(MAX_TILE).expect("room")).collect();
        assert!(atlas.alloc(MIN_TILE).is_none());
        for tile in tiles {
            atlas.free(tile);
        }
        assert_eq!(atlas.free[0], vec![(0, 0)]);
        assert!(atlas.free[1..].iter().all(Vec::is_empty));
    }

    #[test]
    fn mixed_sizes_never_overlap() {
        let mut atlas = ShadowAtlas::default();
        let mut tiles = Vec::new();
        for (i, size) in [512, 128, 256, 128, 512, 256, 128].into_iter().cycle().take(200).enumerate() {
            if let Some(tile) = atlas.alloc(size) {
                tiles.push(tile);
            }
            if i % 3 == 0 && !tiles.is_empty() {
                atlas.free(tiles.swap_remove(i % tiles.len()));
            }
        }
        for (i, a) in tiles.iter().enumerate() {
            assert!(a.x + a.size <= ATLAS_SIZE && a.y + a.size <= ATLAS_SIZE);
            for b in &tiles[i + 1..] {
                let apart = a.x + a.size <= b.x || b.x + b.size <= a.x || a.y + a.size <= b.y || b.y + b.size <= a.y;
                assert!(apart, "{a:?} overlaps {b:?}");
            }
        }
    }
}

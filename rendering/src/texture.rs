use std::sync::atomic::{AtomicU64, Ordering};

use assets::Asset;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct TextureAsset {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGBA8, 4 bytes per pixel
    id: u64,
}

impl TextureAsset {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Unique for the life of the process and never 0, so GPU caches can key by it. An `Arc`
    /// address is not enough: after a hot reload the new scene's textures can reuse the old
    /// ones' addresses.
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl TextureAsset {
    /// Average texel as linear RGBA, with RGB weighted by alpha so cut-out texels don't count.
    /// `srgb` decodes RGB first. Large textures are subsampled.
    pub fn mean(&self, srgb: bool) -> [f32; 4] {
        let texels = self.data.as_chunks::<4>().0;
        let step = (texels.len() / 65_536).max(1);
        let (mut rgb, mut alpha, mut count) = ([0.0f64; 3], 0.0f64, 0usize);
        for texel in texels.iter().step_by(step) {
            let a = texel[3] as f64 / 255.0;
            for (sum, &c) in rgb.iter_mut().zip(texel) {
                let v = c as f32 / 255.0;
                *sum += a * if srgb { srgb_to_linear(v) } else { v } as f64;
            }
            alpha += a;
            count += 1;
        }
        if alpha <= 0.0 {
            return [1.0; 4];
        }
        let [r, g, b] = rgb.map(|c| (c / alpha) as f32);
        [r, g, b, (alpha / count as f64) as f32]
    }

    /// Fraction of the triangle with texture coordinates `uv` whose alpha is at least `cutoff`
    /// (0–1), sampled on a grid of `COVERAGE_GRID`² / 2 points, nearest texel, repeating.
    pub fn coverage(&self, uv: [[f32; 2]; 3], cutoff: f32) -> f32 {
        const COVERAGE_GRID: u32 = 12;
        if self.width == 0 || self.height == 0 {
            return 1.0;
        }
        if cutoff > 1.0 {
            return 0.0;
        }
        let threshold = (cutoff * 255.0).ceil() as u8;
        let (mut inside, mut total) = (0u32, 0u32);
        for i in 0..COVERAGE_GRID {
            for j in 0..COVERAGE_GRID - i {
                let a = (i as f32 + 1.0 / 3.0) / COVERAGE_GRID as f32;
                let b = (j as f32 + 1.0 / 3.0) / COVERAGE_GRID as f32;
                let c = 1.0 - a - b;
                let [u, v] = std::array::from_fn(|k| uv[0][k] * a + uv[1][k] * b + uv[2][k] * c);
                let x = ((u.rem_euclid(1.0) * self.width as f32) as u32).min(self.width - 1);
                let y = ((v.rem_euclid(1.0) * self.height as f32) as u32).min(self.height - 1);
                inside += u32::from(self.data[((y * self.width + x) * 4 + 3) as usize] >= threshold);
                total += 1;
            }
        }
        inside as f32 / total as f32
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

impl Asset for TextureAsset {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_ignores_transparent_texels() {
        let texture = TextureAsset::new(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 0]);
        assert_eq!(texture.mean(false), [1.0, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn coverage_counts_texels_passing_the_cutoff() {
        // Left column opaque, right column transparent.
        let texture = TextureAsset::new(2, 1, vec![0, 0, 0, 255, 0, 0, 0, 0]);
        let left = [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]];
        assert_eq!(texture.coverage(left, 0.5), 1.0);
        let whole = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let c = texture.coverage(whole, 0.5);
        assert!((c - 0.75).abs() < 0.1, "{c}");
        assert_eq!(texture.coverage(whole, 0.0), 1.0);
    }

    #[test]
    fn mean_decodes_srgb() {
        let texture = TextureAsset::new(1, 1, vec![188, 188, 188, 255]);
        let [r, ..] = texture.mean(true);
        assert!((r - 0.5).abs() < 0.01, "{r}");
    }
}

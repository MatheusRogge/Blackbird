use assets::Asset;

#[derive(Debug)]
pub struct TextureAsset {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGBA8, 4 bytes per pixel
}

impl TextureAsset {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
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
    fn mean_decodes_srgb() {
        let texture = TextureAsset::new(1, 1, vec![188, 188, 188, 255]);
        let [r, ..] = texture.mean(true);
        assert!((r - 0.5).abs() < 0.01, "{r}");
    }
}

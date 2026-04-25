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

impl Asset for TextureAsset {}

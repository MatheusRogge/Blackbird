//! `.bbpak`: Blackbird's packed asset container.
//!
//! ```text
//! header   48 bytes, little endian
//!          magic "BBPAK\0\0\0" | version u32 | entry count u32 | toc offset u64
//!          toc stored len u64 | toc raw len u64 | content id u64
//! blobs    one per entry, 16-byte aligned, zstd or stored
//! toc      zstd; per entry, sorted by path:
//!          path len u16 | path utf-8 | codec u8 | offset u64 | stored len u64 | raw len u64 | hash u64
//! ```
//!
//! Entries are decompressed individually on demand. The content id is a hash of the raw TOC,
//! so it changes whenever any entry does; builds embed it to refuse a mismatched pack.

use std::{
    fs::File,
    io::{self, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use crate::{
    manifest::{ManifestEntry, content_hash},
    path::to_logical,
    source::AssetSource,
};

const MAGIC: [u8; 8] = *b"BBPAK\0\0\0";
const VERSION: u32 = 1;
const HEADER_LEN: u64 = 48;
const ALIGN: u64 = 16;
/// Compressed data is only kept when it is at most this fraction of the original. PNG and JPEG
/// rarely shrink further, and storing them avoids a pointless decompression on load.
const MIN_RATIO: f64 = 0.95;
const MAX_ENTRY_LEN: u64 = 1 << 34;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    Stored = 0,
    Zstd = 1,
}

impl Codec {
    fn from_u8(v: u8) -> io::Result<Self> {
        match v {
            0 => Ok(Self::Stored),
            1 => Ok(Self::Zstd),
            _ => Err(invalid(format!("unknown codec {v}"))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackEntry {
    pub path: String,
    pub codec: Codec,
    pub offset: u64,
    pub stored_len: u64,
    pub raw_len: u64,
    pub hash: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PackSummary {
    pub content_id: u64,
    pub entries: usize,
    pub raw_bytes: u64,
    pub stored_bytes: u64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Streams entries into a new pack file; nothing is valid until `finish`.
pub struct PackWriter {
    out: BufWriter<File>,
    offset: u64,
    level: i32,
    entries: Vec<PackEntry>,
}

impl PackWriter {
    pub fn create(path: &Path, level: i32) -> io::Result<Self> {
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(&[0; HEADER_LEN as usize])?;
        Ok(Self { out, offset: HEADER_LEN, level, entries: Vec::new() })
    }

    pub fn add(&mut self, path: &str, bytes: &[u8]) -> io::Result<()> {
        if path.len() > usize::from(u16::MAX) {
            return Err(invalid(format!("path too long: {path}")));
        }
        if self.entries.iter().any(|e| e.path == path) {
            return Err(invalid(format!("duplicate entry: {path}")));
        }

        let compressed = zstd::bulk::compress(bytes, self.level)?;
        let (codec, stored) = if (compressed.len() as f64) <= bytes.len() as f64 * MIN_RATIO {
            (Codec::Zstd, compressed.as_slice())
        } else {
            (Codec::Stored, bytes)
        };

        let padding = self.offset.next_multiple_of(ALIGN) - self.offset;
        self.out.write_all(&[0; ALIGN as usize][..padding as usize])?;
        self.offset += padding;

        self.out.write_all(stored)?;
        self.entries.push(PackEntry {
            path: path.to_string(),
            codec,
            offset: self.offset,
            stored_len: stored.len() as u64,
            raw_len: bytes.len() as u64,
            hash: content_hash(bytes),
        });
        self.offset += stored.len() as u64;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<PackSummary> {
        self.entries.sort_by(|a, b| a.path.cmp(&b.path));

        let mut toc = Vec::new();
        for e in &self.entries {
            toc.extend_from_slice(&(e.path.len() as u16).to_le_bytes());
            toc.extend_from_slice(e.path.as_bytes());
            toc.push(e.codec as u8);
            for v in [e.offset, e.stored_len, e.raw_len, e.hash] {
                toc.extend_from_slice(&v.to_le_bytes());
            }
        }
        let content_id = content_hash(&toc);
        let toc_stored = zstd::bulk::compress(&toc, self.level)?;
        self.out.write_all(&toc_stored)?;

        let mut header = Vec::with_capacity(HEADER_LEN as usize);
        header.extend_from_slice(&MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for v in [self.offset, toc_stored.len() as u64, toc.len() as u64, content_id] {
            header.extend_from_slice(&v.to_le_bytes());
        }
        let mut file = self.out.into_inner().map_err(io::IntoInnerError::into_error)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.sync_all()?;

        Ok(PackSummary {
            content_id,
            entries: self.entries.len(),
            raw_bytes: self.entries.iter().map(|e| e.raw_len).sum(),
            stored_bytes: self.entries.iter().map(|e| e.stored_len).sum(),
        })
    }
}

/// Read access to a `.bbpak`. Only the header and TOC are read up front.
pub struct PackReader {
    path: PathBuf,
    file: Mutex<File>,
    content_id: u64,
    entries: Vec<PackEntry>,
}

impl PackReader {
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();

        let mut header = [0u8; HEADER_LEN as usize];
        file.read_exact(&mut header)?;
        if header[..8] != MAGIC {
            return Err(invalid(format!("{} is not a .bbpak file", path.display())));
        }
        let u32_at = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().expect("4 bytes"));
        let u64_at = |i: usize| u64::from_le_bytes(header[i..i + 8].try_into().expect("8 bytes"));
        let version = u32_at(8);
        if version != VERSION {
            return Err(invalid(format!("unsupported pack version {version}")));
        }
        let count = u32_at(12) as usize;
        let (toc_offset, toc_stored_len, toc_raw_len, content_id) = (u64_at(16), u64_at(24), u64_at(32), u64_at(40));
        if toc_offset.checked_add(toc_stored_len) != Some(file_len) || toc_raw_len > MAX_ENTRY_LEN {
            return Err(invalid("truncated or corrupt pack"));
        }

        file.seek(SeekFrom::Start(toc_offset))?;
        let mut toc_stored = vec![0; toc_stored_len as usize];
        file.read_exact(&mut toc_stored)?;
        let toc = zstd::bulk::decompress(&toc_stored, toc_raw_len as usize)?;
        if content_hash(&toc) != content_id {
            return Err(invalid("pack table of contents is corrupt"));
        }

        let entries = parse_toc(&toc, count, toc_offset)?;
        Ok(Self { path, file: Mutex::new(file), content_id, entries })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn content_id(&self) -> u64 {
        self.content_id
    }

    /// Sorted by path.
    pub fn entries(&self) -> &[PackEntry] {
        &self.entries
    }

    pub fn entry(&self, path: &str) -> Option<&PackEntry> {
        self.entries
            .binary_search_by(|e| e.path.as_str().cmp(path))
            .ok()
            .map(|i| &self.entries[i])
    }

    pub fn manifest_entries(&self) -> impl Iterator<Item = ManifestEntry> + '_ {
        self.entries.iter().map(|e| ManifestEntry { path: e.path.clone(), size: e.raw_len, hash: e.hash })
    }

    pub fn read(&self, path: &str) -> io::Result<Vec<u8>> {
        let entry = self.entry(path).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("{path} is not in {}", self.path.display()))
        })?;

        let mut stored = vec![0; entry.stored_len as usize];
        {
            let mut file = self.file.lock().expect("pack file mutex poisoned");
            file.seek(SeekFrom::Start(entry.offset))?;
            file.read_exact(&mut stored)?;
        }

        let raw = match entry.codec {
            Codec::Stored => stored,
            Codec::Zstd => zstd::bulk::decompress(&stored, entry.raw_len as usize)?,
        };
        if raw.len() as u64 != entry.raw_len || content_hash(&raw) != entry.hash {
            return Err(invalid(format!("{path} is corrupt in {}", self.path.display())));
        }
        Ok(raw)
    }
}

fn parse_toc(toc: &[u8], count: usize, data_end: u64) -> io::Result<Vec<PackEntry>> {
    let mut cursor = toc;
    let mut take = |n: usize| -> io::Result<&[u8]> {
        if cursor.len() < n {
            return Err(invalid("truncated table of contents"));
        }
        let (head, rest) = cursor.split_at(n);
        cursor = rest;
        Ok(head)
    };

    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let path_len = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
        let path = std::str::from_utf8(take(path_len)?)
            .map_err(|_| invalid("non UTF-8 path in pack"))?
            .to_string();
        let codec = Codec::from_u8(take(1)?[0])?;
        let mut fields = [0u64; 4];
        for f in &mut fields {
            *f = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
        }
        let [offset, stored_len, raw_len, hash] = fields;
        if offset.checked_add(stored_len).is_none_or(|end| end > data_end) || raw_len > MAX_ENTRY_LEN {
            return Err(invalid(format!("entry {path} lies outside the pack")));
        }
        entries.push(PackEntry { path, codec, offset, stored_len, raw_len, hash });
    }
    if !entries.is_sorted_by(|a, b| a.path < b.path) {
        return Err(invalid("pack entries are not sorted"));
    }
    Ok(entries)
}

/// Serves logical paths out of a pack. Absolute paths never resolve, so a production build
/// can't reach outside its packaged content.
pub struct PackSource {
    reader: PackReader,
}

impl PackSource {
    pub fn new(reader: PackReader) -> Self {
        Self { reader }
    }

    pub fn reader(&self) -> &PackReader {
        &self.reader
    }
}

impl AssetSource for PackSource {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let logical = to_logical(path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{} is outside the pack; packed builds only load project-relative paths", path.display()),
            )
        })?;
        self.reader.read(&logical)
    }

    fn describe(&self) -> String {
        format!("pack {} ({} entries)", self.reader.path.display(), self.reader.entries.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_pack(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("blackbird-{name}-{}.bbpak", std::process::id()))
    }

    #[test]
    fn round_trip() {
        let path = temp_pack("round-trip");
        let text = b"hello hello hello hello hello hello hello hello".repeat(20);
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let noise: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();

        let mut writer = PackWriter::create(&path, 3).unwrap();
        writer.add("scenes/main.gltf", &text).unwrap();
        writer.add("assets/noise.png", &noise).unwrap();
        writer.add("empty.txt", &[]).unwrap();
        let summary = writer.finish().unwrap();

        let reader = PackReader::open(&path).unwrap();
        assert_eq!(reader.content_id(), summary.content_id);
        assert_eq!(reader.entry("scenes/main.gltf").unwrap().codec, Codec::Zstd);
        assert_eq!(reader.entry("assets/noise.png").unwrap().codec, Codec::Stored);

        let source = PackSource::new(reader);
        assert_eq!(source.read(Path::new("scenes/../scenes/main.gltf")).unwrap(), text);
        assert_eq!(source.read(Path::new("assets/noise.png")).unwrap(), noise);
        assert!(source.read(Path::new("empty.txt")).unwrap().is_empty());
        assert!(source.read(Path::new("missing.bin")).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn detects_corruption() {
        let path = temp_pack("corrupt");
        let mut writer = PackWriter::create(&path, 3).unwrap();
        writer.add("a.txt", &b"abcdefgh".repeat(100)).unwrap();
        writer.finish().unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_LEN as usize + 4] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();

        let reader = PackReader::open(&path).unwrap();
        assert!(reader.read("a.txt").is_err());
        std::fs::remove_file(path).unwrap();
    }
}

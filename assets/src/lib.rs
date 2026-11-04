pub mod codegen;
pub mod manifest;
#[cfg(feature = "pack")]
pub mod pack;
pub mod path;
pub mod source;

use std::{
    any::Any,
    collections::{HashMap, HashSet, VecDeque},
    ffi::OsStr,
    io::{self},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use downcast_rs::{Downcast, impl_downcast};
use thiserror::Error;
use threading::Executor;
use tracing::{info, warn};

pub use manifest::{AssetManifest, ManifestEntry, StaticAsset};
pub use source::{AssetSource, DirectorySource};

#[derive(Error, Debug)]
pub enum AssetError {
    #[error(transparent)]
    IO(#[from] io::Error),

    #[error("{path}: {source}")]
    Read { path: PathBuf, source: io::Error },

    #[error("Inner error: {0}")]
    Other(String),
}

pub trait Asset: Any + Downcast + Send + Sync + 'static {}
impl_downcast!(Asset);

#[derive(Debug)]
pub struct LoadedAsset<A: Asset> {
    pub(crate) inner: A,
}

pub struct ErasedLoadedAsset {
    inner: Box<dyn Asset>,
}

impl<A: Asset> From<LoadedAsset<A>> for ErasedLoadedAsset {
    fn from(value: LoadedAsset<A>) -> Self {
        Self {
            inner: Box::new(value.inner),
        }
    }
}

impl ErasedLoadedAsset {
    pub fn downcast<A: Asset>(mut self) -> Result<LoadedAsset<A>, ErasedLoadedAsset> {
        match self.inner.downcast() {
            Ok(value) => Ok(LoadedAsset { inner: *value }),
            Err(value) => {
                self.inner = value;
                Err(self)
            }
        }
    }
}

/// What a resolver sees: the asset's bytes plus access to files next to it, read through the
/// same `AssetSource`, so a resolver works unchanged against loose files and packs.
pub struct LoadContext<'a> {
    path: &'a Path,
    bytes: Vec<u8>,
    source: &'a dyn AssetSource,
}

impl<'a> LoadContext<'a> {
    pub fn path(&self) -> &Path {
        self.path
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn take_bytes(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }

    /// Reads `relative`, resolved against this asset's directory.
    pub fn read_relative(&self, relative: &str) -> Result<Vec<u8>, AssetError> {
        let path = path::resolve_relative(self.path, relative);
        read_source(self.source, &path)
    }
}

fn read_source(source: &dyn AssetSource, path: &Path) -> Result<Vec<u8>, AssetError> {
    source.read(path).map_err(|e| AssetError::Read { path: path.to_path_buf(), source: e })
}

pub trait ErasedAssetResolver: Send + Sync {
    fn resolve(&self, ctx: &mut LoadContext<'_>) -> Result<ErasedLoadedAsset, AssetError>;
    fn dependencies(&self, ctx: &LoadContext<'_>) -> Result<Vec<String>, AssetError>;
}

pub trait AssetResolver {
    type Asset: Asset;
    type Error: Into<AssetError>;

    fn resolve(&self, ctx: &mut LoadContext<'_>) -> Result<Self::Asset, Self::Error>;

    /// Files this asset loads through `LoadContext::read_relative`, relative to its directory.
    /// Builds follow these to decide what to ship, so a missing entry means a missing file in
    /// the packaged game.
    fn dependencies(&self, _ctx: &LoadContext<'_>) -> Result<Vec<String>, Self::Error> {
        Ok(Vec::new())
    }
}

impl<R> ErasedAssetResolver for R
where
    R: AssetResolver + Send + Sync,
{
    fn resolve(&self, ctx: &mut LoadContext<'_>) -> Result<ErasedLoadedAsset, AssetError> {
        <R as AssetResolver>::resolve(self, ctx)
            .map(|e| ErasedLoadedAsset { inner: Box::new(e) })
            .map_err(Into::into)
    }

    fn dependencies(&self, ctx: &LoadContext<'_>) -> Result<Vec<String>, AssetError> {
        <R as AssetResolver>::dependencies(self, ctx).map_err(Into::into)
    }
}

pub enum AssetLoadState<A: Asset> {
    Pending,
    Ready(Arc<A>),
    Failed(Arc<AssetError>),
}

pub struct AssetLoadHandle<A: Asset> {
    inner: Arc<Mutex<AssetLoadState<A>>>,
}

impl<A: Asset> AssetLoadHandle<A> {
    fn new() -> (Self, Arc<Mutex<AssetLoadState<A>>>) {
        let state = Arc::new(Mutex::new(AssetLoadState::Pending));
        (
            Self {
                inner: Arc::clone(&state),
            },
            state,
        )
    }

    fn failed(err: AssetError) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AssetLoadState::Failed(Arc::new(err)))),
        }
    }

    pub fn try_get(&self) -> Option<Result<Arc<A>, Arc<AssetError>>> {
        match &*self.inner.lock().expect("asset load state mutex poisoned") {
            AssetLoadState::Pending => None,
            AssetLoadState::Ready(a) => Some(Ok(Arc::clone(a))),
            AssetLoadState::Failed(e) => Some(Err(Arc::clone(e))),
        }
    }
}

pub struct AssetManager {
    resolvers: HashMap<String, Arc<dyn ErasedAssetResolver>>,
    executor: Option<Arc<dyn Executor>>,
    source: Arc<dyn AssetSource>,
    manifest: Option<Arc<AssetManifest>>,
}

impl Default for AssetManager {
    fn default() -> Self {
        Self {
            resolvers: HashMap::new(),
            executor: None,
            source: Arc::new(DirectorySource::beside_executable()),
            manifest: None,
        }
    }
}

fn extension_of(path: &Path) -> Result<String, AssetError> {
    path.extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| {
            AssetError::IO(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} has no file extension", path.display()),
            ))
        })
}

fn resolve_with(
    resolver: &dyn ErasedAssetResolver,
    source: &dyn AssetSource,
    path: &Path,
) -> Result<ErasedLoadedAsset, AssetError> {
    let bytes = read_source(source, path)?;
    resolver.resolve(&mut LoadContext { path, bytes, source })
}

fn downcast_loaded<A: Asset>(erased: ErasedLoadedAsset) -> Result<Arc<A>, AssetError> {
    erased
        .downcast::<A>()
        .map(|loaded| Arc::new(loaded.inner))
        .map_err(|_| AssetError::Other("Asset type mismatch on downcast".into()))
}

impl AssetManager {
    pub fn init_executor(&mut self, executor: Arc<dyn Executor>) {
        self.executor = Some(executor);
    }

    /// Replaces where assets are read from. Defaults to the executable's directory.
    pub fn set_source(&mut self, source: Arc<dyn AssetSource>) {
        self.source = source;
    }

    pub fn source(&self) -> &Arc<dyn AssetSource> {
        &self.source
    }

    /// Restricts the expected asset set: loading anything outside it logs a warning, because a
    /// packaged build of the same project would not contain it.
    pub fn set_manifest(&mut self, manifest: AssetManifest) {
        self.manifest = Some(Arc::new(manifest));
    }

    pub fn manifest(&self) -> Option<&AssetManifest> {
        self.manifest.as_deref()
    }

    pub fn add_resolver<R>(&mut self, extension: impl Into<String>, resolver: R)
    where
        R: AssetResolver + Send + Sync + 'static,
    {
        self.resolvers.insert(extension.into().to_ascii_lowercase(), Arc::new(resolver));
    }

    fn resolver_for(&self, path: &Path) -> Result<Arc<dyn ErasedAssetResolver>, AssetError> {
        let extension = extension_of(path)?;
        self.resolvers.get(&extension).cloned().ok_or_else(|| {
            AssetError::IO(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("No loader found for extension: {extension}"),
            ))
        })
    }

    fn check_manifest(&self, path: &Path) {
        let Some(manifest) = &self.manifest else { return };
        match path::to_logical(path) {
            Some(logical) if manifest.contains(&logical) => {}
            _ => warn!("{} is not in the asset manifest and won't ship in a packaged build", path.display()),
        }
    }

    pub fn load_asset<A>(&self, path: impl AsRef<Path>) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        let path = path::normalize(path.as_ref());
        info!("Loading asset {} from {}", path.display(), self.source.describe());
        self.check_manifest(&path);
        let resolver = self.resolver_for(&path)?;
        downcast_loaded(resolve_with(resolver.as_ref(), self.source.as_ref(), &path)?)
    }

    pub fn load_asset_from<A>(
        &self,
        path: impl AsRef<Path>,
        base_path: impl AsRef<Path>,
    ) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        self.load_asset(base_path.as_ref().join(path.as_ref()))
    }

    pub fn load_async<A: Asset>(&self, path: impl AsRef<Path>) -> AssetLoadHandle<A> {
        let path = path::normalize(path.as_ref());
        self.check_manifest(&path);

        let resolver = match self.resolver_for(&path) {
            Ok(r) => r,
            Err(e) => return AssetLoadHandle::failed(e),
        };

        let Some(exec) = &self.executor else {
            return AssetLoadHandle::failed(AssetError::Other(
                "AssetManager has no executor; AssetsPlugin must be registered before loading assets".into(),
            ));
        };

        let (handle, state) = AssetLoadHandle::<A>::new();
        let source = Arc::clone(&self.source);

        exec.spawn(Box::new(move || {
            info!("Loading asset {} from {}", path.display(), source.describe());
            let new_state = match resolve_with(resolver.as_ref(), source.as_ref(), &path).and_then(downcast_loaded) {
                Ok(asset) => AssetLoadState::Ready(asset),
                Err(e) => AssetLoadState::Failed(Arc::new(e)),
            };
            *state.lock().expect("asset load state mutex poisoned") = new_state;
        }));

        handle
    }

    /// Walks `roots` and everything their resolvers depend on, through the current source.
    /// Files without a resolver (buffers, images) are leaves. Every path must stay inside the
    /// source root, since that is all a pack can hold.
    pub fn collect(&self, roots: &[impl AsRef<Path>]) -> Result<AssetManifest, AssetError> {
        let mut manifest = AssetManifest::default();
        let mut seen = HashSet::new();
        let mut queue: VecDeque<(PathBuf, Option<String>)> =
            roots.iter().map(|r| (r.as_ref().to_path_buf(), None)).collect();

        while let Some((path, referrer)) = queue.pop_front() {
            let context = |message: String| match &referrer {
                Some(by) => AssetError::Other(format!("{message} (referenced by {by})")),
                None => AssetError::Other(message),
            };
            let logical = path::to_logical(&path)
                .ok_or_else(|| context(format!("{} is outside the project", path.display())))?;
            if !seen.insert(logical.clone()) {
                continue;
            }

            let bytes = read_source(self.source.as_ref(), Path::new(&logical))
                .map_err(|e| context(e.to_string()))?;
            manifest.insert(ManifestEntry {
                path: logical.clone(),
                size: bytes.len() as u64,
                hash: manifest::content_hash(&bytes),
            });

            let resolver = extension_of(Path::new(&logical)).ok().and_then(|ext| self.resolvers.get(&ext));
            if let Some(resolver) = resolver {
                let ctx = LoadContext { path: Path::new(&logical), bytes, source: self.source.as_ref() };
                for dep in resolver.dependencies(&ctx).map_err(|e| context(format!("{logical}: {e}")))? {
                    queue.push_back((path::resolve_relative(Path::new(&logical), &dep), Some(logical.clone())));
                }
            }
        }

        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MemorySource(HashMap<&'static str, &'static [u8]>);

    impl AssetSource for MemorySource {
        fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
            let logical = path::to_logical(path).unwrap_or_default();
            self.0.get(logical.as_str()).map(|b| b.to_vec()).ok_or_else(|| io::ErrorKind::NotFound.into())
        }

        fn describe(&self) -> String {
            "memory".into()
        }
    }

    struct Text(String);
    impl Asset for Text {}

    /// Each line of a `.list` file is a dependency.
    struct ListResolver;

    impl AssetResolver for ListResolver {
        type Asset = Text;
        type Error = AssetError;

        fn resolve(&self, ctx: &mut LoadContext<'_>) -> Result<Text, AssetError> {
            let mut text = String::from_utf8_lossy(ctx.bytes()).into_owned();
            for dep in AssetResolver::dependencies(self, ctx)? {
                text.push_str(&String::from_utf8_lossy(&ctx.read_relative(&dep)?));
            }
            Ok(Text(text))
        }

        fn dependencies(&self, ctx: &LoadContext<'_>) -> Result<Vec<String>, AssetError> {
            Ok(String::from_utf8_lossy(ctx.bytes()).lines().map(str::to_string).collect())
        }
    }

    fn manager(files: &[(&'static str, &'static [u8])]) -> AssetManager {
        let mut manager = AssetManager::default();
        manager.set_source(Arc::new(MemorySource(files.iter().copied().collect())));
        manager.add_resolver("LIST", ListResolver);
        manager
    }

    #[test]
    fn collects_transitive_dependencies_only() {
        let manager = manager(&[
            ("scenes/a.list", b"../data/b.list"),
            ("data/b.list", b"c.bin"),
            ("data/c.bin", b"!"),
            ("data/unused.bin", b"?"),
        ]);
        let manifest = manager.collect(&["scenes/a.list"]).unwrap();
        let paths: Vec<_> = manifest.entries().map(|e| e.path.as_str()).collect();
        assert_eq!(paths, ["data/b.list", "data/c.bin", "scenes/a.list"]);

        let text = manager.load_asset::<Text>("data/b.list").unwrap();
        assert_eq!(text.0, "c.bin!");
    }

    #[test]
    fn reports_missing_dependency_with_referrer() {
        let manager = manager(&[("a.list", b"missing.bin")]);
        let err = manager.collect(&["a.list"]).unwrap_err().to_string();
        assert!(err.contains("missing.bin") && err.contains("referenced by a.list"), "{err}");
    }

    #[test]
    fn rejects_dependencies_outside_the_root() {
        let manager = manager(&[("a.list", b"../escape.bin")]);
        assert!(manager.collect(&["a.list"]).is_err());
    }
}

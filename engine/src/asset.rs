use std::{
    any::Any,
    collections::HashMap,
    env,
    ffi::OsStr,
    fs::File,
    io::{self},
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};

use downcast_rs::{Downcast, impl_downcast};
use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum AssetError {
    #[error(transparent)]
    IO(#[from] io::Error),

    #[error("Inner error: {0}")]
    Other(String),
}

pub trait Asset: Any + Downcast + 'static {}
impl_downcast!(Asset);

#[derive(Debug)]
pub struct LoadedAsset<A: Asset> {
    inner: A,
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

pub struct AssetHandle {
    inner: Arc<ErasedLoadedAsset>,
}

impl Deref for AssetHandle {
    type Target = ErasedLoadedAsset;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

pub trait ErasedAssetResolver {
    fn resolve(&self, base_path: &Path, file: File) -> Result<ErasedLoadedAsset, AssetError>;
}

pub trait AssetResolver {
    type Asset: Asset;
    type Error: Into<AssetError>;

    fn resolve(&self, base_path: &Path, file: File) -> Result<Self::Asset, Self::Error>;
}

impl<R> ErasedAssetResolver for R
where
    R: AssetResolver,
{
    fn resolve(&self, base_path: &Path, file: File) -> Result<ErasedLoadedAsset, AssetError> {
        <R as AssetResolver>::resolve(self, base_path, file)
            .map(|e| ErasedLoadedAsset { inner: Box::new(e) })
            .map_err(Into::into)
    }
}

#[derive(Default)]
pub struct AssetManager {
    resolvers: HashMap<String, Box<dyn ErasedAssetResolver>>,
}

impl AssetManager {
    fn resolve_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            return path.to_path_buf();
        }

        env::current_exe().unwrap().join(path.as_os_str())
    }

    fn load_resolved<A>(&mut self, resolved: &Path) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        info!("Loading asset at path: {:?}", resolved);

        let extension = resolved
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Failed to file extension"))
            .map(String::from)?;

        info!("Detected extension: {:?}", extension);

        let Some(resolver) = self.resolvers.get(&extension) else {
            return Err(AssetError::IO(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("No loader found for extension: {}", extension),
            )));
        };

        let file = File::open(resolved)?;
        let base_path = resolved.parent().expect("File must have parent");

        let asset = resolver.resolve(base_path, file)?;
        let Ok(loaded_asset) = asset.downcast::<A>() else {
            return Err(AssetError::Other("Failed to cast".to_string()));
        };

        Ok(Arc::new(loaded_asset.inner))
    }

    pub fn load_asset<A>(&mut self, path: impl AsRef<Path>) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        let resolved = self.resolve_path(path.as_ref());
        self.load_resolved::<A>(&resolved)
    }

    pub fn load_asset_from<A>(
        &mut self,
        path: impl AsRef<Path>,
        base_path: impl AsRef<Path>,
    ) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        let resolved = base_path.as_ref().join(path.as_ref());
        self.load_resolved::<A>(&resolved)
    }

    pub fn add_resolver<R>(&mut self, extension: impl Into<String>, resolver: R)
    where
        R: AssetResolver + 'static,
    {
        let k = extension.into();
        let v = Box::new(resolver);

        self.resolvers.insert(k, v);
    }
}

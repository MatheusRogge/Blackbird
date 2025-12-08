use std::{
    any::Any,
    collections::HashMap,
    ffi::OsStr,
    fs::File,
    io::{self},
    path::Path,
    sync::Arc,
};

use downcast_rs::{Downcast, impl_downcast};
use log::debug;
use thiserror::Error;

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

pub trait ErasedAssetResolver {
    fn resolve(&self, file: File) -> Result<ErasedLoadedAsset, AssetError>;
}

pub trait AssetResolver: 'static {
    type Asset: Asset;
    type Error: Into<AssetError>;

    fn resolve(&self, file: File) -> Result<Self::Asset, Self::Error>;
}

impl<R> ErasedAssetResolver for R
where
    R: AssetResolver,
{
    fn resolve(&self, file: File) -> Result<ErasedLoadedAsset, AssetError> {
        <R as AssetResolver>::resolve(self, file)
            .map(|e| ErasedLoadedAsset { inner: Box::new(e) })
            .map_err(Into::into)
    }
}

#[derive(Default)]
pub struct AssetManager {
    _handles: HashMap<String, Arc<ErasedLoadedAsset>>,
    resolvers: HashMap<String, Box<dyn ErasedAssetResolver>>,
}

impl AssetManager {
    pub fn add_resolver<R>(&mut self, extension: impl Into<String>, resolver: R)
    where
        R: AssetResolver + 'static,
    {
        let k = extension.into();
        let v = Box::new(resolver);

        self.resolvers.insert(k, v);
    }

    pub fn load_asset<A>(&mut self, path: impl AsRef<Path>) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        debug!("Loading asset at path: {:?}", path.as_ref());

        let extension = path
            .as_ref()
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Failed to file extension"))
            .map(String::from)?;

        debug!("Detected extension: {:?}", extension);

        match self.resolvers.get(&extension) {
            None => Err(AssetError::IO(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("No loader found for extension: {}", extension),
            ))),
            Some(resolver) => {
                let file = File::open(path.as_ref())?;
                let asset = resolver.resolve(file)?;

                match asset.downcast::<A>() {
                    Ok(value) => Ok(Arc::new(value.inner)),
                    Err(_) => Err(AssetError::Other("Failed to cast".to_string())),
                }
            }
        }
    }
}

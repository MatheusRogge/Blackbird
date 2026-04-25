use std::{
    any::Any,
    collections::HashMap,
    env,
    ffi::OsStr,
    fs::File,
    io::{self},
    ops::Deref,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use downcast_rs::{Downcast, impl_downcast};
use thiserror::Error;
use threading::Executor;
use tracing::info;

#[derive(Error, Debug)]
pub enum AssetError {
    #[error(transparent)]
    IO(#[from] io::Error),

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

pub struct AssetHandle {
    inner: Arc<ErasedLoadedAsset>,
}

impl Deref for AssetHandle {
    type Target = ErasedLoadedAsset;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

pub trait ErasedAssetResolver: Send + Sync {
    fn resolve(&self, base_path: &Path, file: File) -> Result<ErasedLoadedAsset, AssetError>;
}

pub trait AssetResolver {
    type Asset: Asset;
    type Error: Into<AssetError>;

    fn resolve(&self, base_path: &Path, file: File) -> Result<Self::Asset, Self::Error>;
}

impl<R> ErasedAssetResolver for R
where
    R: AssetResolver + Send + Sync,
{
    fn resolve(&self, base_path: &Path, file: File) -> Result<ErasedLoadedAsset, AssetError> {
        <R as AssetResolver>::resolve(self, base_path, file)
            .map(|e| ErasedLoadedAsset { inner: Box::new(e) })
            .map_err(Into::into)
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
        match &*self.inner.lock().unwrap() {
            AssetLoadState::Pending => None,
            AssetLoadState::Ready(a) => Some(Ok(Arc::clone(a))),
            AssetLoadState::Failed(e) => Some(Err(Arc::clone(e))),
        }
    }
}

#[derive(Default)]
pub struct AssetManager {
    resolvers: HashMap<String, Arc<dyn ErasedAssetResolver>>,
    executor: Option<Arc<dyn Executor>>,
}

impl AssetManager {
    pub fn init_executor(&mut self, executor: Arc<dyn Executor>) {
        self.executor = Some(executor);
    }

    fn resolve_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            return path.to_path_buf();
        }
        env::current_exe().unwrap().join(path.as_os_str())
    }

    fn load_resolved<A>(&self, resolved: &Path) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        info!("Loading asset at path: {:?}", resolved);

        let extension = resolved
            .extension()
            .and_then(OsStr::to_str)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "Failed to get file extension")
            })
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

    pub fn load_asset<A>(&self, path: impl AsRef<Path>) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        let resolved = self.resolve_path(path.as_ref());
        self.load_resolved::<A>(&resolved)
    }

    pub fn load_asset_from<A>(
        &self,
        path: impl AsRef<Path>,
        base_path: impl AsRef<Path>,
    ) -> Result<Arc<A>, AssetError>
    where
        A: Asset,
    {
        let resolved = base_path.as_ref().join(path.as_ref());
        self.load_resolved::<A>(&resolved)
    }

    pub fn load_async<A: Asset>(&self, path: impl AsRef<Path>) -> AssetLoadHandle<A> {
        let resolved = self.resolve_path(path.as_ref());

        let ext = match resolved.extension().and_then(OsStr::to_str) {
            Some(e) => e.to_string(),
            None => {
                return AssetLoadHandle::failed(AssetError::IO(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Path has no extension",
                )));
            }
        };

        let Some(resolver) = self.resolvers.get(&ext).cloned() else {
            return AssetLoadHandle::failed(AssetError::IO(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("No loader found for extension: {}", ext),
            )));
        };

        let Some(exec) = &self.executor else {
            return AssetLoadHandle::failed(AssetError::Other(
                "AssetManager has no executor; AssetsPlugin must be registered before loading assets".into(),
            ));
        };

        let (handle, state) = AssetLoadHandle::<A>::new();

        exec.spawn(Box::new(move || {
            let parent = resolved.parent().unwrap_or(Path::new("")).to_path_buf();
            let result = File::open(&resolved)
                .map_err(AssetError::IO)
                .and_then(|f| resolver.resolve(&parent, f));

            let new_state = match result {
                Ok(erased) => match erased.downcast::<A>() {
                    Ok(loaded) => AssetLoadState::Ready(Arc::new(loaded.inner)),
                    Err(_) => AssetLoadState::Failed(Arc::new(AssetError::Other(
                        "Asset type mismatch on downcast".into(),
                    ))),
                },
                Err(e) => AssetLoadState::Failed(Arc::new(e)),
            };

            *state.lock().unwrap() = new_state;
        }));

        handle
    }

    pub fn add_resolver<R>(&mut self, extension: impl Into<String>, resolver: R)
    where
        R: AssetResolver + Send + Sync + 'static,
    {
        self.resolvers.insert(extension.into(), Arc::new(resolver));
    }
}

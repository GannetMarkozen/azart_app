pub extern crate serde;
pub extern crate serde_derive;
pub use serde_derive::{Serialize, Deserialize};

use bevy::{ecs::{component::Component, system::Resource}, prelude::Deref, reflect::Reflect, tasks::{IoTaskPool, Task}, utils::{hashbrown::hash_map::{Entry, EntryRef}, HashMap}};
use crossbeam::utils::CachePadded;
use either::Either;
use futures::{future::{join_all, Shared, WeakShared}, FutureExt};
use std::{any::{Any, TypeId}, cell::RefCell, fmt::Debug, io::Write, marker::PhantomData, path::{Path, PathBuf}, sync::{Arc, LazyLock, RwLock, Weak}, task::Poll};

static CACHE: LazyLock<AssetCache> = LazyLock::new(|| AssetCache {
  assets: CachePadded::new(RwLock::new(HashMap::new())),
  registry: CachePadded::new(RwLock::new(HashMap::new())),
});

thread_local! {
  static CONTEXT: RefCell<Context> = const { RefCell::new(Context::None) };
}

/// A typed asset.
#[derive(Deref, Reflect, Component, Resource)]
pub struct Asset<T: AssetTrait>(Arc<AssetInner<T>>);

impl<T: AssetTrait> Asset<T> {
  #[inline]
  const fn new(inner: Arc<AssetInner<T>>) -> Self {
    Self(inner)
  }

  #[inline]
  pub fn downgrade(this: &Self) -> WeakAsset<T> {
    WeakAsset(Arc::downgrade(&*this))
  }

  #[inline]
  pub fn into_any(self) -> AnyAsset {
    AnyAsset::new(self.into_inner() as _)
  }

  #[inline]
  fn into_inner(self) -> Arc<AssetInner<T>> {
    self.0
  }
}

impl<T:  AssetTrait> Clone for Asset<T> {
  fn clone(&self) -> Self {
    Self(Arc::clone(&self.0))
  }
}

impl<'de, T: AssetTrait + serde::de::DeserializeOwned> Asset<T> {
  pub fn load(path: impl AsRef<Path>) -> Load<T> {
    let path = path.as_ref();
    if let Some(a) = Self::get(path) {
      return Load::Loaded(a);
    }

    let mut assets = CACHE.assets.write().unwrap();

    match assets.entry_ref(path) {
      EntryRef::Occupied(mut e) => match e.get() {
        AssetState::Loaded(a) => match a.upgrade() {
          Some(a) => Load::Loaded(a.downcast::<T>().unwrap()),
          None => {
            let task = Self::spawn_load_task(Arc::clone(e.key())).shared();
            *e.get_mut() = AssetState::Loading(task.downgrade().expect("Task completed while write-lock is acquired?"));

            Load::Loading(task)
          },
        },
        AssetState::Loading(task) => match task.upgrade() {
          Some(task) => Load::Loading(task),
          None => {// Can happen if the caller calls Self::load and drops the Future before it's completed executing.
            let task = Self::spawn_load_task(Arc::clone(e.key())).shared();
            *e.get_mut() = AssetState::Loading(task.downgrade().expect("Task completed while write-lock is acquired?"));

            Load::Loading(task)
          },
        },
      },
      EntryRef::Vacant(e) => {
        let path = e.into_key();
        let task = Self::spawn_load_task(Arc::clone(&path)).shared();
        let previous = assets.insert(path, AssetState::Loading(task.downgrade().expect("Task completed while write-lock is acquired?")));
        assert!(previous.is_none(), "Duplicated asset!");

        Load::Loading(task)
      },
    }
  }

  #[cold]
  fn spawn_load_task(path: Arc<Path>) -> Task<CloneIoResult<AnyAsset>> {
    IoTaskPool::get().spawn(async move {
      match Self::spawn_load_task_inner(Arc::clone(&path)).await {
        Ok(a) => {
          // Mark asset as loaded in the AssetCache.
          let mut assets = CACHE.assets.write().unwrap();
          let state = assets.get_mut(&path).expect("Asset state should already be set!");
          assert!(matches!(state, AssetState::Loading(_)), "AssetState must be loading!");

          let asset = a.into_any();
          *state = AssetState::Loaded(AnyAsset::downgrade(&asset));

          CloneIoResult::Ok(asset)
        },
        Err(e) => {
          // Remove asset since it has failed to load.
          let previous = CACHE.assets.write().unwrap().remove(&path);
          assert!(matches!(previous, Some(AssetState::Loading(_))));
          CloneIoResult::Err(Arc::new(e))
        },
      }
    })
  }

  /// Helper returning Result so operator ? can be used.
  /// Passes in asset which is where the result will be written to.
  async fn spawn_load_task_inner(path: Arc<Path>) -> std::io::Result<Self> {
    let bytes = azart_utils::io::read(&path)?;
    let (header, body) = find_header_and_body(&bytes).ok_or_else(|| std::io::Error::other(format!("Could not find header/body divider {HEADER_DIVIDER} in file {path:?}")))?;

    let header: Header = ron::de::from_bytes(header).map_err(std::io::Error::other)?;
    assert_eq!(std::any::type_name::<T>(), header.type_name, "Type mismatch when deserializing asset at path {path:?}!");

    // Load dependencies first. They must remain referenced until the asset is fully-loaded.
    let dependencies = header.dependencies.iter().map(AnyAsset::load);
    let results = join_all(dependencies).await;

    let mut dependencies = Vec::with_capacity(results.len());
    for result in results.into_iter() {
      dependencies.push(result.map_err(|e| std::io::Error::other(format!("Dependency {path:?}: {e:?}")))?);
    }

    // Deserialize in correct format.
    let asset = match header.format {
      AssetFormat::Bin => Self::load_bin(Arc::clone(&path), body, header).map_err(std::io::Error::other)?,
      AssetFormat::Ron => Self::load_ron(Arc::clone(&path), body, header).map_err(std::io::Error::other)?,
    };

    Ok(asset)
  }

  fn load_bin(path: Arc<Path>, data: &[u8], header: Header) -> Result<Self, bincode::error::DecodeError> {
    let _scope = scoped_context(Context::Deserializing(header.dependencies));
    let asset = bincode::serde::decode_from_slice(data, bincode::config::standard())?.0;
    Ok(Self(Arc::new(AssetInner {
      path,
      value: asset,
    })))
  }

  fn load_ron(path: Arc<Path>, data: &[u8], header: Header) -> Result<Self, ron::de::Error> {
    let _scope = scoped_context(Context::Deserializing(header.dependencies));
    let asset = ron::de::from_bytes(data)?;
    Ok(Self(Arc::new(AssetInner {
      path,
      value: asset,
    })))
  }

  /// Returns whether this is the first time register for T has been called.
  pub fn register() -> bool {
    let ident = std::any::type_name::<T>();
    if CACHE.registry.read().unwrap().contains_key(ident) {
      false
    } else {
      CACHE.registry.write().unwrap().insert(ident, Self::vtable()).is_none()
    }
  }

  const fn vtable() -> &'static AssetVTable {
    &AssetVTable {
      load_bin: |path, data, header| Self::load_bin(path, data, header).map(Self::into_any),
      load_ron: |path, data, header| Self::load_ron(path, data, header).map(Self::into_any),
    }
  }
}

impl<T: AssetTrait> Asset<T> {
  /// Acquire a strong handle to the asset if it's already loaded by it's path.
  pub fn get(path: impl AsRef<Path>) -> Option<Self> {
    CACHE
      .assets
      .read()
      .unwrap()
      .get(path.as_ref())
      .and_then(|a| match a {
        AssetState::Loaded(a) => a
          .upgrade()
          .map(|a| a.downcast::<T>().unwrap()),
        _ => None,
      })
  }
}

impl<T: AssetTrait + serde::Serialize> Asset<T> {
  /// Asynchronously stores the asset at the specified path. Optionally returns the previous asset version that's been loaded.
  /// This will not overwrite any existing assets loaded from that path. Subsequent loads will return this asset.
  pub fn store(path: impl AsRef<Path>, asset: T, format: AssetFormat) -> (Store<T>, Option<Self>) {
    let path = path.as_ref();
    let mut assets = CACHE.assets.write().unwrap();

    match assets.entry_ref(path) {
      EntryRef::Occupied(mut e) => match e.get() {
        AssetState::Loaded(previous) => {
          let previous = previous.upgrade().map(|a| a.downcast::<T>().unwrap());
          let task = Self::spawn_store_task(Arc::clone(e.key()), asset, format, previous.clone().map(Either::Left)).shared();
          *e.get_mut() = AssetState::Loading(task.downgrade().expect("Load completed while write-lock is acquired?"));
          (Store::new(task), previous)
        },
        AssetState::Loading(task) => match task.upgrade() {
          Some(task) => {
            let task = Self::spawn_store_task(Arc::clone(e.key()), asset, format, Some(Either::Right(task))).shared();
            *e.get_mut() = AssetState::Loading(task.downgrade().expect("Load completed while write-lock is acquired?"));
            (Store::new(task), None)
          },
          None => {
            let task = Self::spawn_store_task(Arc::clone(e.key()), asset, format, None).shared();
            *e.get_mut() = AssetState::Loading(task.downgrade().expect("Load completed while write-lock is acquired?"));
            (Store::new(task), None)
          },
        }
      },
      EntryRef::Vacant(e) => {
        let path = e.into_key();
        let task = Self::spawn_store_task(Arc::clone(&path), asset, format, None).shared();
        assets.insert(path, AssetState::Loading(task.downgrade().expect("Load completed while write-lock is acquired?")));
        (Store::new(task), None)
      },
    }
  }

  fn spawn_store_task(path: Arc<Path>, asset: T, format: AssetFormat, previous: Option<Either<Self, Shared<Task<CloneIoResult<AnyAsset>>>>>) -> Task<CloneIoResult<AnyAsset>> {
    IoTaskPool::get().spawn(async move {
      let previous = match previous {
        Some(Either::Left(a)) => Some(a),
        Some(Either::Right(task)) => match task.await {
          CloneIoResult::Ok(a) => Some(a.downcast::<T>().unwrap()),
          _ => None,
        },
        None => None,
      };

      match Self::spawn_store_task_inner(Arc::clone(&path), asset, format).await {
        Ok(a) => {
          let asset = a.into_any();
          *CACHE.assets.write().unwrap().get_mut(&path).unwrap() = AssetState::Loaded(AnyAsset::downgrade(&asset));
          CloneIoResult::Ok(asset)
        },
        Err(e) => {
          match previous {
            Some(previous) => {
              let previous = previous.into_any();
              *CACHE.assets.write().unwrap().get_mut(&path).unwrap() = AssetState::Loaded(AnyAsset::downgrade(&previous));
              CloneIoResult::Ok(previous)
            },
            None => {
              let previous = CACHE.assets.write().unwrap().remove(&path);
              assert!(matches!(previous, Some(AssetState::Loading(_))));
              CloneIoResult::Err(Arc::new(e))
            },
          }
        },
      }
    })
  }

  async fn spawn_store_task_inner(path: Arc<Path>, asset: T, format: AssetFormat) -> std::io::Result<Self> {
    // Gather dependencies.
    let mut dependencies = Vec::new();
    asset.visit(&mut |dependency| dependencies.push(Arc::clone(dependency.path())));

    // Create header.
    let header = Header {
      type_name: std::any::type_name::<T>().to_owned(),
      format,
      dependencies: dependencies
        .iter()
        .map(|path| (**path).to_owned())
        .collect(),
    };

    let mut writer = std::io::BufWriter::new(Vec::new());

    ron::ser::to_writer_pretty(&mut writer, &header, ron::ser::PrettyConfig::default()).map_err(std::io::Error::other)?;

    // Add header divider.
    writer.write_all(HEADER_DIVIDER.as_bytes())?;

    let scope = scoped_context(Context::Serializing({
      let mut count = 0;
      let mut dependency_map = HashMap::new();
      for dependency in dependencies.into_iter() {
        if let Entry::Vacant(e) = dependency_map.entry(dependency) {
          e.insert(count);
          count += 1;
        }
      }

      dependency_map
    }));

    match format {
      AssetFormat::Bin => _ = bincode::serde::encode_into_std_write(&asset, &mut writer, bincode::config::standard()).map_err(std::io::Error::other)?,
      AssetFormat::Ron => ron::ser::to_writer_pretty(&mut writer, &asset, ron::ser::PrettyConfig::default()).map_err(std::io::Error::other)?,
    }

    drop(scope);

    let data = writer.into_inner()?;

    std::fs::write(&path, &data)?;
    drop(data);

    // Success. Return the newly created asset (now mirrored by the asset at the specified path).
    Ok(Self(Arc::new(AssetInner {
      path,
      value: asset,
    })))
  }
}

impl<T: AssetTrait> Debug for Asset<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct(pretty_type_name::pretty_type_name::<Self>().as_str())
      .field("value", &self.value)
      .field("path", self.path())
      .finish()
  }
}

#[derive(Serialize, Deserialize)]
#[serde(rename = "Asset")]
struct SerdeAsset {
  entry: u32,
}

impl<T: AssetTrait> serde::Serialize for Asset<T> {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let entry = CONTEXT.with_borrow(|cx| match cx {
      Context::Serializing(entries) => entries[self.path()],
      cx => panic!("Context was set to {cx:?} during serialization!"),
    });

    /*let mut fields = serializer.serialize_struct("Asset", 1)?;
    fields.serialize_field("entry", &entry)?;
    fields.end()*/

    SerdeAsset { entry }.serialize(serializer)
  }
}

impl<'de, T: AssetTrait> serde::Deserialize<'de> for Asset<T> {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    /*struct AssetVisitor<T: AssetTrait>(PhantomData<T>);
    impl<'de, T: AssetTrait> serde::de::Visitor<'de> for AssetVisitor<T> {
      type Value = Asset<T>;

      fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "an asset reference")
      }

      fn visit_u32<E: serde::de::Error>(self, entry: u32) -> Result<Self::Value, E> {
        CONTEXT.with_borrow(|cx| match cx {
          Context::Deserializing(paths) => {
            let path = &paths[entry as usize];
            Ok(Asset::<T>::get(path).unwrap_or_else(|| panic!("Dependency {path:?} at entry {entry} was not loaded before it's outer!")))
          },
          cx => panic!("Context was set to {cx:?} during serialization!"),
        })
      }
    }

    deserializer.deserialize_struct("Asset", &["entry"], AssetVisitor::<T>(PhantomData))*/

    SerdeAsset::deserialize(deserializer)
      .map(|SerdeAsset { entry }| CONTEXT.with_borrow(|cx| match cx {
        Context::Deserializing(paths) => {
          let path = &paths[entry as usize];
          Asset::<T>::get(path).unwrap_or_else(|| panic!("Dependency {path:?} at entry {entry} was not loaded before it's outer!"))
        },
        cx => panic!("Context was set to {cx:?} during serialization!"),
      }))
  }
}

#[derive(Clone, Debug, Deref, Reflect)]
pub struct WeakAsset<T: AssetTrait>(Weak<AssetInner<T>>);

impl<T: AssetTrait> WeakAsset<T> {
  #[inline]
  pub const fn new() -> Self {
    Self(Weak::new())
  }

  #[inline]
  pub fn upgrade(&self) -> Option<Asset<T>> {
    (**self).upgrade().map(Asset::new)
  }
}

/// An untyped asset that can be converted to an Asset<T>.
#[derive(Clone, Debug, Component, Resource)]
pub struct AnyAsset(Arc<dyn AnyAssetInner>);

impl AnyAsset {
  #[inline]
  const fn new(inner: Arc<dyn AnyAssetInner>) -> Self {
    Self(inner)
  }

  pub fn get(path: impl AsRef<Path>) -> Option<AnyAsset> {
    CACHE
      .assets
      .read()
      .unwrap()
      .get(path.as_ref())
      .and_then(|a| match a {
        AssetState::Loaded(a) => a.upgrade(),
        _ => None,
      })
  }

  pub fn load(path: impl AsRef<Path>) -> AnyLoad {
    let path = path.as_ref();
    if let Some(a) = Self::get(path) {
      return AnyLoad::Loaded(a);
    }

    let mut assets = CACHE.assets.write().unwrap();

    match assets.entry_ref(path) {
      EntryRef::Occupied(mut e) => match e.get() {
        AssetState::Loaded(a) => match a.upgrade() {
          Some(a) => AnyLoad::Loaded(a),
          None => {
            let task = Self::spawn_load_task(Arc::clone(e.key())).shared();
            *e.get_mut() = AssetState::Loading(task.downgrade().expect("Task completed while write-lock is acquired?"));

            AnyLoad::Loading(task)
          },
        },
        AssetState::Loading(task) => match task.upgrade() {
          Some(task) => AnyLoad::Loading(task),
          None => {// Can happen if the caller calls Self::load and drops the Future before it's completed executing.
            let task = Self::spawn_load_task(Arc::clone(e.key())).shared();
            *e.get_mut() = AssetState::Loading(task.downgrade().expect("Task completed while write-lock is acquired?"));

            AnyLoad::Loading(task)
          },
        },
      },
      EntryRef::Vacant(e) => {
        let path = e.into_key();
        let task = Self::spawn_load_task(Arc::clone(&path)).shared();
        let previous = assets.insert(path, AssetState::Loading(task.downgrade().expect("Task completed while write-lock is acquired?")));
        assert!(previous.is_none(), "Duplicated asset!");

        AnyLoad::Loading(task)
      },
    }
  }

  #[cold]
  fn spawn_load_task(path: Arc<Path>) -> Task<CloneIoResult<AnyAsset>> {
    IoTaskPool::get().spawn(async move {
      match Self::spawn_load_task_inner(Arc::clone(&path)).await {
        Ok(a) => {
          // Mark asset as loaded in the AssetCache.
          let mut assets = CACHE.assets.write().unwrap();
          let state = assets.get_mut(&path).expect("Asset state should already be set!");
          assert!(matches!(state, AssetState::Loading(_)), "AssetState must be loading!");

          *state = AssetState::Loaded(AnyAsset::downgrade(&a));

          CloneIoResult::Ok(a)
        },
        Err(e) => {
          // Remove asset since it has failed to load.
          CACHE.assets.write().unwrap().remove(&path);
          CloneIoResult::Err(Arc::new(e))
        },
      }
    })
  }

  /// Helper returning Result so operator ? can be used.
  /// Passes in asset which is where the result will be written to.
  async fn spawn_load_task_inner(path: Arc<Path>) -> std::io::Result<Self> {
    let bytes = azart_utils::io::read(&path)?;
    let (header, body) = find_header_and_body(&bytes)
      .ok_or_else(|| std::io::Error::other(format!("Could not find header/body divider {HEADER_DIVIDER} in file {path:?}")))?;

    let header: Header = ron::de::from_bytes(header).map_err(std::io::Error::other)?;

    let vtable = *CACHE
      .registry
      .read()
      .unwrap()
      .get(header.type_name.as_str())
      .unwrap_or_else(|| panic!("Asset type {} is pending load but is unregistered!", pretty_type_name::pretty_type_name_str(header.type_name.as_str())));

    // Load dependencies first. They must remain referenced until the asset is fully-loaded.
    let dependencies = header.dependencies.iter().map(AnyAsset::load);
    let results = join_all(dependencies).await;

    let mut dependencies = Vec::with_capacity(results.len());
    for result in results.into_iter() {
      dependencies.push(result.map_err(|e| std::io::Error::other(format!("Dependency {path:?}: {e:?}")))?);
    }

    let asset = match header.format {
      AssetFormat::Bin => (vtable.load_bin)(Arc::clone(&path), body, header).map_err(std::io::Error::other)?,
      AssetFormat::Ron => (vtable.load_ron)(Arc::clone(&path), body, header).map_err(std::io::Error::other)?,
    };

    Ok(asset)
  }

  #[inline]
  pub fn is<T: AssetTrait>(&self) -> bool {
    (*self.0).type_id() == TypeId::of::<AssetInner<T>>()
  }

  #[inline]
  pub fn downcast<T: AssetTrait>(self) -> Result<Asset<T>, Self> {
    if self.is::<T>() {
      Ok(Asset::new(Arc::downcast::<AssetInner<T>>(self.into_inner()).unwrap()))
    } else {
      Err(self)
    }
  }

  #[inline]
  pub fn downgrade(this: &Self) -> WeakAnyAsset {
    WeakAnyAsset(Arc::downgrade(&this.0))
  }

  #[inline]
  pub fn inner(&self) -> &dyn Any {
    self.0.get()
  }

  #[inline]
  pub fn path(&self) -> &Arc<Path> {
    self.0.path()
  }

  #[inline]
  fn into_inner(self) -> Arc<dyn AnyAssetInner> {
    self.0
  }
}

#[derive(Clone, Debug, Component, Resource)]
pub struct WeakAnyAsset(Weak<dyn AnyAssetInner>);

impl WeakAnyAsset {
  #[inline]
  pub const fn new() -> Self {
    Self(Weak::<AssetInner<AnyAsset>>::new())
  }

  #[inline]
  pub fn upgrade(&self) -> Option<AnyAsset> {
    self.0.upgrade().map(AnyAsset::new)
  }
}

#[derive(Debug, Deref, Reflect)]
pub struct AssetInner<T: AssetTrait> {
  path: Arc<Path>,
  #[deref]
  value: T,
}

impl<T: AssetTrait> AssetInner<T> {
  #[inline]
  pub fn path(&self) -> &Arc<Path> {
    &self.path
  }
}

impl<T: AssetTrait> AnyAssetInner for AssetInner<T> {
  fn path(&self) -> &Arc<Path> {
    &self.path
  }

  fn get(&self) -> &dyn Any {
    &self.value as _
  }
}

trait AnyAssetInner: AssetDependencies + Debug + Any + Send + Sync + 'static {
  fn path(&self) -> &Arc<Path>;
  fn get(&self) -> &dyn Any;
}

struct AssetCache {
  assets: CachePadded<RwLock<HashMap<Arc<Path>, AssetState>>>,
  registry: CachePadded<RwLock<HashMap<&'static str, &'static AssetVTable>>>,
}

enum AssetState {
  Loaded(WeakAnyAsset),
  Loading(WeakShared<Task<CloneIoResult<AnyAsset>>>),
}

/// Result and std::io::Error are not cloneable which is required by shared Futures. This is a workaround to that.
#[derive(Clone, Debug)]
pub enum CloneIoResult<T: Clone> {
  Ok(T),
  Err(Arc<std::io::Error>),// std::io::Error is not cloneable so is moved into an Arc.
}

#[derive(Debug)]
pub enum AnyLoad {
  Loaded(AnyAsset),
  Loading(Shared<Task<CloneIoResult<AnyAsset>>>),
  Awaited,
}

impl Future for AnyLoad {
  type Output = Result<AnyAsset, Arc<std::io::Error>>;
  fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
    match &mut *self {
      Self::Loaded(_) => {
        let Self::Loaded(a) = std::mem::replace(&mut *self, Self::Awaited) else {
          unreachable!();
        };

        Poll::Ready(Ok(a))
      },
      Self::Loading(task) => match std::pin::Pin::new(task).poll(cx) {
        Poll::Ready(CloneIoResult::Ok(a)) => Poll::Ready(Ok(a)),
        Poll::Ready(CloneIoResult::Err(e)) => Poll::Ready(Err(e)),
        Poll::Pending => Poll::Pending,
      },
      Self::Awaited => unreachable!("Attempted to poll a Future that's already been awaited!"),
    }
  }
}

#[derive(Debug)]
#[must_use = "Future must be awaited!"]
pub enum Load<T: AssetTrait> {
  Loaded(Asset<T>),
  Loading(Shared<Task<CloneIoResult<AnyAsset>>>),
  Awaited,
}

impl<T: AssetTrait> Future for Load<T> {
  type Output = Result<Asset<T>, Arc<std::io::Error>>;
  fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
    match &mut *self {
      Self::Loaded(_) => {
        let Self::Loaded(a) = std::mem::replace(&mut *self, Self::Awaited) else {
          unreachable!();
        };

        Poll::Ready(Ok(a))
      },
      Self::Loading(task) => match std::pin::Pin::new(task).poll(cx) {
        Poll::Ready(CloneIoResult::Ok(a)) => Poll::Ready(Ok(a.downcast::<T>().unwrap())),
        Poll::Ready(CloneIoResult::Err(e)) => Poll::Ready(Err(e)),
        Poll::Pending => Poll::Pending,
      },
      Load::Awaited => unreachable!("Attempted to poll a Future that's already been awaited!"),
    }
  }
}

#[derive(Debug)]
#[must_use = "Future must be awaited!"]
pub struct Store<T: AssetTrait>(Shared<Task<CloneIoResult<AnyAsset>>>, PhantomData<&'static T>);

impl<T: AssetTrait> Store<T> {
  #[inline]
  const fn new(inner: Shared<Task<CloneIoResult<AnyAsset>>>) -> Self {
    Self(inner, PhantomData)
  }
}

impl<T: AssetTrait> Future for Store<T> {
  type Output = Result<Asset<T>, Arc<std::io::Error>>;
  fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
    match std::pin::Pin::new(&mut self.0).poll(cx) {
      Poll::Ready(CloneIoResult::Ok(a)) => Poll::Ready(Ok(a.downcast::<T>().unwrap())),
      Poll::Ready(CloneIoResult::Err(e)) => Poll::Ready(Err(e)),
      Poll::Pending => Poll::Pending,
    }
  }
}

pub trait AssetTrait: AssetDependencies + Any + Debug + Send + Sync + 'static {}
impl<T: AssetDependencies + Any + Debug + Send + Sync + 'static> AssetTrait for T {}

const HEADER_DIVIDER: &str = "\n---\n";

#[inline]
pub fn find_header_and_body(data: &[u8]) -> Option<(&[u8], &[u8])> {
  memchr::memmem::find(data, HEADER_DIVIDER.as_bytes()).map(|pos| (&data[0..pos], &data[pos + HEADER_DIVIDER.len()..]))
}

struct AssetVTable {
  load_bin: fn(path: Arc<Path>, data: &[u8], header: Header) -> Result<AnyAsset, bincode::error::DecodeError>,
  load_ron: fn(path: Arc<Path>, data: &[u8], header: Header) -> Result<AnyAsset, ron::de::Error>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Header {
  type_name: String,
  format: AssetFormat,
  dependencies: Vec<PathBuf>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum AssetFormat {
  Bin,
  Ron,
}

/// Global context struct used to
#[derive(Debug)]
enum Context {
  Serializing(HashMap<Arc<Path>, u32>),
  Deserializing(Vec<PathBuf>),
  None,
}

#[inline]
#[must_use = "RAII scope must be bound!"]
fn scoped_context(cx: Context) -> impl Drop {
  let previous = CONTEXT.replace(cx);
  assert!(matches!(previous, Context::None));
  scopeguard::guard((), |_| CONTEXT.set(Context::None))
}

pub trait AssetDependencies {
  /// Default implementation is noop.
  #[allow(unused_variables)]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) { /* noop */ }
}

impl AssetDependencies for AnyAsset {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    visitor(self);
  }
}

impl<T: AssetTrait> AssetDependencies for Asset<T> {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    visitor(&self.clone().into_any());
  }
}

impl<T: AssetTrait> AssetDependencies for AssetInner<T> {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    (**self).visit(visitor);
  }
}

impl<T: AssetDependencies> AssetDependencies for [T] {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    for v in self.iter() {
      v.visit(visitor);
    }
  }
}

impl<K, T: AssetDependencies> AssetDependencies for HashMap<K, T> {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    for (_, v) in self.iter() {
      v.visit(visitor);
    }
  }
}

impl<T: AssetDependencies> AssetDependencies for Option<T> {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    if let Some(dependency) = self {
      dependency.visit(visitor);
    }
  }
}

impl<T: AssetDependencies> AssetDependencies for &T {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    (**self).visit(visitor);
  }
}

impl<T: AssetDependencies> AssetDependencies for &mut T {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    (**self).visit(visitor);
  }
}

impl<T: AssetDependencies> AssetDependencies for Box<T> {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    (**self).visit(visitor);
  }
}

impl<T: AssetDependencies> AssetDependencies for Arc<T> {
  #[inline]
  fn visit(&self, visitor: &mut dyn FnMut(&AnyAsset)) {
    (**self).visit(visitor);
  }
}

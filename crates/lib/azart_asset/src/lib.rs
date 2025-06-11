use std::{any::{Any, TypeId}, cell::RefCell, fmt::Debug, io::Write, ops::Deref, path::Path, sync::{Arc, LazyLock, Mutex, Once, OnceLock, RwLock, Weak}};
use std::io::Error;
use bevy::{app::Plugin, ecs::{component::Component, system::Resource}, prelude::{Deref, DerefMut}, reflect::Reflect, tasks::{block_on, IoTaskPool, Task}, utils::{hashbrown::hash_map::EntryRef, HashMap}};
use crossbeam::utils::CachePadded;
use either::Either;
use futures::{future::{Shared, WeakShared}, FutureExt};
use scopeguard::defer;
use serde::{Deserialize, Serialize};

/// Exports.
pub use serde;
pub use serde_bytes;
pub use bincode;
pub use ron;
pub use serde_yaml;

/// @TODO: Should be selected by build configuration.
const METADATA_ENCODING: AssetEncoding = AssetEncoding::Ron;

pub struct AssetPlugin;

impl Plugin for AssetPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    app.insert_resource(AssetCache::new());
  }
}

pub struct RegisterAssetPlugin<T: AssetHandler> {
  handler: Mutex<Option<T>>,
}

impl<T: AssetHandler> RegisterAssetPlugin<T> {
  #[inline]
  pub const fn with_handler(handler: T) -> Self {
    Self { handler: Mutex::new(Some(handler)) }
  }
}

impl<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned> Default for RegisterAssetPlugin<RonAssetHandler<T>> {
  #[inline]
  fn default() -> Self {
    Self::with_handler(RonAssetHandler::<T>::default())
  }
}

impl<T: AssetHandler> Plugin for RegisterAssetPlugin<T> {
  fn build(&self, app: &mut bevy::app::App) {
    let mut cache = app
      .world_mut()
      .get_resource_mut::<AssetCache>()
      .unwrap_or_else(|| panic!("RegisterAssetPlugin built before AssetCache was inserted as a resource!"));

    let handler = self
      .handler
      .lock()
      .unwrap()
      .take()
      .unwrap();

    cache.register_with_handler(handler);
  }
}

#[derive(Default, Resource)]
pub struct AssetCache {
  inner: Arc<AssetCacheInner>,
}

impl AssetCache {
  #[inline]
  pub fn new() -> Self {
    Self::default()
  }

  /// Registers a type with the AssetCache. Returns whether the type has already been registered or not.
  pub fn register<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned>(&mut self) -> Option<Arc<dyn DynAssetHandler>> {
    self.register_with_handler(RonAssetHandler::<T>::default()).1
  }

  /// Registers a type with a custom AssetHandler (defines how assets are loaded and stored). Returns the previous displaced handler.
  pub fn register_with_handler<H: AssetHandler>(&mut self, handler: H) -> (Arc<H>, Option<Arc<dyn DynAssetHandler>>) {
    let inner = Arc::get_mut(&mut self.inner).expect("Attempted to register type to AssetCache after it's in-use!");
    let AssetTypeRegistry { type_map, name_map } = &mut *inner.registry;

    let handler = Arc::new(handler);
    type_map.insert(TypeId::of::<H::Target>(), Arc::clone(&handler) as _);
    let previous = name_map.insert(std::any::type_name::<H::Target>(), Arc::clone(&handler) as _);
    (handler, previous)
  }

  pub fn get<T: Sized + AssetResource>(&self, path: impl AsRef<Path>) -> Option<Asset<T>> {
    self
      .get_any(path)
      .map(|a| a.downcast::<T>().unwrap())
  }

  pub fn get_any(&self, path: impl AsRef<Path>) -> Option<AnyAsset> {
    self
      .inner
      .assets
      .read()
      .unwrap()
      .get(path.as_ref())
      .and_then(|a| match a {
        AssetState::Loaded(a) => a.upgrade(),
        _ => None,
      })
  }

  pub fn load<T: AssetResource>(&self, path: impl AsRef<Path>) -> Load<T> {
    match self.load_any(path) {
      AnyLoad::Loaded(a) => Load::Loaded(a.downcast::<T>().unwrap()),
      AnyLoad::Pending(task) => Load::Pending(task),
      AnyLoad::Awaited => unreachable!(),
    }
  }

  pub fn load_any(&self, path: impl AsRef<Path>) -> AnyLoad {
    let path = path.as_ref();

    if let Some(a) = self.get_any(path) {
      return AnyLoad::Loaded(a);
    }

    let mut assets = self.inner.assets.write().unwrap();
    match assets.entry_ref(path) {
      EntryRef::Occupied(mut e) => match e.get() {
        AssetState::Loaded(a) => match a.upgrade() {
          Some(a) => AnyLoad::Loaded(a),
          None => {
            let task = self.spawn_load_task(Arc::clone(e.key())).shared();
            *e.get_mut() = AssetState::Pending(task.downgrade().unwrap());
            AnyLoad::Pending(task)
          },
        },
        AssetState::Pending(task) => match task.upgrade() {
          Some(task) => AnyLoad::Pending(task),
          None => {
            let task = self.spawn_load_task(Arc::clone(e.key())).shared();
            *e.get_mut() = AssetState::Pending(task.downgrade().unwrap());
            AnyLoad::Pending(task)
          },
        }
      },
      EntryRef::Vacant(e) => {
        let path = e.into_key();
        let task = self.spawn_load_task(Arc::clone(&path)).shared();
        assets.insert(path, AssetState::Pending(task.downgrade().unwrap()));
        AnyLoad::Pending(task)
      },
    }
  }

  /// Stores the new asset value at the file path. Optionally returning the previous asset. Existing assets will not be overwritten.
  pub fn store<T: AssetResource + serde::Serialize>(&self, path: impl AsRef<Path>, value: T) -> Store<T> {
    let path = path.as_ref();
    let mut assets = self.inner.assets.write().unwrap();
    match assets.entry_ref(path) {
      EntryRef::Occupied(mut e) => {
        let previous = match e.get() {
          AssetState::Loaded(a) => a.upgrade().map(Either::Left),
          AssetState::Pending(t) => t.upgrade().map(Either::Right),
        };

        let task = self.spawn_store_task(Arc::clone(e.key()), value, previous).shared();
        *e.get_mut() = AssetState::Pending(task.downgrade().unwrap());

        Store::new(task)
      },
      EntryRef::Vacant(e) => {
        let path = e.into_key();
        let task = self.spawn_store_task(Arc::clone(&path), value, None).shared();
        let previous = assets.insert(path, AssetState::Pending(task.downgrade().unwrap()));
        assert!(previous.is_none());

        Store::new(task)
      },
    }
  }

  fn spawn_load_task(&self, path: Arc<Path>) -> Task<CloneIoResult<AnyAsset>> {
    let inner = Arc::clone(&self.inner);
    IoTaskPool::get().spawn(async move {
      let cache = Self { inner };
      match cache.load_inner(Arc::clone(&path)).await {
        Ok(a) => {
          // Mark as loaded.
          let previous = std::mem::replace(cache.inner.assets.write().unwrap().get_mut(&path).unwrap(), AssetState::Loaded(Asset::downgrade(&a)));
          assert!(matches!(previous, AssetState::Pending(_)));

          CloneIoResult::Ok(a)
        },
        Err(e) => {
          // Remove from cache. Failed to load.
          let previous = cache.inner.assets.write().unwrap().remove(&path);
          assert!(matches!(previous, Some(AssetState::Pending(_))));

          CloneIoResult::Err(Arc::new(e))
        },
      }
    })
  }

  fn spawn_store_task<T: AssetResource + serde::Serialize>(&self, path: Arc<Path>, value: T, previous: Option<Either<AnyAsset, Shared<Task<CloneIoResult<AnyAsset>>>>>) -> Task<CloneIoResult<AnyAsset>> {
    let inner = Arc::clone(&self.inner);
    IoTaskPool::get().spawn(async move {
      // Must await previous task (if exists) before continuing.
      let previous = match previous {
        Some(Either::Left(a)) => Some(a),
        Some(Either::Right(t)) => match t.await {
          CloneIoResult::Ok(a) => Some(a),
          _ => None,
        },
        None => None,
      };

      let cache = Self { inner };
      match cache.store_inner(Arc::clone(&path), value).await {
        Ok(a) => {
          let a = a.into_any();

          // Mark as loaded.
          let previous = std::mem::replace(cache.inner.assets.write().unwrap().get_mut(&path).unwrap(), AssetState::Loaded(Asset::downgrade(&a)));
          assert!(matches!(previous, AssetState::Pending(_)));

          CloneIoResult::Ok(a)
        },
        Err(e) => {
          let mut assets = cache.inner.assets.write().unwrap();
          if let Some(a) = previous {
            *assets.get_mut(&path).unwrap() = AssetState::Loaded(Asset::downgrade(&a));
          } else {
            let previous = assets.remove(&path);
            assert!(matches!(previous, Some(AssetState::Pending(_))));
          }

          CloneIoResult::Err(Arc::new(e))
        },
      }
    })
  }

  async fn load_inner(&self, path: Arc<Path>) -> std::io::Result<AnyAsset> {
    use std::io::Error;

    let data = azart_utils::io::read(&path)?;

    let (type_name, data) = type_name_and_body(&data)
      .ok_or_else(|| Error::other(format!("Failed to find type_name metadata in {path:?}!")))?;

    let handler = self
      .inner
      .registry
      .name_map
      .get(type_name)
      .unwrap_or_else(|| panic!("Type {} has not been registered!", type_name));

    let previous_cx = SERDE_DESERIALIZE_CONTEXT.with_borrow_mut(|cx| std::mem::replace(cx, Some(AssetCache { inner: Arc::clone(&self.inner) })));
    defer! { SERDE_DESERIALIZE_CONTEXT.with_borrow_mut(|cx| *cx = previous_cx); }

    let asset = handler.dyn_load(Arc::clone(&path), data)?;

    Ok(asset)
  }

  async fn store_inner<T: AssetResource + serde::Serialize>(&self, path: Arc<Path>, value: T) -> std::io::Result<Asset<T>> {
    let handler = self
      .inner
      .registry
      .type_map
      .get(&TypeId::of::<T>())
      .unwrap_or_else(|| panic!("Type {} has not been registered!", std::any::type_name::<T>()));

    let mut bytes = format!("{}\n", handler.dyn_type_name()).into_bytes();

    // @TODO: Use Writer and avoid extra allocation. Doesn't seem to work with Ron though.
    bytes.extend_from_slice(handler.dyn_store(&value)?.as_slice());

    azart_utils::io::write(&path, &bytes)?;

    Ok(Asset(Arc::new(AssetInner {
      path,
      value,
    })))
  }
}

#[derive(Default)]
struct AssetCacheInner {
  registry: CachePadded<AssetTypeRegistry>,
  assets: CachePadded<RwLock<HashMap<Arc<Path>, AssetState>>>,
}

enum AssetState {
  Loaded(WeakAnyAsset),
  /// Represents either a load or store task.
  Pending(WeakShared<Task<CloneIoResult<AnyAsset>>>),
}

#[derive(Clone)]
pub enum CloneIoResult<T: Clone> {
  Ok(T),
  Err(Arc<std::io::Error>),
}

#[derive(Default)]
struct AssetTypeRegistry {
  type_map: HashMap<TypeId, Arc<dyn DynAssetHandler>>,
  name_map: HashMap<&'static str, Arc<dyn DynAssetHandler>>,
}

#[derive(Reflect, Deref, DerefMut, Component, Resource)]
pub struct Asset<T: ?Sized + AssetResource>(Arc<AssetInner<T>>);

impl<T: ?Sized + AssetResource> Asset<T> {
  #[inline]
  pub fn is<Into: Sized + AssetResource>(&self) -> bool {
    self.value.type_id() == TypeId::of::<Into>()
  }

  pub fn downcast<Into: Sized + AssetResource>(self) -> Result<Asset<Into>, Self> {
    if self.is::<Into>() {
      Ok(Asset(unsafe {
        let raw = Arc::into_raw(self.0);
        Arc::from_raw(raw as *const _)
      }))
    } else {
      Err(self)
    }
  }

  pub fn downgrade(this: &Self) -> WeakAsset<T> {
    WeakAsset(Arc::downgrade(&this.0))
  }
}

impl<T: AssetResource> Asset<T> {
  #[inline]
  pub fn into_any(self) -> AnyAsset {
    Asset(self.0 as _)
  }
}

impl<T: ?Sized + AssetResource> Clone for Asset<T> {
  fn clone(&self) -> Self {
    Self(Arc::clone(&self.0))
  }
}

impl<T: ?Sized + AssetResource> Debug for Asset<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self.path())
  }
}

impl<T: ?Sized + AssetResource> serde::Serialize for Asset<T> {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    SerdeAsset::new(self.path()).serialize(serializer)
  }
}

impl<'de, T: AssetResource> serde::Deserialize<'de> for Asset<T> {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    let SerdeAsset { path } = SerdeAsset::deserialize(deserializer)?;
    // @TODO: Maybe replace block_on call? Would be nice if this function was async.
    Ok(block_on(SERDE_DESERIALIZE_CONTEXT.with_borrow(|cache| cache.as_ref().unwrap().load::<T>(path)))
      .map_err(|e| serde::de::Error::custom(format!("Aborted early while loading {path:?}!: {e}")))?
    )
  }
}

impl<'de> serde::Deserialize<'de> for AnyAsset {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    let SerdeAsset { path } = SerdeAsset::deserialize(deserializer)?;
    // @TODO: Maybe replace block_on call? Would be nice if this function was async.
    Ok(block_on(SERDE_DESERIALIZE_CONTEXT.with_borrow(|cache| cache.as_ref().unwrap().load_any(path))).map_err(serde::de::Error::custom)?)
  }
}

/// An asset reference with weak semantics. Can not be serialized. Must be upgraded to an Asset.
#[derive(Clone, Debug)]
pub struct WeakAsset<T: ?Sized + AssetResource>(Weak<AssetInner<T>>);

impl<T: ?Sized + AssetResource> WeakAsset<T> {
  #[inline]
  #[must_use]
  pub fn upgrade(&self) -> Option<Asset<T>> {
    self.0.upgrade().map(Asset)
  }
}

impl<T: AssetResource> WeakAsset<T> {
  #[inline]
  pub const fn new() -> Self {
    Self(Weak::new())
  }
}

pub type AnyAsset = Asset<dyn AssetResource>;
pub type WeakAnyAsset = WeakAsset<dyn AssetResource>;

#[derive(Reflect, Debug, Deref, DerefMut)]
pub struct AssetInner<T: ?Sized + AssetResource> {
  path: Arc<Path>,
  #[deref]
  value: T,
}

impl<T: ?Sized + AssetResource> AssetInner<T> {
  #[inline]
  pub fn path(&self) -> &Arc<Path> {
    &self.path
  }
}

#[derive(Debug)]
#[must_use = "Must be awaited! If this is dropped before completion, the load task will be cancelled!"]
pub enum Load<T: AssetResource> {
  Loaded(Asset<T>),
  Pending(Shared<Task<CloneIoResult<AnyAsset>>>),
  Awaited,
}

impl<T: AssetResource> Future for Load<T> {
  type Output = Result<Asset<T>, Arc<std::io::Error>>;

  fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
    use std::task::Poll;
    match &mut *self {
      Self::Loaded(_) => {
        let Self::Loaded(a) = std::mem::replace(&mut *self, Self::Awaited) else {
          unreachable!();
        };

        Poll::Ready(Ok(a))
      },
      Self::Pending(task) => match std::pin::Pin::new(task).poll(cx) {
        Poll::Ready(CloneIoResult::Ok(a)) => Poll::Ready(Ok(a.downcast::<T>().unwrap())),
        Poll::Ready(CloneIoResult::Err(e)) => Poll::Ready(Err(e)),
        Poll::Pending => Poll::Pending,
      },
      Self::Awaited => unreachable!("Attempted to poll a Future that's already been awaited!"),
    }
  }
}

#[derive(Debug)]
#[must_use = "Must be awaited! If this is dropped before completion, the load task will be cancelled!"]
pub enum AnyLoad {
  Loaded(AnyAsset),
  Pending(Shared<Task<CloneIoResult<AnyAsset>>>),
  Awaited,
}

impl Future for AnyLoad {
  type Output = Result<AnyAsset, Arc<std::io::Error>>;
  fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
    use std::task::Poll;
    match &mut *self {
      Self::Loaded(_) => {
        let Self::Loaded(a) = std::mem::replace(&mut *self, Self::Awaited) else {
          unreachable!();
        };

        Poll::Ready(Ok(a))
      },
      Self::Pending(task) => match std::pin::Pin::new(task).poll(cx) {
        Poll::Ready(CloneIoResult::Ok(a)) => Poll::Ready(Ok(a)),
        Poll::Ready(CloneIoResult::Err(e)) => Poll::Ready(Err(e)),
        Poll::Pending => Poll::Pending,
      },
      Self::Awaited => unreachable!("Attempted to poll a Future that's already been awaited!"),
    }
  }
}

#[derive(Debug)]
#[must_use = "Must be awaited! If this is dropped before completion, the store task will be cancelled!"]
pub struct Store<T: AssetResource>(Shared<Task<CloneIoResult<AnyAsset>>>, std::marker::PhantomData<&'static T>);

impl<T: AssetResource> Store<T> {
  #[inline]
  pub const fn new(task: Shared<Task<CloneIoResult<AnyAsset>>>) -> Self {
    Self(task, std::marker::PhantomData)
  }
}

impl<T: AssetResource> Future for Store<T> {
  type Output = Result<Asset<T>, Arc<Error>>;

  fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
    use std::task::Poll;
    match std::pin::Pin::new(&mut self.0).poll(cx) {
      Poll::Ready(CloneIoResult::Ok(a)) => Poll::Ready(Ok(a.downcast::<T>().unwrap())),
      Poll::Ready(CloneIoResult::Err(e)) => Poll::Ready(Err(e)),
      Poll::Pending => Poll::Pending,
    }
  }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum AssetEncoding {
  Bin,
  Ron,
  Yaml,
}

impl AssetEncoding {
  pub const BIN: u8 = Self::Bin as _;
  pub const RON: u8 = Self::Ron as _;
  pub const YAML: u8 = Self::Yaml as _;
}

impl TryInto<AssetEncoding> for u8 {
  type Error = &'static str;
  fn try_into(self) -> Result<AssetEncoding, Self::Error> {
    match self {
      AssetEncoding::BIN => Ok(AssetEncoding::Bin),
      AssetEncoding::RON => Ok(AssetEncoding::Ron),
      AssetEncoding::YAML => Ok(AssetEncoding::Yaml),
      _ => Err("Out of range!"),
    }
  }
}

pub trait AssetResource: Any + Send + Sync + 'static {}
impl<T: Any + Send + Sync + 'static> AssetResource for T {}

/// The Serde representation of the Asset. Only the path is serialized.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Deref, Serialize, Deserialize)]
#[serde(rename = "Asset")]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct SerdeAsset<'a> {
  /// @HACK: &'a Path doesn't work (Serde doesn't place the correct lifetime constraints).
  #[deref]
  pub path: &'a Path,
}

impl<'a> SerdeAsset<'a> {
  #[inline]
  pub fn new(path: &'a Path) -> Self {
    Self { path }
  }
}

thread_local! {
  static SERDE_DESERIALIZE_CONTEXT: RefCell<Option<AssetCache>> = const { RefCell::new(None) };
}

pub struct BinAssetHandler<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned>(std::marker::PhantomData<Box<T>>);

impl<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned> Default for BinAssetHandler<T> {
  #[inline]
  fn default() -> Self {
    Self(std::marker::PhantomData)
  }
}

impl<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned> AssetHandler for BinAssetHandler<T> {
  type Target = T;

  fn load(&self, data: &[u8]) -> std::io::Result<Self::Target> {
    Ok(bincode::serde::decode_from_slice(data, bincode::config::standard()).map_err(Error::other)?.0)
  }

  fn store(&self, value: &Self::Target) -> std::io::Result<Vec<u8>> {
    Ok(bincode::serde::encode_to_vec(&value, bincode::config::standard()).map_err(Error::other)?)
  }
}


pub struct RonAssetHandler<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned>(std::marker::PhantomData<Box<T>>);

impl<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned> Default for RonAssetHandler<T> {
  #[inline]
  fn default() -> Self {
    Self(std::marker::PhantomData)
  }
}

impl<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned> AssetHandler for RonAssetHandler<T> {
  type Target = T;

  fn load(&self, data: &[u8]) -> std::io::Result<Self::Target> {
    Ok(ron::de::from_bytes(data).map_err(Error::other)?)
  }

  fn store(&self, value: &Self::Target) -> std::io::Result<Vec<u8>> {
    Ok(ron::ser::to_string_pretty(&value, ron::ser::PrettyConfig::default()).map(String::into_bytes).map_err(Error::other)?)
  }
}

pub struct YamlAssetHandler<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned>(std::marker::PhantomData<Box<T>>);

impl<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned> Default for YamlAssetHandler<T> {
  #[inline]
  fn default() -> Self {
    Self(std::marker::PhantomData)
  }
}

impl<T: AssetResource + serde::Serialize + serde::de::DeserializeOwned> AssetHandler for YamlAssetHandler<T> {
  type Target = T;

  fn load(&self, data: &[u8]) -> std::io::Result<Self::Target> {
    Ok(serde_yaml::from_slice(data).map_err(Error::other)?)
  }

  fn store(&self, value: &Self::Target) -> std::io::Result<Vec<u8>> {
    Ok(serde_yaml::to_string(&value).map_err(Error::other)?.into_bytes())
  }
}

pub type DefaultAssetHandler<T> = YamlAssetHandler<T>;

pub trait AssetHandler: Send + Sync + 'static {
  type Target: AssetResource;

  fn type_name(&self) -> &str { std::any::type_name::<Self::Target>() }
  fn load(&self, data: &[u8]) -> std::io::Result<Self::Target>;
  fn store(&self, value: &Self::Target) -> std::io::Result<Vec<u8>>;
}

/// Dynamic-dispatch wrapper over AssetHandler. Automatically implemented for any type that implements AssetHandler.
pub trait DynAssetHandler: Send + Sync + 'static {
  fn dyn_type_name(&self) -> &str;
  fn dyn_load(&self, path: Arc<Path>, data: &[u8]) -> std::io::Result<AnyAsset>;
  fn dyn_store(&self, value: &dyn Any) -> std::io::Result<Vec<u8>>;
}

impl<T> DynAssetHandler for T
where
  T: AssetHandler,
  T::Target: AssetResource + Sized,
{
  fn dyn_type_name(&self) -> &str {
    <_ as AssetHandler>::type_name(self)
  }

  fn dyn_load(&self, path: Arc<Path>, data: &[u8]) -> std::io::Result<AnyAsset> {
    <_ as AssetHandler>::load(self, data)
      .map(|value| Asset(Arc::new(AssetInner {
        path,
        value,
      })).into_any())
  }

  fn dyn_store(&self, value: &dyn Any) -> std::io::Result<Vec<u8>> {
    <_ as AssetHandler>::store(self, value.downcast_ref::<T::Target>().unwrap())
  }
}

fn type_name_and_body(data: &[u8]) -> Option<(&str, &[u8])> {
  data
    .iter()
    .enumerate()
    .find_map(|(i, &v)| (v == b'\n').then(|| i))
    .and_then(|i| str::from_utf8(&data[..i])
      .ok()
      .map(|name| (name, &data[i + 1..]))
    )
}

/// Store a raw asset at the file path with metadata to read it from the asset cache.
pub fn store(path: impl AsRef<Path>, type_name: &str, data: &[u8]) -> std::io::Result<()> {
  let mut bytes = format!("{type_name}\n").into_bytes();

  // @TODO: Use Writer and avoid extra allocation. Doesn't seem to work with Ron though.
  bytes.extend_from_slice(data);

  azart_utils::io::write(path.as_ref(), &bytes)?;

  Ok(())
}

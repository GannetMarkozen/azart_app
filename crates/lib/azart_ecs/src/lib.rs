mod bitset;

use std::any::TypeId;
use std::array::from_fn;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::{mem, slice};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock, RwLock};
use azart_task::Task;

#[must_use = "App does nothing unless explicitly ran!"]
pub struct App {
	
}

impl App {
	pub const fn new() -> Self {
		Self {}
	}

	pub fn system<S>(&mut self, s: S) -> &mut Self
		where
			S: System + 'static,
	{
		

		self
	}
}

pub struct World {
	
}

impl World {
}

pub trait System: Send {
	fn run(&self, world: &World) -> Option<Task<()>>;
}

pub struct SystemDesc<T: System> {
	pub system: T,
}

thread_local! {
	static LOCAL_MAP: RefCell<HashMap<TypeId, CompId>> = RefCell::new(HashMap::new());
}

static MAP: LazyLock<Mutex<HashMap<TypeId, CompId>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompId(u32);

impl CompId {
	#[inline]
	pub fn of<T: ?Sized + 'static>() -> Self {
		let id = TypeId::of::<T>();
		LOCAL_MAP.with(|m| *m.borrow_mut().entry(id).or_insert_with(|| Self::register(id)))
	}
	
	#[cold]
	#[inline(never)]
	fn register(id: TypeId) -> CompId {
		let mut map = MAP.lock().unwrap();
		let len = map.len();
		match map.entry(id) {
			Entry::Occupied(e) => *e.get(),
			Entry::Vacant(e) => *e.insert(CompId(len as u32)),
		}
	}
}

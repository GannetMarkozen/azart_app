use std::borrow::Cow;
use std::ops::Deref;
use bevy::prelude::*;

// A string that's constructed via dformat!(..) that only yields a value in debug_assertion builds.
#[derive(Debug, Clone, Reflect, Component)]
pub struct DebugString {
	#[cfg(debug_assertions)]
	string: Cow<'static, str>,
}

impl DebugString {
	// This should not be used. Use dbgfmt!(..).
	pub fn _new(string: Cow<'static, str>) -> Self {
		Self {
			#[cfg(debug_assertions)]
			string,
		}
	}

	pub fn as_str(&self) -> &str {
		#[cfg(debug_assertions)]
		return &self.string;

		#[cfg(not(debug_assertions))]
		return "NULL";
	}
}

impl From<&'static str> for DebugString {
	fn from(string: &'static str) -> Self {
		Self::_new(string.into())
	}
}

impl Default for DebugString {
	fn default() -> Self {
		Self {
			#[cfg(debug_assertions)]
			string: "uninit".into(),
		}
	}
}

impl Deref for DebugString {
	type Target = str;

	fn deref(&self) -> &Self::Target {
		self.as_str()
	}
}

impl std::fmt::Display for DebugString {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		self.as_str().fmt(f)
	}
}


// Only runs formatting logic in debug_assertion builds.
#[macro_export]
#[cfg(debug_assertions)]
macro_rules! dbgfmt {
	($($arg:tt)*) => {
		DebugString::_new(format!($($arg)*).into())
	}
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! dbgfmt {
	($($arg:tt)*) => {
		DebugString::default()
	}
}

pub use dbgfmt;
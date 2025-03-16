use std::borrow::Cow;
use bevy::prelude::*;
use ash::vk;

pub trait GpuResource {}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ShaderPath<'a> {
	path: &'a str,
}

impl<'a> ShaderPath<'a> {
	// Prefer shader_path!
	pub const fn new(path: &'a str) -> Self {
		Self { path }
	}

	pub const fn as_str(&'a self) -> &'a str {
		self.path
	}
}

#[macro_export]
macro_rules! shader_path {
	($path:expr) => {
		ShaderPath::new(concat!(env!("CARGO_MANIFEST_DIR"), "/spv/", $path, ".spv"))
	}
}

#[macro_export]
macro_rules! asset_path {
	($path:expr) => {
		concat!(env!("CARGO_MANIFEST_DIR"), "/assets/", $path)
	}
}

//pub use shader_path;

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Reflect, Resource)]
pub enum MsaaCount {
	#[default]
	Sample1,// No MSAA.
	Sample2,
	Sample4,
	Sample8,
}

impl MsaaCount {
	// If Msaa > 1.
	#[inline(always)]
	pub const fn enabled(&self) -> bool {
		!matches!(self, MsaaCount::Sample1)
	}

	#[inline(always)]
	pub const fn as_u32(&self) -> u32 {
		match self {
			MsaaCount::Sample1 => 1,
			MsaaCount::Sample2 => 2,
			MsaaCount::Sample4 => 4,
			MsaaCount::Sample8 => 8,
		}
	}
	
	#[inline(always)]
	pub const fn as_vk_sample_count(&self) -> vk::SampleCountFlags {
		match self {
			MsaaCount::Sample1 => vk::SampleCountFlags::TYPE_1,
			MsaaCount::Sample2 => vk::SampleCountFlags::TYPE_2,
			MsaaCount::Sample4 => vk::SampleCountFlags::TYPE_4,
			MsaaCount::Sample8 => vk::SampleCountFlags::TYPE_8,
		}
	}
}

impl Into<u32> for MsaaCount {
	#[inline(always)]
	fn into(self) -> u32 {
		self.as_u32()
	}
}

impl Into<vk::SampleCountFlags> for MsaaCount {
	#[inline(always)]
	fn into(self) -> vk::SampleCountFlags {
		self.as_vk_sample_count()
	}
}

impl From<vk::SampleCountFlags> for MsaaCount {
	fn from(x: vk::SampleCountFlags) -> Self {
		match x {
			vk::SampleCountFlags::TYPE_2 => MsaaCount::Sample2,
			vk::SampleCountFlags::TYPE_4 => MsaaCount::Sample4,
			vk::SampleCountFlags::TYPE_8 => MsaaCount::Sample8,
			_ => MsaaCount::Sample1,
		}
	}
}
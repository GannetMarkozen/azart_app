use bevy::prelude::*;
use ash::vk;

#[derive(Default, Copy, Clone, Eq, PartialEq, Hash, Debug, Reflect, Resource)]
pub enum MsaaCount {
	#[default]
	Sample1,
	Sample2,
	Sample4,
	Sample8,
	Sample16,
}

impl MsaaCount {
	#[inline(always)]
	pub const fn as_u32(&self) -> u32 {
		match self {
			MsaaCount::Sample1 => 1,
			MsaaCount::Sample2 => 2,
			MsaaCount::Sample4 => 4,
			MsaaCount::Sample8 => 8,
			MsaaCount::Sample16 => 16,
			_ => unreachable!(),
		}
	}
	
	#[inline(always)]
	pub const fn as_vk_sample_count(&self) -> vk::SampleCountFlags {
		match self {
			MsaaCount::Sample1 => vk::SampleCountFlags::TYPE_1,
			MsaaCount::Sample2 => vk::SampleCountFlags::TYPE_2,
			MsaaCount::Sample4 => vk::SampleCountFlags::TYPE_4,
			MsaaCount::Sample8 => vk::SampleCountFlags::TYPE_8,
			MsaaCount::Sample16 => vk::SampleCountFlags::TYPE_16,
			_ => unreachable!(),
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
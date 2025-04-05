use std::{env, mem};
use std::ffi::OsStr;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, OnceLock};
use ash::vk;
use bevy::prelude::*;

pub trait GpuResource {}
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Reflect)]
pub enum Format {
	Undefined = vk::Format::UNDEFINED.as_raw() as u8,
	RgU4Norm = 1,
	RgbaU4Norm = 2,
	BgraU4Norm = 3,
	R5G6B5U16Norm = 4,
	B5G6R5U16Norm = 5,
	R5G5B5A1U16Norm = 6,
	B5G5R5A1U16Norm = 7,
	A1R5G5B5U16Norm = 8,
	RU8Norm = 9,
	RI8Norm = 10,
	RU8Scaled = 11,
	RI8Scaled = 12,
	RU8 = 13,
	RI8 = 14,
	RU8Srgb = 15,
	RgU8Norm = 16,
	RgI8Norm = 17,
	RgU8Scaled = 18,
	RgI8Scaled = 19,
	RgU8 = 20,
	RgI8 = 21,
	RgU8Srgb = 22,
	RgbU8Norm = 23,
	RgbI8Norm = 24,
	RgbU8Scaled = 25,
	RgbI8Scaled = 26,
	RgbU8 = 27,
	RgbI8 = 28,
	RgbU8Srgb = 29,
	BgrU8Norm = 30,
	BgrI8Norm = 31,
	BgrU8Scaled = 32,
	BgrI8Scaled = 33,
	BgrU8 = 34,
	BgrI8 = 35,
	BgrU8Srgb = 36,
	RgbaU8Norm = 37,
	RgbaI8Norm = 38,
	RgbaU8Scaled = 39,
	RgbaI8Scaled = 40,
	RgbaU8 = 41,
	RgbaI8 = 42,
	RgbaU8Srgb = 43,
	BgraU8Norm = 44,
	BgraI8Norm = 45,
	BgraU8Scaled = 46,
	BgraI8Scaled = 47,
	BgraU8 = 48,
	BgraI8 = 49,
	BgraU8Srgb = 50,
	AbgrU8Norm = 51,
	AbgrI8Norm = 52,
	AbgrU8Scaled = 53,
	AbgrI8Scaled = 54,
	AbgrU8 = 55,
	AbgrI8 = 56,
	AbgrU8Srgb = 57,
	A2R10G10B10U32Norm = 58,
	A2R10G10B10I32Norm = 59,
	A2R10G10B10U32Scaled = 60,
	A2R10G10B10I32Scaled = 61,
	A2R10G10B10U32 = 62,
	A2R10G10B10I32 = 63,
	A2B10G10R10U32Norm = 64,
	A2B10G10R10I32Norm = 65,
	A2B10G10R10U32Scaled = 66,
	A2B10G10R10I32Scaled = 67,
	A2B10G10R10U32 = 68,
	A2B10G10R10I32 = 69,
	RU16Norm = 70,
	RI16Norm = 71,
	RU16Scaled = 72,
	RI16Scaled = 73,
	RU16 = 74,
	RI16 = 75,
	RF16 = 76,
	RgU16Norm = 77,
	RgI16Norm = 78,
	RgU16Scaled = 79,
	RgI16Scaled = 80,
	RgU16 = 81,
	RgI16 = 82,
	RgF16 = 83,
	RgbU16Norm = 84,
	RgbI16Norm = 85,
	RgbU16Scaled = 86,
	RgbI16Scaled = 87,
	RgbU16 = 88,
	RgbI16 = 89,
	RgbF16 = 90,
	RgbaU16Norm = 91,
	RgbaI16Norm = 92,
	RgbaU16Scaled = 93,
	RgbaI16Scaled = 94,
	RgbaU16 = 95,
	RgbaI16 = 96,
	RgbaF16 = 97,
	RU32 = 98,
	RI32 = 99,
	RF32 = 100,
	RgU32 = 101,
	RgI32 = 102,
	RgF32 = 103,
	RgbU32 = 104,
	RgbI32 = 105,
	RgbF32 = 106,
	RgbaU32 = 107,
	RgbaI32 = 108,
	RgbaF32 = 109,
	RU64 = 110,
	RI64 = 111,
	RF64 = 112,
	RgU64 = 113,
	RgI64 = 114,
	RgF64 = 115,
	RgbU64 = 116,
	RgbI64 = 117,
	RgbF64 = 118,
	RgbaU64 = 119,
	RgbaI64 = 120,
	RgbaF64 = 121,
	B10G11R11F32 = 122,
	E5B9G9R9F32 = 123,
	D16U16Norm = 124,
	X8D24U32Norm = 125,
	DF32 = 126,
	SU8 = 127,
	D16U16NormS8U8 = 128,
	D24U24NormS8U8 = 129,
	Df32S8U8 = 130,
	Bc1RgbUNormBlock = 131,
	Bc1RgbU8SrgbBlock = 132,
	Bc1RgbaUNormBlock = 133,
	Bc1RgbaU8SrgbBlock = 134,
	Bc2UNormBlock = 135,
	Bc2U8SrgbBlock = 136,
	Bc3UNormBlock = 137,
	Bc3U8SrgbBlock = 138,
	Bc4UNormBlock = 139,
	Bc4INormBlock = 140,
	Bc5UNormBlock = 141,
	Bc5INormBlock = 142,
	Bc6hUF32Block = 143,
	Bc6hF32Block = 144,
	Bc7UNormBlock = 145,
	Bc7U8SrgbBlock = 146,
	Etc2RgbU8NormBlock = 147,
	Etc2RgbU8SrgbBlock = 148,
	Etc2RgbA1U8NormBlock = 149,
	Etc2RgbA1U8SrgbBlock = 150,
	Etc2RgbA8U8NormBlock = 151,
	Etc2RgbA8U8SrgbBlock = 152,
	EacR11UNormBlock = 153,
	EacR11INormBlock = 154,
	EacR11G11UNormBlock = 155,
	EacR11G11INormBlock = 156,
	Astc4x4U8NormBlock = 157,
	Astc4x4U8SrgbBlock = 158,
	Astc5x4U8NormBlock = 159,
	Astc5x4U8SrgbBlock = 160,
	Astc5x5U8NormBlock = 161,
	Astc5x5U8SrgbBlock = 162,
	Astc6x5U8NormBlock = 163,
	Astc6x5U8SrgbBlock = 164,
	Astc6x6U8NormBlock = 165,
	Astc6x6U8SrgbBlock = 166,
	Astc8x5U8NormBlock = 167,
	Astc8x5U8SrgbBlock = 168,
	Astc8x6U8NormBlock = 169,
	Astc8x6U8SrgbBlock = 170,
	Astc8x8U8NormBlock = 171,
	Astc8x8U8SrgbBlock = 172,
	Astc10x5U8NormBlock = 173,
	Astc10x5U8SrgbBlock = 174,
	Astc10x6U8NormBlock = 175,
	Astc10x6U8SrgbBlock = 176,
	Astc10x8U8NormBlock = 177,
	Astc10x8U8SrgbBlock = 178,
	Astc10x10U8NormBlock = 179,
	Astc10x10U8SrgbBlock = 180,
	Astc12x10U8NormBlock = 181,
	Astc12x10U8SrgbBlock = 182,
	Astc12x12U8NormBlock = 183,
	Astc12x12U8SrgbBlock = 184,
}

impl Format {
	pub const fn block_size(&self) -> usize {
		use Format::*;

		match *self {
			Undefined => 0,

			RgU4Norm | RU8Norm | RI8Norm | RU8Scaled | RI8Scaled | RU8 | RI8 | RU8Srgb | SU8 => 1,

			RgbaU4Norm | BgraU4Norm
				| R5G6B5U16Norm | B5G6R5U16Norm | R5G5B5A1U16Norm | B5G5R5A1U16Norm | A1R5G5B5U16Norm
				| RgU8Norm | RgI8Norm | RgU8Scaled | RgI8Scaled | RgU8 | RgI8 | RgU8Srgb
				| RU16Norm | RI16Norm | RU16Scaled | RI16Scaled | RU16 | RI16 | RF16
				| D16U16Norm => 2,

			RgbU8Norm | RgbI8Norm | RgbU8Scaled | RgbI8Scaled | RgbU8 | RgbI8 | RgbU8Srgb
				| BgrU8Norm | BgrI8Norm | BgrU8Scaled | BgrI8Scaled | BgrU8 | BgrI8 | BgrU8Srgb
				| D16U16NormS8U8 => 3,

			RgbaU8Norm | RgbaI8Norm | RgbaU8Scaled | RgbaI8Scaled | RgbaU8 | RgbaI8 | RgbaU8Srgb | BgraU8Norm | BgraI8Norm | BgraU8Scaled | BgraI8Scaled | BgraU8 | BgraI8 | BgraU8Srgb
				| AbgrU8Norm | AbgrI8Norm | AbgrU8Scaled | AbgrI8Scaled | AbgrU8 | AbgrI8 | AbgrU8Srgb | A2R10G10B10U32Norm | A2R10G10B10I32Norm
				| A2R10G10B10U32Scaled | A2R10G10B10I32Scaled | A2R10G10B10U32 | A2R10G10B10I32 | A2B10G10R10U32Norm | A2B10G10R10I32Norm | A2B10G10R10U32Scaled | A2B10G10R10I32Scaled | A2B10G10R10U32 | A2B10G10R10I32
				| RgU16Norm | RgI16Norm | RgU16Scaled | RgI16Scaled | RgU16 | RgI16 | RgF16
				| RU32 | RI32 | RF32
				| X8D24U32Norm | DF32 | D24U24NormS8U8
				| B10G11R11F32 | E5B9G9R9F32 => 4,

			Df32S8U8 => 5,

			RgbU16Norm | RgbI16Norm | RgbU16Scaled | RgbI16Scaled | RgbU16 | RgbI16 | RgbF16 => 6,

			RgbaU16Norm | RgbaI16Norm | RgbaU16Scaled | RgbaI16Scaled | RgbaF16
				| RgbaU16 | RgbaI16 | RgU32 | RgI32 | RgF32
				| RU64 | RI64 | RF64 => 8,

			RgbU32 | RgbI32 | RgbF32 => 12,

			RgbaU32 | RgbaI32 | RgbaF32 | RgU64 | RgI64 | RgF64 => 16,

			RgbU64 | RgbI64 | RgbF64 => 24,

			RgbaU64 | RgbaI64 | RgbaF64 => 32,

			Bc1RgbUNormBlock | Bc1RgbU8SrgbBlock | Bc1RgbaUNormBlock | Bc1RgbaU8SrgbBlock | Bc4UNormBlock | Bc4INormBlock
				| Etc2RgbU8NormBlock | Etc2RgbU8SrgbBlock | Etc2RgbA1U8NormBlock | Etc2RgbA1U8SrgbBlock
				| EacR11UNormBlock | EacR11INormBlock => 8,

			Bc2UNormBlock | Bc2U8SrgbBlock | Bc3UNormBlock | Bc3U8SrgbBlock | Bc5UNormBlock | Bc5INormBlock | Bc6hUF32Block | Bc6hF32Block | Bc7UNormBlock | Bc7U8SrgbBlock
				| Etc2RgbA8U8NormBlock | Etc2RgbA8U8SrgbBlock | EacR11G11UNormBlock | EacR11G11INormBlock
				| Astc4x4U8NormBlock | Astc4x4U8SrgbBlock | Astc5x4U8NormBlock | Astc5x4U8SrgbBlock | Astc5x5U8NormBlock | Astc5x5U8SrgbBlock | Astc6x5U8NormBlock | Astc6x5U8SrgbBlock | Astc6x6U8NormBlock | Astc6x6U8SrgbBlock | Astc8x5U8NormBlock | Astc8x5U8SrgbBlock
				| Astc8x6U8NormBlock | Astc8x6U8SrgbBlock | Astc8x8U8NormBlock | Astc8x8U8SrgbBlock | Astc10x5U8NormBlock | Astc10x5U8SrgbBlock | Astc10x6U8NormBlock | Astc10x6U8SrgbBlock | Astc10x8U8NormBlock | Astc10x8U8SrgbBlock | Astc10x10U8NormBlock | Astc10x10U8SrgbBlock
				| Astc12x10U8NormBlock | Astc12x10U8SrgbBlock | Astc12x12U8NormBlock | Astc12x12U8SrgbBlock => 16,
		}
	}

	pub const fn block_dim(&self) -> UVec2 {
		use Format::*;

		match *self {
			Undefined => UVec2::ZERO,

			Bc1RgbUNormBlock | Bc1RgbU8SrgbBlock | Bc1RgbaUNormBlock | Bc1RgbaU8SrgbBlock | Bc2UNormBlock | Bc2U8SrgbBlock | Bc3UNormBlock | Bc3U8SrgbBlock | Bc4UNormBlock | Bc4INormBlock
				|Bc5UNormBlock | Bc5INormBlock | Bc6hUF32Block | Bc6hF32Block | Bc7UNormBlock | Bc7U8SrgbBlock | Etc2RgbU8NormBlock | Etc2RgbU8SrgbBlock | Etc2RgbA1U8NormBlock | Etc2RgbA1U8SrgbBlock
				| Etc2RgbA8U8NormBlock | Etc2RgbA8U8SrgbBlock | EacR11UNormBlock | EacR11INormBlock | EacR11G11UNormBlock | EacR11G11INormBlock | Astc4x4U8NormBlock | Astc4x4U8SrgbBlock => UVec2::new(4, 4),

			Astc5x4U8NormBlock | Astc5x4U8SrgbBlock => UVec2::new(5, 4),

			Astc5x5U8NormBlock | Astc5x5U8SrgbBlock => UVec2::new(5, 5),

			Astc6x5U8NormBlock | Astc6x5U8SrgbBlock => UVec2::new(6, 5),

			Astc6x6U8NormBlock | Astc6x6U8SrgbBlock => UVec2::new(6, 6),

			Astc8x5U8NormBlock | Astc8x5U8SrgbBlock => UVec2::new(8, 5),

			Astc8x6U8NormBlock | Astc8x6U8SrgbBlock => UVec2::new(8, 6),

			Astc8x8U8NormBlock | Astc8x8U8SrgbBlock => UVec2::new(8, 8),

			Astc10x5U8NormBlock | Astc10x5U8SrgbBlock => UVec2::new(10, 5),

			Astc10x6U8NormBlock | Astc10x6U8SrgbBlock => UVec2::new(10, 6),

			Astc10x8U8NormBlock | Astc10x8U8SrgbBlock => UVec2::new(10, 8),

			Astc10x10U8NormBlock | Astc10x10U8SrgbBlock => UVec2::new(10, 10),

			Astc12x10U8NormBlock | Astc12x10U8SrgbBlock => UVec2::new(12, 10),

			Astc12x12U8NormBlock | Astc12x12U8SrgbBlock => UVec2::new(12, 12),

			_ => UVec2::ONE,
		}
	}
}

impl From<vk::Format> for Format {
	#[inline(always)]
	fn from(x: vk::Format) -> Self {
		unsafe { mem::transmute(x.as_raw() as u8) }
	}
}

impl From<Format> for vk::Format {
	#[inline(always)]
	fn from(x: Format) -> Self {
		vk::Format::from_raw(x as i32)
	}
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct ShaderPath(pub PathBuf);

impl Deref for ShaderPath {
	type Target = PathBuf;

	#[inline(always)]
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl From<PathBuf> for ShaderPath {
	#[inline(always)]
	fn from(x: PathBuf) -> Self {
		ShaderPath(x)
	}
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct AssetPath(pub PathBuf);

impl Deref for AssetPath {
	type Target = PathBuf;

	#[inline(always)]
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl From<PathBuf> for AssetPath {
	#[inline(always)]
	fn from(x: PathBuf) -> Self {
		AssetPath(x)
	}
}

#[must_use]
pub fn shader_path(path: impl AsRef<Path>) -> ShaderPath {
	let path = path.as_ref();

	let asset_path_root = match Path::new("assets/").exists() {
		true => Path::new("assets/"),
		false => Path::new("../../assets/"),
	};

	let mut shader_path = asset_path_root.join("spv/").join(path);
	match path.extension() {
		Some(ext) => _ = shader_path.set_extension(format!("{}.ron", ext.to_str().unwrap())),
		None => _ = shader_path.set_extension("ron"),
	}

	ShaderPath(shader_path)
}

#[must_use]
pub fn asset_path(path: impl AsRef<Path>) -> AssetPath {
	let asset_path_root = match Path::new("assets/").exists() {
		true => Path::new("assets/"),
		false => Path::new("../../assets/"),
	};

	let asset_path = asset_path_root.join(path.as_ref());

	AssetPath(asset_path)
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Reflect, Resource)]
pub enum MsaaCount {
	#[default]
	None,// No MSAA.
	Sample2,
	Sample4,
	Sample8,
}

impl MsaaCount {
	// If Msaa > 1.
	#[inline(always)]
	pub const fn enabled(&self) -> bool {
		!matches!(self, MsaaCount::None)
	}

	#[inline(always)]
	pub const fn as_u32(&self) -> u32 {
		match self {
			MsaaCount::None => 1,
			MsaaCount::Sample2 => 2,
			MsaaCount::Sample4 => 4,
			MsaaCount::Sample8 => 8,
		}
	}
	
	#[inline(always)]
	pub const fn as_vk_sample_count(&self) -> vk::SampleCountFlags {
		match self {
			MsaaCount::None => vk::SampleCountFlags::TYPE_1,
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
			_ => MsaaCount::None,
		}
	}
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Reflect, Resource)]
pub enum CullMode {
	#[default]
	None,
	Front,
	Back,
}

#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Reflect, Resource)]
pub enum TriangleFillMode {
	#[default]
	Fill,
	Wireframe,
}
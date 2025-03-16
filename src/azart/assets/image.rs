use std::io::Cursor;
use std::sync::Arc;
use bevy::prelude::*;
use crate::gfx;
use image::ImageReader;
use ash::vk;

#[derive(Asset, TypePath)]
pub struct Image {
	pub resource: Arc<gfx::Image>,
	pub data: Option<Box<[u8]>>,
}

impl Image {
	pub fn from_bytes(bytes: &[u8]) -> Image {
		let image = ImageReader::new(Cursor::new(bytes))
			.with_guessed_format()
			.unwrap()
			.decode()
			.unwrap();
		
		use image::DynamicImage::*;
		use vk::Format;
		
		let format = match &image {
			ImageLuma8(_) | ImageLumaA8(_) => Format::R8_UINT,
			ImageRgb8(_) => Format::R8G8B8_UNORM,
			ImageRgba8(_) => Format::R8G8B8A8_UNORM,
			ImageLuma16(_) | ImageLumaA16(_) => Format::R16_UINT,
			ImageRgb16(_) => Format::R16G16B16_UINT,
			ImageRgba16(_) => Format::R16G16B16A16_UINT,
			ImageRgb32F(_) => Format::R32G32B32_SFLOAT,
			ImageRgba32F(_) => Format::R32G32B32A32_SFLOAT,
			_ => unreachable!("Unimplemented format!")
		};
		
		todo!();
	}
}
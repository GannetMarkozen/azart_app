use std::borrow::Cow;
use std::io::Error;
use std::path::{Path, PathBuf};
use anyhow::{anyhow, Context};
use clap::Parser;
use walkdir::WalkDir;
use azart::prelude::{asset::*, gfx::pbr::mesh::*};
use azart::prelude::bevy::math::{UVec2, Vec3};
use azart::prelude::gfx::{Image, SerdeImage};
use azart::prelude::gfx::gfx_utils::Format;
use azart::prelude::asset::SerdeAsset;
use image::GenericImageView;
use serde::{Serialize, Deserialize};

const SRC_DIR: &str = "..\\..\\src_assets";
const DST_DIR: &str = "..\\..\\assets";

#[derive(Parser)]
struct Cli {
	#[arg(short, long)]
	src: PathBuf,
	#[arg(short, long)]
	dst: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
	let Cli { src, dst } = Cli::parse();

	let dst = match dst.as_ref() {
		Some(dst) => Path::new(DST_DIR).join(dst),
		None => Path::new(DST_DIR).join(&src),
	};
	let src = Path::new(SRC_DIR).join(src);

	println!("Attempting to compile {src:?} to {dst:?}");

	let (document, buffers, images) = gltf::import(&src)?;

	let mut image_assets = Vec::new();

	let mut unnamed_image_count = 0;
	for texture in document.textures() {
		let (data, mime_type) = match texture.source().source() {
			gltf::image::Source::Uri { uri, mime_type } => {
				let path = src.parent().unwrap().join(uri);
				(Cow::Owned(std::fs::read(&path)?), mime_type)
			},
			gltf::image::Source::View { view, mime_type } => {
				let buffer = &buffers[view.buffer().index()];
				let start = view.offset();
				let end = start + view.length();
				(Cow::Borrowed(&buffer[start..end]), Some(mime_type))
			},
		};

		let image = image::load_from_memory(&data)?;
		let (width, height) = image.dimensions();

		let (format, data) = match &image {
			image::DynamicImage::ImageLuma8(image) => (Format::RU8, Cow::Borrowed(image.as_raw().as_slice())),
			image::DynamicImage::ImageRgba8(image) => (Format::RgbaU8, Cow::Borrowed(image.as_raw().as_slice())),
			// RgbU8 is usually an unsupported format so just always use RgbaU8 with an extraneous alpha channel.
			image::DynamicImage::ImageRgb8(image) => (Format::RgbaU8, Cow::Owned({
				image
					.pixels()
					.flat_map(|&image::Rgb([r, g, b])| [r, g, b, u8::MAX])
					.collect::<Vec<_>>()
			})),
			image::DynamicImage::ImageRgba32F(image) => (Format::A2R10G10B10U32, Cow::Owned({
				image
					.pixels()
					.flat_map(|&image::Rgba([r, g, b, a])| {
						const U10_MAX: f32 = (u32::MAX >> (32 - 10)) as f32;
						const U2_MAX: f32 = (u32::MAX >> (32 - 2)) as f32;

						let r = (r.clamp(0.0, 1.0) * U10_MAX).round() as u32;
						let g  = (g.clamp(0.0, 1.0) * U10_MAX).round() as u32;
						let b = (b.clamp(0.0, 1.0) * U10_MAX).round() as u32;
						let a = (a.clamp(0.0, 1.0) * U2_MAX).round() as u32;

						(a | r << 2 | g << 12 | b << 22).to_ne_bytes()
					})
					.collect::<Vec<_>>()
			})),
			image => return Err(anyhow!("Unhandled image format {image:?}")),
		};

		let name = match texture.name() {
			Some(name) => Cow::Borrowed(name),
			None => match texture.source().source() {
				gltf::image::Source::Uri { uri, .. } => Cow::Borrowed(Path::new(uri)
					.file_stem()
					.and_then(|s| s.to_str())
					.unwrap_or(uri)
				),
				gltf::image::Source::View { view, .. } => match view.name() {
					Some(name) => Cow::Borrowed(name),
					None => Cow::Owned({
						let i = unnamed_image_count;
						unnamed_image_count += 1;
						format!("unnamed_image[{i}]")
					}),
				},
			},
		};

		let image = SerdeImage {
			name: &name,
			resolution: UVec2::new(width, height),
			format,
			// @TODO: Create mips here.
			mips: vec![&data],
		};

		let dst = dst.join(&*name).with_extension("image");
		let bytes = bincode::serde::encode_to_vec(&image, bincode::config::standard())?;
		store(&dst, std::any::type_name::<Image>(), &bytes)?;

		println!("Created image asset {name} at {dst:?}");

		image_assets.push(dst.strip_prefix("../..")?.to_owned());
	}

	let mut materials = Vec::new();
	let mut unnamed_material_count = 0;
	for material in document.materials() {
		#[derive(Serialize, Deserialize)]
		#[serde(rename = "Material")]
		#[serde(bound(deserialize = "'de: 'a"))]
		struct SerdeMaterial<'a> {
			base_color: SerdeAsset<'a>,
			normals: Option<SerdeAsset<'a>>,
			metallic_roughness: Option<SerdeAsset<'a>>,
		}

		let name = match material.name() {
			Some(name) => Cow::Borrowed(name),
			None => Cow::Owned({
				let i = unnamed_material_count;
				unnamed_material_count += 1;
				format!("unnamed_material[{i}]")
			}),
		};

		let material = SerdeMaterial {
			base_color: SerdeAsset::new(&image_assets[material.pbr_metallic_roughness().base_color_texture().context("No base_color!")?.texture().index()]),
			normals: material.normal_texture().map(|texture| SerdeAsset::new(&image_assets[texture.texture().index()])),
			metallic_roughness: material.pbr_metallic_roughness().metallic_roughness_texture().map(|texture| SerdeAsset::new(&image_assets[texture.texture().index()])),
		};

		let dst = dst.join(&*name).with_extension("mat");
		let bytes = serde_yaml::to_string(&material).map_err(Error::other)?.into_bytes();

		println!("Created material {name} at {dst:?}");

		store(&dst, std::any::type_name::<Material>(), &bytes)?;

		materials.push(dst.strip_prefix("../..")?.to_owned());
	}

	for mesh in document.meshes() {
		let name = mesh.name().context("No mesh name!")?;
		for mesh in mesh.primitives() {
			#[derive(Serialize, Deserialize)]
			#[serde(bound(deserialize = "'de: 'a"))]
			struct SerdeMesh<'a> {
				material: SerdeAsset<'a>,
				indices: Vec<u32>,
				positions: Vec<Vec3>,
				attributes: Vec<VertexAttributes>,
			}

			let reader = mesh.reader(|buffer| Some(&buffers[buffer.index()]));

			let indices = reader
				.read_indices()
				.context("Failed to read indices!")?
				.into_u32()
				.collect::<Vec<_>>();

			let positions = reader
				.read_positions()
				.context("Failed to read positions!")?
				.map(|[x, y, z]| Vec3::new(x, y, z))
				.collect::<Vec<_>>();

			let attributes = reader
				.read_tex_coords(0)
				.context("Failed to read tex coords!")?
				.into_f32()
				.zip(reader
					.read_normals()
					.context("Failed to read normals!")?
				)
				.map(|(uv, normal)| VertexAttributes { uv, normal })
				.collect::<Vec<_>>();

			let material = SerdeAsset::new(&materials[mesh.material().index().context("No material index!")?]);

			let mesh = SerdeMesh {
				material,
				indices,
				positions,
				attributes,
			};

			let dst = dst.join(&*name).with_extension("mesh");

			let bytes = bincode::serde::encode_to_vec(&mesh, bincode::config::standard())?;

			store(&dst,  std::any::type_name::<Mesh>(), &bytes)?;

			println!("Created mesh {name} at {dst:?}");
		}
	}

	Ok(())
}

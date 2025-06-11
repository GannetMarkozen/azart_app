use std::path::Path;
use azart_utils::io;
use bevy::math::*;
use bevy::prelude::*;
use either::{for_both, Either};
use image::GenericImageView;

#[derive(Debug)]
pub struct Scene {
	pub objects: Vec<Object>,
	pub meshes: Vec<Mesh>,
	pub textures: Vec<Texture>,
}

impl Scene {
	pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
		let path = path.as_ref();
		assert_eq!(path.extension().expect("No extension!").to_str().unwrap(), "gltf");

		// @NOTE: This won't work on Android.
		let (document, buffers, images) = gltf::import(path).expect("Failed to import slice!");

		let mut textures = Vec::new();
		let mut meshes = Vec::new();
		let mut objects = Vec::new();

		for texture in document.textures() {
			let (data, mime_type) = match texture.source().source() {
				gltf::image::Source::Uri { uri, mime_type } => {
					let path = path.parent().unwrap().join(uri);
					(Either::Left(io::read(path).expect("Failed to read texture!")), mime_type)
				},
				gltf::image::Source::View { view, mime_type } => {
					let buffer = &buffers[view.buffer().index()];
					let start = view.offset();
					let end = start + view.length();
					(Either::Right(&buffer[start..end]), Some(mime_type))
				},
			};

			let data = match &data {
				Either::Left(data) => data.as_slice(),
				Either::Right(data) => data,
			};

			let image = image::load_from_memory(data).expect("Failed to load texture");
			textures.push(Texture(image));
		}

		for mesh in document.meshes() {
			let Some(mesh) = mesh.primitives().next() else {
				continue;
			};

			let reader = mesh.reader(|buffer| Some(&buffers[buffer.index()]));

			let indices = reader
				.read_indices()
				.map(|indices| match indices {
					gltf::mesh::util::ReadIndices::U8(indices) => Indices::U8(indices.into_iter().collect()),
					gltf::mesh::util::ReadIndices::U16(indices) => Indices::U16(indices.into_iter().collect()),
					gltf::mesh::util::ReadIndices::U32(indices) => Indices::U32(indices.into_iter().collect()),
				})
				.expect("No indices!");

			let positions = reader
				.read_positions()
				.map(|positions| positions.into_iter().collect())
				.expect("No positions!");

			let uvs = reader
				.read_tex_coords(0)
				.map(|uvs| match uvs {
					gltf::mesh::util::ReadTexCoords::U8(uvs) => uvs.into_iter().map(|[u, v]| [u as f32 / u8::MAX as f32, v as f32 / u8::MAX as f32]).collect(),
					gltf::mesh::util::ReadTexCoords::U16(uvs) => uvs.into_iter().map(|[u, v]| [u as f32 / u16::MAX as f32, v as f32 / u16::MAX as f32]).collect(),
					gltf::mesh::util::ReadTexCoords::F32(uvs) => uvs.into_iter().collect(),
				});

			let normals = reader
				.read_normals()
				.map(|normals| normals.into_iter().collect());

			let material = mesh.material();

			let base_color_texture = material
				.pbr_metallic_roughness()
				.base_color_texture()
				.map(|texture| texture.texture().index());

			let normal_texture = material
				.normal_texture()
				.map(|texture| texture.texture().index());

			meshes.push(Mesh {
				indices,
				positions,
				uvs,
				normals,
				base_color_texture,
				normal_texture,
			});
		}

		for scene in document.scenes() {
			for node in scene.nodes() {
				let Some(mesh) = node.mesh() else {
					continue;
				};

				let transform = match node.transform() {
					gltf::scene::Transform::Matrix { matrix} => Transform::from_matrix(Mat4::from_cols_array_2d(&matrix).transpose()),
					gltf::scene::Transform::Decomposed { translation, rotation, scale } => Transform { translation: Vec3::from_array(translation), rotation: Quat::from_array(rotation), scale: Vec3::from_array(scale) },
				};

				objects.push(Object {
					transform,
					mesh: mesh.index(),
				});
			}
		}

		Ok(Self {
			objects,
			meshes,
			textures,
		})
	}
}

#[derive(Debug)]
pub struct Object {
	pub transform: Transform,
	pub mesh: usize,
}

#[derive(Debug)]
pub struct Mesh {
	pub indices: Indices,
	pub positions: Vec<[f32; 3]>,
	pub uvs: Option<Vec<[f32; 2]>>,
	pub normals: Option<Vec<[f32; 3]>>,
	pub base_color_texture: Option<usize>,
	pub normal_texture: Option<usize>,
}

#[derive(Debug)]
pub enum Indices {
	U8(Vec<u8>),
	U16(Vec<u16>),
	U32(Vec<u32>),
}

#[derive(Debug, Deref, DerefMut)]
pub struct Texture(pub image::DynamicImage);


/*use std::path::Path;
use azart_utils::io;
use bevy::math::{Vec2, Vec3};
use bevy::utils::{Entry, HashMap};
use either::{for_both, Either};
use image::GenericImageView;

pub struct Scene {
	pub meshes: HashMap<usize, Mesh>,
	pub textures: HashMap<usize, Texture>,
}

impl Scene {
	pub fn load(path: impl AsRef<Path>) -> Self {
		let path = path.as_ref();

		let gltf = {
			let data = io::read(path.as_ref()).unwrap_or_else(|e| panic!("Failed to read file {path:?}!: {e}"));
			match path.extension().expect("No extension!").to_str().unwrap() {
				"gltf" => Either::Left(gltf::Gltf::from_slice(&data).expect("Failed to load gltf!")),
				"glb" => Either::Right(gltf::Glb::from_slice(&data).expect("Failed to load glb!")),
				ext => panic!("Unsupported extension: {ext}"),
			}
		};

		let (document, buffer) = match &gltf {
			Either::Left(gltf) => (gltf.document.clone(), gltf.blob.as_ref().unwrap().as_slice()),
			Either::Right(glb) => {
				let json = gltf::json::Root::from_slice(&glb.json).unwrap();
				(gltf::Document::from_json_without_validation(json), glb.bin.as_ref().unwrap().iter().as_slice())
			},
		};

		let mut meshes = HashMap::<usize, Mesh>::new();
		let mut textures = HashMap::<usize, Texture>::new();

		for texture in document.textures() {
			let entry = match textures.entry(texture.index()) {
				Entry::Occupied(_) => continue,
				Entry::Vacant(entry) => entry,
			};

			let gltf::image::Source::View { view, mime_type } = texture.source().source() else {
				panic!("Unsupported texture source!");
			};

			let start = view.offset();
			let end = start + view.length();
			let data = &buffer[start..end];

			let data = match mime_type {
				"image/ktx2" => panic!("KTX2 textures are not supported!"),
				_ => data.to_vec(),
			};

			let image = image::load_from_memory(&data).expect("Failed to load texture!");
			let data = image
				.pixels()
				.flat_map(|(_, _, image::Rgba(rgba))| rgba)
				.collect::<Vec<_>>();

			entry.insert(Texture {
				resolution: [image.width(), image.height()],
				data,
			});
		}

		for mesh in document.meshes() {
			let entry = match meshes.entry(mesh.index()) {
				Entry::Occupied(_) => continue,
				Entry::Vacant(entry) => entry,
			};

			let primitives = mesh
				.primitives()
				.map(|primitive| {
					fn collect<T>(primitive: &gltf::Primitive, buffer: &[u8], semantic: &gltf::Semantic) -> Option<Vec<T>> {
						primitive
							.get(semantic)
							.and_then(|accessor| gltf::accessor::util::Iter::<T>::new(accessor, &buffer)
								.unwrap()
								.collect::<Vec<_>>()
							)
					}

					let get_buffer = |view: &gltf::buffer::Buffer| Some(&buffer[view.])
					let indices = primitive
						.indices()
						.map(|accessor| match accessor.data_type() {
							gltf::accessor::DataType::U8 => gltf::accessor::util::Iter::<u8>::new(accessor, |buffer| Some(&buffer.))
								.unwrap()
								.map(|i| i as u32)
								.collect::<Vec<_>>(),
							gltf::accessor::DataType::U16 => gltf::accessor::util::Iter::<u16>::new(accessor, accessor.view().unwrap().buffer())
								.unwrap()
								.map(|i| i as u32)
								.collect::<Vec<_>>(),
							gltf::accessor::DataType::U32 => gltf::accessor::util::Iter::<u32>::new(accessor, accessor.view().unwrap().buffer())
								.unwrap()
								.collect::<Vec<_>>(),
							data_type => panic!("Unsupported data type: {data_type:?}"),
						})
						.expect("No indices!");


					let positions = collect::<Vec3>(&primitive, &buffer, &gltf::Semantic::Positions).expect("No position buffer!");
					let uvs = collect::<Vec2>(&primitive, &buffer, &gltf::Semantic::TexCoords(0)).expect("No uvs!");
					let normals = collect::<Vec3>(&primitive, &buffer, &gltf::Semantic::Normals).expect("No normals!");

					let (base_color, normal) = {
						let material = primitive.material();

						(
							material.pbr_metallic_roughness().base_color_texture().expect("No base color texture!").texture(),
							material.normal_texture().expect("No normal texture!").texture(),
						)
					};

					assert!(textures.contains_key(&base_color.index()));
					assert!(textures.contains_key(&normal.index()));

					Primitive {
						base_color_texture: base_color.index(),
						normal_texture: normal.index(),
						indices,
						positions,
						uvs,
						normals,
					}
				})
				.collect::<Vec<_>>();

			entry.insert(Mesh {
				primitives,
			});
		}

		Self {
			meshes,
			textures,
		}
	}
}

pub struct Mesh {
	pub primitives: Vec<Primitive>,
}

pub struct Primitive {
	pub base_color_texture: usize,
	pub normal_texture: usize,
	pub indices: Vec<u32>,
	pub positions: Vec<Vec3>,
	pub uvs: Vec<Vec2>,
	pub normals: Vec<Vec3>,
}

pub struct Texture {
	pub resolution: [u32; 2],
	pub data: Vec<u8>,
}*/

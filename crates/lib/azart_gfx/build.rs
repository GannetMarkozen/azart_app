use std::env;
use std::io::Write;
use std::path::Path;
use azart_gfx_utils::Format;
use azart_gfx_utils::spirv::*;
use bevy::prelude::PartialReflect;
use spirv_reflect::types::{ReflectBuiltIn, ReflectDecorationFlags, ReflectDescriptorType, ReflectFormat, ReflectNumericTraitsScalar, ReflectStorageClass, ReflectTypeDescription, ReflectTypeFlags};
use spirv_headers::{BuiltIn, Op};
use bevy::utils::HashMap;
use bevy::reflect::{ReflectSerialize, TypeRegistry};
use bevy::reflect::serde::{ReflectSerializer, TypedReflectSerializer};
use bevy::tasks::futures_lite::io::BufWriter;
use serde::Serialize;
use walkdir::WalkDir;

const SHADER_DIR: &str = "..\\..\\..\\shaders";
const SPV_DIR: &str = "..\\..\\..\\assets\\spv";
const ENTRY_POINT: &str = "main";

fn main() {
	color_eyre::install().unwrap();

	assert!(Path::new(SHADER_DIR).exists(), "shader directory \"{SHADER_DIR}\" does not exist!");
	println!("cargo:rerun-if-changed={}", SHADER_DIR);

	let release_build = matches!(env::var("PROFILE"), Ok(profile) if profile == "release");

	let compiler = shaderc::Compiler::new().unwrap();

	let mut compile_errs = vec![];
	for entry in WalkDir::new(SHADER_DIR).into_iter() {
		let Ok(entry) = entry else {
			continue;
		};

		let path = entry.path();
		let Some(ext) = path.extension() else {
			continue;
		};

		let Some(ext) = ext.to_str() else {
			continue;
		};

		let stage = match ext {
			"vert" => ShaderStage::Vertex,
			"frag" => ShaderStage::Fragment,
			"comp" => ShaderStage::Compute,
			"geom" => ShaderStage::Geometry,
			"tese" => ShaderStage::TessControl,
			"tesc" => ShaderStage::TessEval,
			"mesh" => ShaderStage::Mesh,
			"task" => ShaderStage::Task,
			"raygen" => ShaderStage::RayGen,
			"anyhit" => ShaderStage::AnyHit,
			_ => panic!("Unsupported shader extension \"{ext}\" for file {entry:?}!"),
		};

		let file_name = entry
			.file_name()
			.to_str()
			.unwrap();

		println!("Compiling {entry:?}...");

		let shaderc_stage = match stage {
			ShaderStage::Vertex => shaderc::ShaderKind::Vertex,
			ShaderStage::Fragment => shaderc::ShaderKind::Fragment,
			ShaderStage::Compute => shaderc::ShaderKind::Compute,
			ShaderStage::Geometry => shaderc::ShaderKind::Geometry,
			ShaderStage::TessControl => shaderc::ShaderKind::TessControl,
			ShaderStage::TessEval => shaderc::ShaderKind::TessEvaluation,
			ShaderStage::Mesh => shaderc::ShaderKind::Mesh,
			ShaderStage::Task => shaderc::ShaderKind::Task,
			ShaderStage::RayGen => shaderc::ShaderKind::RayGeneration,
			ShaderStage::AnyHit => shaderc::ShaderKind::AnyHit,
		};

		let mut compile_options = shaderc::CompileOptions::new().unwrap();
		compile_options.set_optimization_level(if release_build { shaderc::OptimizationLevel::Performance } else { shaderc::OptimizationLevel::Zero });
		compile_options.set_auto_bind_uniforms(true);
		compile_options.set_auto_map_locations(true);
		compile_options.set_forced_version_profile(450, shaderc::GlslProfile::Core);

		let result = compiler.compile_into_spirv(
			&std::fs::read_to_string(path).unwrap(),
			shaderc_stage,
			path.to_str().unwrap(),
			ENTRY_POINT,
			Some(&compile_options),
		);

		let result = match result {
			Ok(result) => result,
			Err(err) => {
				compile_errs.push((file_name.to_owned(), err));
				continue;
			},
		};

		let module = spirv_reflect::create_shader_module(result.as_binary_u8()).unwrap();
		/*let mut bindings = HashMap::new();
		let mut specialization_constants = HashMap::new();

		for binding in module.enumerate_descriptor_bindings(Some(ENTRY_POINT)).unwrap().into_iter() {
			let type_desc = binding.type_description.unwrap();
			if matches!(*type_desc.op, Op::SpecConstant | Op::SpecConstantComposite | Op::SpecConstantTrue | Op::SpecConstantFalse) {// Specialization constant.
				specialization_constants.insert(binding.name, SpecializationConstant {
					binding: binding.binding,
					specialization_constant_type: match binding.type_description.unwrap().type_flags {
						x if x.contains(ReflectTypeFlags::BOOL) => PrimType::Bool,
						x if x.contains(ReflectTypeFlags::INT) => match type_desc.traits.numeric.scalar {
							ReflectNumericTraitsScalar { width: 32, signedness: 0 } => PrimType::U32,
						}
					},
				});
			} else {// Binding.

			}
		}*/

		for binding in module.enumerate_descriptor_bindings(Some(ENTRY_POINT)).unwrap().into_iter() {
			if !matches!(binding.descriptor_type, ReflectDescriptorType::UniformBuffer | ReflectDescriptorType::UniformBufferDynamic | ReflectDescriptorType::StorageBuffer) {
				continue;
			}

			let mut out = "
				#[repr(c)]\n
				#[derive(Clone, Debug, Eq, PartialEq, Hash, bytemuck::Pod, bytemuck::Zeroable)]\n
				{\n
				\t"
				.to_owned();
		}

		let bindings = module
			.enumerate_descriptor_bindings(Some(ENTRY_POINT))
			.unwrap()
			.into_iter()
			.map(|binding| {
				let descriptor_binding = DescriptorBinding {
					set: binding.set,
					binding: binding.binding,
					descriptor_type: match binding.descriptor_type {
						ReflectDescriptorType::Sampler => DescriptorType::Sampler,
						ReflectDescriptorType::CombinedImageSampler => DescriptorType::CombinedImageSampler,
						ReflectDescriptorType::SampledImage => DescriptorType::SampledImage,
						ReflectDescriptorType::StorageImage => DescriptorType::StorageImage,
						ReflectDescriptorType::UniformTexelBuffer => DescriptorType::UniformTexelBuffer,
						ReflectDescriptorType::StorageTexelBuffer => DescriptorType::StorageTexelBuffer,
						ReflectDescriptorType::UniformBuffer => DescriptorType::UniformBuffer,
						ReflectDescriptorType::StorageBuffer => DescriptorType::StorageBuffer,
						ReflectDescriptorType::UniformBufferDynamic => DescriptorType::UniformBufferDynamic,
						ReflectDescriptorType::InputAttachment => DescriptorType::InputAttachment,
						ReflectDescriptorType::AccelerationStructureNV => DescriptorType::AccelerationStructureNV,
						_ => panic!("Unsupported descriptor type {:?}!", binding.descriptor_type),
					},
					container_type: match (binding.type_description.as_ref().map(|desc| *desc.op), binding.array.dims.as_slice()) {
						(Some(Op::TypeArray), &[x]) => ContainerType::Array(x),
						(Some(Op::TypeArray), &[x, y]) => ContainerType::Array2D([x, y]),
						(Some(Op::TypeArray), &[x, y, z]) => ContainerType::Array3D([x, y, z]),
						(Some(Op::TypeRuntimeArray), _) => ContainerType::RuntimeArray,
						(Some(Op::TypeArray), dims) => panic!("Unsupported array dimensionality {dims:?}!"),
						(None, _) => panic!("Unsupported reflected binding {binding:?}!"),
						_ => ContainerType::Single,
					},
				};

				(binding.name, descriptor_binding)
			})
			.collect::<HashMap<_, _>>();

		let vertex_attributes = module
			.enumerate_input_variables(Some(ENTRY_POINT))
			.unwrap()
			.into_iter()
			.filter(|input| !input.decoration_flags.contains(ReflectDecorationFlags::BUILT_IN))
			.map(|input| {
				let attribute = VertexAttribute {
					location: input.location,
					format: match input.format {
						ReflectFormat::Undefined => Format::Undefined,
						ReflectFormat::R32_UINT => Format::RU32,
						ReflectFormat::R32_SINT => Format::RI32,
						ReflectFormat::R32_SFLOAT => Format::RF32,
						ReflectFormat::R32G32_UINT => Format::RgU32,
						ReflectFormat::R32G32_SINT => Format::RgI32,
						ReflectFormat::R32G32_SFLOAT => Format::RgF32,
						ReflectFormat::R32G32B32_UINT => Format::RgbU32,
						ReflectFormat::R32G32B32_SINT => Format::RgbI32,
						ReflectFormat::R32G32B32_SFLOAT => Format::RgbF32,
						ReflectFormat::R32G32B32A32_UINT => Format::RgbaU32,
						ReflectFormat::R32G32B32A32_SINT => Format::RgbaI32,
						ReflectFormat::R32G32B32A32_SFLOAT => Format::RgbaF32,
					},
				};

				(input.name, attribute)
			})
			.collect::<HashMap<_, _>>();

		let spirv = Spirv {
			code: result.as_binary().to_vec(),
			stage,
			bindings,
			vertex_attributes,
			push_constants: HashMap::new(),
			specialization_constants: HashMap::new(),
		};

		//let path = SPV_DIR.to_owned() + path + ".spv";
		let path = Path::new(SPV_DIR)
			.join(file_name)
			.with_extension(file_name
				.rfind('.')
				.map(|ext| format!("{}.spv", &file_name[(ext + 1)..]))
				.unwrap_or_else(|| "spv".to_owned())
			);

		let mut file = match std::fs::File::create(&path) {
			Ok(f) => f,
			Err(e) if e.kind() == std::io::ErrorKind::NotFound => panic!("Could not find path {path:?} for writing!"),
			Err(e) => panic!("Failed to open file {path:?} for writing! Error: {e:?}"),
		};

		let mut registry = TypeRegistry::new();
		registry.register::<Spirv>();

		let serializer = TypedReflectSerializer::new(spirv.as_partial_reflect(), &registry);
		let serialized_value = ron::ser::to_string_pretty(&serializer, ron::ser::PrettyConfig::default()).unwrap();

		file.write_all(serialized_value.as_bytes()).unwrap();
	}

	if !compile_errs.is_empty() {
		let msg = compile_errs
			.into_iter()
			.map(|(file_name, err)| format!("Failed to compile {file_name}!\nError: {err}\n"))
			.collect::<String>();

		panic!("\n{msg}");
	}
}

/*
// TODO: Only recompile changed shader files.
fn main() {
	color_eyre::install().unwrap();

	println!("cargo:rerun-if-changed={}", SHADER_DIR);

	if !Path::new(SHADER_DIR).exists() {
		println!("cargo:warning=Shader directory \"{SHADER_DIR}\" does not exist!");
		return;
	}

	let compiled_count = fs::read_dir(SHADER_DIR)
		.unwrap()
		.filter_map(|x| {
			let Ok(entry) = x else {
				return None;
			};

			let path = entry.path();
			let Some(ext) = path.extension() else {
				println!("cargo:warning=No extension for file: \"{}\"", path.display());
				return None;
			};

			if !matches!(ext.to_str().unwrap(), "vert" | "frag" | "comp" | "geom" | "tese" | "tesc" | "mesh") {
				println!("cargo:warning=Unsupported shader extension: \"{ext:?}\" for {path:?}!");
				return None;
			}


			Some(path)
		})
		.map(|path| compile_shader(&path) as usize)
		.sum::<usize>();

	println!("Successfully compiled {compiled_count} shaders!");
}

fn compile_shader(path: &Path) -> bool {
	let shader_name = path.file_name().unwrap().to_str().unwrap();

	let shader_path = path.to_str().unwrap();
	let spv_path = shader_path.replace(SHADER_DIR, SPV_DIR) + ".spv";

	println!("Compiling {shader_name}...");

	let output = Command::new("glslangValidator")
		.arg("-V")
		.arg(shader_path)
		.arg("-o")
		.arg(spv_path)
		.output()
		.expect("Failed to execute glslangValidator");

	let success = output.status.success();
	if success {
		println!("Compiled {shader_name}! Log: {}", String::from_utf8_lossy(&output.stdout));
	} else {
		panic!("Failed to compile {shader_name}!\nError: {}", String::from_utf8_lossy(&output.stdout));
	}

	success
}*/
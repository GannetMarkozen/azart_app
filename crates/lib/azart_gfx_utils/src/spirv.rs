use bevy::utils::HashMap;
use bevy::reflect::Reflect;
use crate::Format;
use ash::vk;

#[derive(Default, Clone, Debug, Reflect)]
pub struct Spirv {
	pub code: Vec<u32>,
	pub stage: ShaderStage,
	pub bindings: HashMap<String, DescriptorBinding>,
	pub vertex_attributes: HashMap<String, VertexAttribute>,
	pub push_constants: HashMap<String, PushConstant>,
	pub specialization_constants: HashMap<String, SpecializationConstant>,
}

#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, Reflect)]
pub enum ShaderStage {
	#[default]
	Vertex,
	Fragment,
	Compute,
	Geometry,
	TessControl,
	TessEval,
	Mesh,
	Task,
	RayGen,
	AnyHit,
}

impl Into<vk::ShaderStageFlags> for ShaderStage {
	fn into(self) -> vk::ShaderStageFlags {
		use ShaderStage::*;
		match self {
			Vertex => vk::ShaderStageFlags::VERTEX,
			Fragment => vk::ShaderStageFlags::FRAGMENT,
			Compute => vk::ShaderStageFlags::COMPUTE,
			Geometry => vk::ShaderStageFlags::GEOMETRY,
			TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
			TessEval => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
			Mesh => vk::ShaderStageFlags::MESH_EXT,
			Task => vk::ShaderStageFlags::TASK_EXT,
			RayGen => vk::ShaderStageFlags::RAYGEN_KHR,
			AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
		}
	}
}

#[derive(Default, Clone, Eq, PartialEq, Hash, Debug, Reflect)]
pub struct DescriptorBinding {
	pub set: u32,
	pub binding: u32,
	pub descriptor_type: DescriptorType,
	pub container_type: ContainerType,
}

#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, Reflect)]
pub enum DescriptorType {
	Sampler,
	CombinedImageSampler,
	#[default]
	SampledImage,
	StorageImage,
	UniformTexelBuffer,
	StorageTexelBuffer,
	UniformBuffer,
	StorageBuffer,
	UniformBufferDynamic,
	InputAttachment,
	AccelerationStructureNV,
}

impl From<vk::DescriptorType> for DescriptorType {
	fn from(value: vk::DescriptorType) -> Self {
		use DescriptorType::*;
		match value {
			vk::DescriptorType::SAMPLER => Sampler,
			vk::DescriptorType::COMBINED_IMAGE_SAMPLER => CombinedImageSampler,
			vk::DescriptorType::SAMPLED_IMAGE => SampledImage,
			vk::DescriptorType::STORAGE_IMAGE => StorageImage,
			vk::DescriptorType::UNIFORM_TEXEL_BUFFER => UniformTexelBuffer,
			vk::DescriptorType::STORAGE_TEXEL_BUFFER => StorageTexelBuffer,
			vk::DescriptorType::UNIFORM_BUFFER => UniformBuffer,
			vk::DescriptorType::STORAGE_BUFFER => StorageBuffer,
			vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => UniformBufferDynamic,
			vk::DescriptorType::INPUT_ATTACHMENT => InputAttachment,
			vk::DescriptorType::ACCELERATION_STRUCTURE_NV => AccelerationStructureNV,
			_ => panic!("Unsupported descriptor type {value:?}"),
		}
	}
}

impl Into<vk::DescriptorType> for DescriptorType {
	fn into(self) -> vk::DescriptorType {
		use DescriptorType::*;
		match self {
			Sampler => vk::DescriptorType::SAMPLER,
			CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
			SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
			StorageImage => vk::DescriptorType::STORAGE_IMAGE,
			UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
			StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
			UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
			StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
			UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
			InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
			AccelerationStructureNV => vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
		}
	}
}

#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, Reflect)]
pub enum ContainerType {
	#[default]
	Single,
	Array(u32),
	Array2D([u32; 2]),
	Array3D([u32; 3]),
	RuntimeArray,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Reflect)]
pub struct VertexAttribute {
	pub location: u32,
	pub format: Format,
}

impl Default for VertexAttribute {
	fn default() -> Self {
		Self {
			location: 0,
			format: Format::RgbaU8,
		}
	}
}

#[derive(Clone, Debug, Reflect)]
pub struct PushConstant {

}

#[derive(Clone, Debug, Reflect)]
pub struct SpecializationConstant {
	pub binding: u32,
	pub specialization_constant_type: PrimType,
	pub container_type: ContainerType,// Can't be RuntimeArray with specialization constants.
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Reflect)]
pub enum PrimType {
	Bool,
	U32,
	I32,
	U64,
	I64,
	F32,
	F64,
}

#[derive(Clone, Copy, Debug)]
pub enum PrimValue {
	Bool(bool),
	U32(u32),
	I32(i32),
	U64(u64),
	I64(i64),
	F32(f32),
	F64(f64),
}

impl Into<PrimType> for PrimValue {
	fn into(self) -> PrimType {
		match self {
			PrimValue::Bool(_) => PrimType::Bool,
			PrimValue::U32(_) => PrimType::U32,
			PrimValue::I32(_) => PrimType::I32,
			PrimValue::U64(_) => PrimType::U64,
			PrimValue::I64(_) => PrimType::I64,
			PrimValue::F32(_) => PrimType::F32,
			PrimValue::F64(_) => PrimType::F64,
		}
	}
}

impl From<bool> for PrimValue {
	fn from(value: bool) -> Self {
		PrimValue::Bool(value)
	}
}

impl From<u32> for PrimValue {
	fn from(value: u32) -> Self {
		PrimValue::U32(value)
	}
}

impl From<i32> for PrimValue {
	fn from(value: i32) -> Self {
		PrimValue::I32(value)
	}
}

impl From<u64> for PrimValue {
	fn from(value: u64) -> Self {
		PrimValue::U64(value)
	}
}

impl From<i64> for PrimValue {
	fn from(value: i64) -> Self {
		PrimValue::I64(value)
	}
}

impl From<f32> for PrimValue {
	fn from(value: f32) -> Self {
		PrimValue::F32(value)
	}
}

impl From<f64> for PrimValue {
	fn from(value: f64) -> Self {
		PrimValue::F64(value)
	}
}
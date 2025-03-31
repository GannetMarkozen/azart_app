use std::collections::BTreeMap;
use std::num::NonZero;
use std::slice;
use std::sync::Arc;
use ash::vk;
use bevy::utils::hashbrown::hash_map::Entry;
use bevy::utils::HashMap;
use spirv_reflect::types::*;
use crate::GpuContext;
use azart_gfx_utils::{GpuResource, MsaaCount, ShaderPath};
use azart_utils::debug_string::DebugString;
use spirv_headers::Op;

pub struct GraphicsPipeline {
	pub(crate) handle: vk::Pipeline,
	pub(crate) layout: vk::PipelineLayout,
	pub(crate) descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
	pub(crate) context: Arc<GpuContext>,
}

impl GraphicsPipeline {
	// render_pass must outlive the construction time but can be safely destroyed afterward.
	pub unsafe fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		render_pass: vk::RenderPass,
		create_info: &GraphicsPipelineCreateInfo,
	) -> Self {
		assert_ne!(render_pass, vk::RenderPass::null());

		// TODO: Allow entry-point as input.
		const ENTRY_POINT: &str = "main";

		let vertex_shader_module = context.create_shader_module(&create_info.vertex_shader).unwrap();
		let fragment_shader_module = context.create_shader_module(&create_info.fragment_shader).unwrap();
		let reflect_vertex_shader_module = spirv_reflect::create_shader_module(bytemuck::cast_slice(&vertex_shader_module.spirv.code)).unwrap();
		let reflect_fragment_shader_module = spirv_reflect::create_shader_module(bytemuck::cast_slice(&fragment_shader_module.spirv.code)).unwrap();
		
		let (pipeline_layout, descriptor_set_layouts) = {
			assert!(!vertex_shader_module.spirv.code.is_empty());
			assert!(!fragment_shader_module.spirv.code.is_empty());

			#[derive(Clone, PartialEq, Eq, Hash, Debug)]
			enum StorageType {
				Single,
				Array(u32),
				Array2D([u32; 2]),
				Array3D([u32; 3]),
				RuntimeArray,// Always single-dimensional.
			}

			#[derive(Clone, PartialEq, Eq, Hash, Debug)]
			struct Binding {
				name: String,
				binding: u32,
				storage_type: StorageType,
				descriptor_type: vk::DescriptorType,
			}

			let modules = [
				(&reflect_vertex_shader_module, vk::ShaderStageFlags::VERTEX),
				(&reflect_fragment_shader_module, vk::ShaderStageFlags::FRAGMENT),
			];

			// Using BTreeMap because the key (descriptor set index) needs to be sorted in min..max value.
			let mut sets = BTreeMap::<u32, HashMap<Binding, vk::ShaderStageFlags>>::new();
			for (module, stage) in modules.into_iter() {
				for set in module.enumerate_descriptor_sets(None).unwrap().into_iter() {
					let bindings = sets.entry(set.set).or_default();

					for binding in set.bindings.into_iter() {
						let storage_type = match (binding.type_description.as_ref().map(|desc| *desc.op), binding.array.dims.as_slice()) {
							(Some(Op::TypeArray), &[x]) => StorageType::Array(x),
							(Some(Op::TypeArray), &[x, y]) => StorageType::Array2D([x, y]),
							(Some(Op::TypeArray), &[x, y, z]) => StorageType::Array3D([x, y, z]),
							(Some(Op::TypeRuntimeArray), _) => StorageType::RuntimeArray,
							(Some(Op::TypeArray), dims) => panic!("Unsupported array dimensionality {dims:?}!"),
							(None, _) => panic!("Unsupported reflected binding {binding:?}!"),
							_ => StorageType::Single,
						};

						let descriptor_type = match binding.descriptor_type {
							ReflectDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
							ReflectDescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
							ReflectDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
							ReflectDescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
							ReflectDescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
							ReflectDescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
							ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
							ReflectDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
							ReflectDescriptorType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
							ReflectDescriptorType::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
							ReflectDescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
							ReflectDescriptorType::AccelerationStructureNV => vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
							_ => panic!("Unsupported descriptor type {:?}", binding.descriptor_type),
						};

						let binding = Binding {
							name: binding.name,
							binding: binding.binding,
							storage_type,
							descriptor_type,
						};

						match bindings.entry(binding) {
							Entry::Occupied(entry) => *entry.into_mut() |= stage,
							Entry::Vacant(entry) => _ = entry.insert(stage),
						}
					}
				}
			}
			
			assert!(
				sets
					.iter()
					.zip(sets
						.iter()
						.skip(1)
					)
					.all(|((&a, _), (&b, _))| a + 1 == b),
				"Descriptor set indices must be consecutive: {:?}!", sets.keys()
			);

			let layouts = sets
				.into_values()
				.map(|bindings| {
					let bindings = bindings
						.into_iter()
						.map(|(binding, stages)| vk::DescriptorSetLayoutBinding::default()
							.binding(binding.binding)
							.descriptor_type(binding.descriptor_type)
							.descriptor_count(match binding.storage_type {
								StorageType::Single => 1,
								StorageType::Array(x) => x,
								StorageType::Array2D([x, y]) => x * y,
								StorageType::Array3D([x, y, z]) => x * y * z,
								StorageType::RuntimeArray => vk::REMAINING_ARRAY_LAYERS,
							})
							.stage_flags(stages)
						)
						.collect::<Vec<_>>();

					let create_info = vk::DescriptorSetLayoutCreateInfo::default()
						.bindings(&bindings);

					unsafe { context.device.create_descriptor_set_layout(&create_info, None) }.unwrap()
				})
				.collect::<Vec<_>>();

			let create_info = vk::PipelineLayoutCreateInfo::default()
				.set_layouts(&layouts);
			
			let pipeline_layout = unsafe { context.device.create_pipeline_layout(&create_info, None) }.unwrap();

			(pipeline_layout, layouts)
		};

		let (vertex_binding_descriptions, vertex_attribute_descriptions) = {
			#[derive(Debug, PartialEq, Eq, Hash)]
			struct VertexAttribute {
				location: u32,
				format: vk::Format,
			}
			
			let attributes = reflect_vertex_shader_module
				.enumerate_input_variables(Some(ENTRY_POINT))
				.unwrap()
				.into_iter()
				.filter(|x| !x.decoration_flags.contains(ReflectDecorationFlags::BUILT_IN))
				.map(|x| {
					let attribute = VertexAttribute {
						location: x.location,
						format: match x.format {
							ReflectFormat::R32_UINT => vk::Format::R32_UINT,
							ReflectFormat::R32_SINT => vk::Format::R32_SINT,
							ReflectFormat::R32_SFLOAT => vk::Format::R32_SFLOAT,
							ReflectFormat::R32G32_UINT => vk::Format::R32G32_UINT,
							ReflectFormat::R32G32_SINT => vk::Format::R32G32_SINT,
							ReflectFormat::R32G32_SFLOAT => vk::Format::R32G32_SFLOAT,
							ReflectFormat::R32G32B32_UINT => vk::Format::R32G32B32_UINT,
							ReflectFormat::R32G32B32_SINT => vk::Format::R32G32B32_SINT,
							ReflectFormat::R32G32B32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
							ReflectFormat::R32G32B32A32_UINT => vk::Format::R32G32B32A32_UINT,
							ReflectFormat::R32G32B32A32_SINT => vk::Format::R32G32B32A32_SINT,
							ReflectFormat::R32G32B32A32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
							ReflectFormat::Undefined => panic!("Need to implement user-defined vertex attributes!\nInput: {x:?}"),
						},
					};

					(x.name, attribute)
				}
				)
				.collect::<HashMap<_, _>>();

			let vertex_binding_descriptions = create_info.vertex_inputs
				.iter()
				.enumerate()
				.map(|(binding, input)| vk::VertexInputBindingDescription::default()
					.binding(binding as u32)
					.stride(input.stride.unwrap().get())
					.input_rate(vk::VertexInputRate::VERTEX)
				)
				.collect::<Vec<_>>();
			
			let vertex_attributes = attributes
				.iter()
				.map(|(name, attribute)| {
					let (binding, vertex_attribute) = create_info.vertex_inputs
						.iter()
						.enumerate()
						.find_map(|(binding, input)| input.attributes
							.iter()
							.find(|x| x.name == *name)
							.map(|x| (binding, x))
						)
						.unwrap_or_else(|| panic!("Vertex attribute {name} not in vertex inputs: {:?}!", create_info.vertex_inputs));
					
					vk::VertexInputAttributeDescription::default()
						.binding(binding as u32)
						.location(attribute.location)
						.format(attribute.format)
						.offset(vertex_attribute.offset)
				})
				.collect::<Vec<_>>();

			(vertex_binding_descriptions, vertex_attributes)
		};
		
		let pipeline = {
			let stages = [
				vk::PipelineShaderStageCreateInfo::default()
					.stage(vk::ShaderStageFlags::VERTEX)
					.module(vertex_shader_module.handle)
					.name(c"main"),
				vk::PipelineShaderStageCreateInfo::default()
					.stage(vk::ShaderStageFlags::FRAGMENT)
					.module(fragment_shader_module.handle)
					.name(c"main"),
			];
			
			let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
				.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
				.cull_mode(vk::CullModeFlags::NONE)
				.rasterizer_discard_enable(false)
				.line_width(1.0);

			let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
				.dynamic_states(&[
					vk::DynamicState::VIEWPORT,
					vk::DynamicState::SCISSOR,
				]);

			let viewport_state = vk::PipelineViewportStateCreateInfo::default()
				.viewport_count(1)
				.scissor_count(1);

			let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
				.vertex_binding_descriptions(&vertex_binding_descriptions)
				.vertex_attribute_descriptions(&vertex_attribute_descriptions);
			
			let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
				.topology(vk::PrimitiveTopology::TRIANGLE_LIST)
				.primitive_restart_enable(false);
			
			let color_attachments = [
				vk::PipelineColorBlendAttachmentState::default()
					.color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A)
					.blend_enable(false)
					.color_blend_op(vk::BlendOp::ADD)
					.src_color_blend_factor(vk::BlendFactor::ONE)
					.dst_color_blend_factor(vk::BlendFactor::ZERO),
			];
			
			let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
				.logic_op_enable(false)
				.attachments(&color_attachments);
			
			let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
				.depth_test_enable(true)
				.depth_write_enable(true)
				.depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
				.depth_bounds_test_enable(false)
				.stencil_test_enable(false);
			
			let msaa_state = vk::PipelineMultisampleStateCreateInfo::default()
				.rasterization_samples(create_info.msaa.as_vk_sample_count())
				.sample_shading_enable(false)
				.min_sample_shading(1.0);
			
			let create_info = vk::GraphicsPipelineCreateInfo::default()
				.render_pass(render_pass)
				.subpass(0)
				.stages(&stages)
				.rasterization_state(&rasterization_state)
				.dynamic_state(&dynamic_state)
				.viewport_state(&viewport_state)
				.vertex_input_state(&vertex_input_state)
				.input_assembly_state(&input_assembly_state)
				.color_blend_state(&color_blend_state)
				.depth_stencil_state(&depth_stencil_state)
				.multisample_state(&msaa_state)
				.layout(pipeline_layout)
				.base_pipeline_handle(vk::Pipeline::null())
				.base_pipeline_index(-1);
			
			unsafe { context.device.create_graphics_pipelines(vk::PipelineCache::null(), slice::from_ref(&create_info), None).unwrap()[0] }
		};
		
		#[cfg(debug_assertions)]
		unsafe {
			context.set_debug_name(name.as_str(), pipeline);
		}

		drop((vertex_shader_module, fragment_shader_module));
		
		Self {
			handle: pipeline,
			descriptor_set_layouts,
			layout: pipeline_layout,
			context,
		}
	}
}

impl Drop for GraphicsPipeline {
	fn drop(&mut self) {
		unsafe {
			self.context.device.destroy_pipeline_layout(self.layout, None);
			self.context.device.destroy_pipeline(self.handle, None);
		}
	}
}

impl GpuResource for GraphicsPipeline {}

pub struct GraphicsPipelineCreateInfo<'a> {
	pub vertex_shader: &'a ShaderPath,
	pub fragment_shader: &'a ShaderPath,
	pub vertex_inputs: &'a [VertexInput<'a>],// Can be zero-len.
	pub msaa: MsaaCount,
}

#[derive(Debug)]
pub struct VertexInput<'a> {
	pub stride: Option<NonZero<u32>>,
	pub attributes: &'a [VertexAttribute<'a>],
}

#[derive(Debug)]
pub struct VertexAttribute<'a> {
	pub name: &'a str,
	pub offset: u32,
}
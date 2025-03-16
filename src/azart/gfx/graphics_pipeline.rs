use std::collections::BTreeMap;
use std::slice;
use std::sync::Arc;
use ash::vk;
use bevy::utils::{HashMap, HashSet};
use spirv_reflect::types::{ReflectDescriptorBinding, ReflectDescriptorType};
use crate::azart::gfx::GpuContext;
use crate::azart::gfx::misc::{GpuResource, MsaaCount, ShaderPath};
use crate::azart::utils::debug_string::DebugString;

pub struct GraphicsPipeline {
	pub(crate) handle: vk::Pipeline,
	pub(crate) layout: vk::PipelineLayout,
	pub(crate) context: Arc<GpuContext>,
}

impl GraphicsPipeline {
	// render_pass must outlive the construction time but can be safely destroyed afterward. Not thread-safe!
	pub unsafe fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		render_pass: vk::RenderPass,
		create_info: &GraphicsPipelineCreateInfo,
	) -> Self {
		assert_ne!(render_pass, vk::RenderPass::null());

		let vertex_shader_module = context.create_shader_module(create_info.vertex_shader).unwrap();
		let fragment_shader_module = context.create_shader_module(create_info.fragment_shader).unwrap();

		#[derive(Clone, PartialEq, Eq, Hash, Debug)]
		enum StorageType {
			Single,
			Array(Vec<u32>),// Multi-dimensional. At least 1 element in array.
			UnboundedArray,// Always single-dimensional.
		}

		#[derive(Clone, PartialEq, Eq, Hash, Debug)]
		struct Binding {
			name: String,
			binding: u32,
			storage_type: StorageType,
			descriptor_type: vk::DescriptorType,
		}
		
		let layout = {
			assert!(!vertex_shader_module.code.is_empty());
			assert!(!fragment_shader_module.code.is_empty());

			let modules = [
				spirv_reflect::create_shader_module(&vertex_shader_module.code).expect("Failed to reflect vertex shader module!"),
				spirv_reflect::create_shader_module(&fragment_shader_module.code).expect("Failed to reflect fragment shader module!"),
			];

			// Using BTreeMap because the key (set) needs to be sorted in min..max value.
			let mut sets = BTreeMap::<u32, HashSet<Binding>>::new();
			for module in modules.into_iter() {
				for set in module.enumerate_descriptor_sets(None).unwrap().into_iter() {
					let bindings = sets.entry(set.set).or_insert(HashSet::new());
					for binding in set.bindings.into_iter() {
						bindings.insert(Binding {
							name: binding.name,
							binding: binding.binding,
							storage_type: match (binding.count, binding.array.dims.len()) {
								(1, 0) => StorageType::Single,
								(1.., 1..) => StorageType::Array(binding.array.dims),
								(0, 1..) => StorageType::UnboundedArray,
								_ => unreachable!(),
							},
							descriptor_type: match binding.descriptor_type {
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
							},
						});
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
				.into_iter()
				.map(|(set, bindings)| {
					let bindings = bindings
						.into_iter()
						.map(|binding| vk::DescriptorSetLayoutBinding::default()
							.binding(binding.binding)
							.descriptor_type(binding.descriptor_type)
							.descriptor_count(match &binding.storage_type {
								StorageType::Single => 1,
								StorageType::Array(dims) => dims.iter().product(),
								StorageType::UnboundedArray => vk::REMAINING_ARRAY_LAYERS,
							})
						)
						.collect::<Vec<_>>();

					let create_info = vk::DescriptorSetLayoutCreateInfo::default()
						.bindings(&bindings);

					unsafe { context.device.create_descriptor_set_layout(&create_info, None) }.unwrap()
				})
				.collect::<Vec<_>>();

			let create_info = vk::PipelineLayoutCreateInfo::default()
				.set_layouts(&layouts);
			
			unsafe { context.device.create_pipeline_layout(&create_info, None) }.unwrap()
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
				.vertex_binding_descriptions(&[])
				.vertex_attribute_descriptions(&[]);
			
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
				.layout(layout)
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
			layout,
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
	pub vertex_shader: ShaderPath<'a>,
	pub fragment_shader: ShaderPath<'a>,
	pub msaa: MsaaCount,
}
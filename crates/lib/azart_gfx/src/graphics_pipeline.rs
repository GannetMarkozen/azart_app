use std::collections::BTreeMap;
use std::slice;
use std::sync::Arc;
use ash::vk;
use bevy::utils::hashbrown::hash_map::Entry;
use bevy::utils::HashMap;
use crate::{GpuContext, ShaderModule};
use azart_gfx_utils::{spirv, GpuResource, Msaa, ShaderPath, TriangleFillMode};
use azart_gfx_utils::spirv::*;
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
	pub fn new(
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

		let (vertex_binding_descriptions, vertex_attribute_descriptions) = {
			let vertex_binding_descriptions = create_info
				.vertex_inputs
				.iter()
				.enumerate()
				.filter_map(|(binding, input)| input
					.attributes
					.iter()
					.any(|attribute| vertex_shader_module
						.spirv
						.vertex_attributes
						.contains_key(attribute.name)
					)
					.then(|| vk::VertexInputBindingDescription::default()
						.binding(binding as u32)
						.stride(input.stride)
						.input_rate(vk::VertexInputRate::VERTEX)
					)
				)
				.collect::<Vec<_>>();

			let vertex_attribute_descriptions = vertex_shader_module
				.spirv
				.vertex_attributes
				.iter()
				.map(|(name, attribute)| {
					let (binding, vertex_attribute) = create_info.vertex_inputs
						.iter()
						.enumerate()
						.find_map(|(binding, input)| input.attributes
							.iter()
							.find(|&&VertexAttribute { name: attribute_name, .. }| name == attribute_name)
							.map(|attribute| (binding, attribute))
						)
						.unwrap_or_else(|| panic!("Vertex attribute {name} not in vertex inputs: {:?}!", create_info.vertex_inputs));

					vk::VertexInputAttributeDescription::default()
						.binding(binding as u32)
						.location(attribute.location)
						.format(attribute.format.into())
						.offset(vertex_attribute.offset)
				})
				.collect::<Vec<_>>();

			(vertex_binding_descriptions, vertex_attribute_descriptions)
		};

		// TODO: Pool PipelineLayouts and re-use.
		let (pipeline_layout, descriptor_set_layouts) = {
			let mut sets = BTreeMap::<u32, HashMap<u32, (&DescriptorBinding, vk::ShaderStageFlags)>>::new();
			for spirv in [&vertex_shader_module.spirv, &fragment_shader_module.spirv] {
				for (_, binding) in spirv.bindings.iter() {
					let binding_entry = sets
						.entry(binding.set)
						.or_default()
						.entry(binding.binding);

					match binding_entry {
						Entry::Occupied(entry) => {
							assert_eq!(*binding, *entry.get().0, "Binding {} is already bound to a different descriptor set layout binding!", binding.binding);
							entry.into_mut().1 |= spirv.stage.into();
						},
						Entry::Vacant(entry) => _ = entry.insert((binding, spirv.stage.into())),
					}
				}
			}

			assert!(matches!(sets.first_key_value(), None | Some((0, _))), "First descriptor set layout location must be 0!");
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

			let descriptor_set_layouts = sets
				.into_iter()
				.enumerate()
				.map(|(i, (_, bindings))| {
					let bindings = bindings
						.into_iter()
						.map(|(_, (binding, stages))| vk::DescriptorSetLayoutBinding::default()
							.binding(binding.binding)
							.descriptor_type(binding.descriptor_type.into())
							.descriptor_count(match binding.container_type {
								ContainerType::Single => 1,
								ContainerType::Array(x) => x,
								ContainerType::Array2D([x, y]) => x * y,
								ContainerType::Array3D([x, y, z]) => x * y * z,
								ContainerType::RuntimeArray => vk::REMAINING_ARRAY_LAYERS,
							})
							.stage_flags(stages)
						)
						.collect::<Vec<_>>();

					let create_info = vk::DescriptorSetLayoutCreateInfo::default()
						.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
						.bindings(&bindings);

					let descriptor_set_layout = unsafe { context.device.create_descriptor_set_layout(&create_info, None) }.unwrap();

					#[cfg(debug_assertions)]
					unsafe {
						context.set_debug_name(format!("{name}_descriptor_set_layout[{i}]").as_str(), descriptor_set_layout);
					}

					descriptor_set_layout
				})
				.collect::<Vec<_>>();

			let create_info = vk::PipelineLayoutCreateInfo::default()
				.set_layouts(&descriptor_set_layouts);

			let pipeline_layout = unsafe { context.device.create_pipeline_layout(&create_info, None) }.unwrap();

			(pipeline_layout, descriptor_set_layouts)
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
				.cull_mode(vk::CullModeFlags::BACK)
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

			assert_eq!(vertex_binding_descriptions.is_empty(), vertex_attribute_descriptions.is_empty());

			let vertex_input_state = match vertex_binding_descriptions.is_empty() {
				true => vk::PipelineVertexInputStateCreateInfo::default(),
				false => vk::PipelineVertexInputStateCreateInfo::default()
					.vertex_binding_descriptions(&vertex_binding_descriptions)
					.vertex_attribute_descriptions(&vertex_attribute_descriptions),
			};

			let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
				.topology(match create_info.fill_mode {
					TriangleFillMode::Fill => vk::PrimitiveTopology::TRIANGLE_LIST,
					TriangleFillMode::Wireframe => vk::PrimitiveTopology::LINE_LIST,
				})
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
			context.set_debug_name(format!("{name}_layout").as_str(), pipeline_layout);
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
			for &descriptor_set_layout in self.descriptor_set_layouts.iter() {
				self.context.device.destroy_descriptor_set_layout(descriptor_set_layout, None);
			}

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
	pub msaa: Msaa,
	pub fill_mode: TriangleFillMode,
}

#[derive(Debug)]
pub struct VertexInput<'a> {
	pub stride: u32,
	pub attributes: &'a [VertexAttribute<'a>],
}

#[derive(Debug)]
pub struct VertexAttribute<'a> {
	pub name: &'a str,
	pub offset: u32,
}

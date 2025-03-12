use std::path::Path;
use std::slice;
use std::sync::Arc;
use ash::vk;
use bevy::tasks::AsyncComputeTaskPool;
use crate::azart::gfx::GpuContext;
use crate::azart::gfx::misc::ShaderPath;
use crate::azart::gfx::render_pass::RenderPass;
use crate::azart::utils::debug_string::DebugString;

pub struct GraphicsPipeline {
	pub(crate) handle: vk::Pipeline,
	pub(crate) render_pass: vk::RenderPass,
	pub(crate) layout: vk::PipelineLayout,
	pub(crate) context: Arc<GpuContext>,
}

impl GraphicsPipeline {
	pub unsafe fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		render_pass: vk::RenderPass,
		create_info: &GraphicsPipelineCreateInfo,
	) -> Self {
		assert_ne!(render_pass, vk::RenderPass::null());
		
		let layout = {
			let create_info = vk::PipelineLayoutCreateInfo::default()
				.set_layouts(&[]);
			
			unsafe { context.device.create_pipeline_layout(&create_info, None) }.unwrap()
		};
		
		let pipeline = {
			let vertex_shader_module = context.create_shader_module(create_info.vertex_shader).unwrap();
			let fragment_shader_module = context.create_shader_module(create_info.fragment_shader).unwrap();
			
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
				.depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
				.depth_bounds_test_enable(false)
				.stencil_test_enable(false);
			
			let msaa_state = vk::PipelineMultisampleStateCreateInfo::default()
				.rasterization_samples(vk::SampleCountFlags::TYPE_1)
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
		
		Self {
			handle: pipeline,
			render_pass,
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

pub struct GraphicsPipelineCreateInfo<'a> {
	pub vertex_shader: ShaderPath<'a>,
	pub fragment_shader: ShaderPath<'a>,
}
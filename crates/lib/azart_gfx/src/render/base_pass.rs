use std::sync::Arc;
use ash::vk;
use azart_gfx_utils::Msaa;
use bevy::math::UVec2;
use either::{for_both, Either};
use crate::{rhi_label, GpuContext};
use crate::graphics_pipeline::GraphicsPipeline;
use crate::render::plugin::{align_to, CubeMesh, Sampler, Texture, ViewMatrices, ViewMatricesBuffer};

pub fn record_base_pass(
	context: &Arc<GpuContext>,
	cmd: vk::CommandBuffer,
	image: vk::Image,
	render_pass: vk::RenderPass,
	frame_buffer: vk::Framebuffer,
	resolution: UVec2,
	pipeline: &GraphicsPipeline,
	mesh: &CubeMesh,
	view_matrices: &ViewMatricesBuffer,
	texture: &Texture,
	sampler: &Sampler,
	frame_index: usize,
	msaa: Msaa,
) {
	rhi_label!("base_pass", context, &cmd);

	/*unsafe {
		context.pipeline_barrier(
			cmd,
			None,
			&[],
			&[ImageBarrier {
				previous_accesses: &[],
				next_accesses: &[AccessType::TransferWrite],
				previous_layout: ImageLayout::Optimal,
				next_layout: ImageLayout::Optimal,
				discard_contents: true,
				src_queue_family_index: context.queue_families.graphics,
				dst_queue_family_index: context.queue_families.graphics,
				image: swapchain_frame_buffer.image,
				range: vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.level_count(1)
					.layer_count(2),
			}],
		);

		context.device.cmd_clear_color_image(
			cmd,
			swapchain_frame_buffer.image,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			&vk::ClearColorValue { float32: [0.0, 1.0, 0.0, 1.0] },
			&[vk::ImageSubresourceRange::default()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.level_count(1)
				.layer_count(2),
			]
		);

		context.pipeline_barrier(
			cmd,
			None,
			&[],
			&[ImageBarrier {
				previous_accesses: &[AccessType::TransferWrite],
				next_accesses: &[AccessType::TransferRead],
				previous_layout: ImageLayout::Optimal,
				next_layout: ImageLayout::Optimal,
				discard_contents: false,
				src_queue_family_index: context.queue_families.graphics,
				dst_queue_family_index: context.queue_families.graphics,
				image: swapchain_frame_buffer.image,
				range: vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.level_count(1)
					.layer_count(2),
			}],
		);
	}*/

	let device = &context.device;

	unsafe {
		///
		/// BEGIN RENDER PASS
		///

		let clear_values = match msaa {
			Msaa::None => Either::Left([
				vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
				vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } },
			]),
			_ => Either::Right([
				vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
				vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } },
				vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
			])
		};

		context.exts.create_render_pass2.cmd_begin_render_pass2(
			cmd,
			&vk::RenderPassBeginInfo::default()
				.render_pass(render_pass)
				.clear_values(for_both!(&clear_values, v => v.as_slice()))
				.framebuffer(frame_buffer)
				.render_area(vk::Rect2D {
					offset: vk::Offset2D { x: 0, y: 0 },
					extent: vk::Extent2D {
						width: resolution.x as _,
						height: resolution.y as _,
					},
				}),
			&vk::SubpassBeginInfo::default()
				.contents(vk::SubpassContents::INLINE)
		);

		device.cmd_set_viewport(
			cmd,
			0,
			&[vk::Viewport {
				x: 0.0,
				y: 0.0,
				width: resolution.x as _,
				height: resolution.y as _,
				min_depth: 1.0,// Reverse depth for better precision.
				max_depth: 0.0,
			}],
		);

		device.cmd_set_scissor(
			cmd,
			0,
			&[vk::Rect2D {
				offset: vk::Offset2D { x: 0, y: 0 },
				extent: vk::Extent2D {
					width: resolution.x as _,
					height: resolution.y as _,
				},
			}],
		);

		///
		/// DRAW
		///

		context.exts.push_descriptor.cmd_push_descriptor_set(
			cmd,
			vk::PipelineBindPoint::GRAPHICS,
			pipeline.layout,
			0,
			&[
				vk::WriteDescriptorSet::default()
					.dst_binding(0)
					.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
					.buffer_info(&[vk::DescriptorBufferInfo::default()
						.buffer(view_matrices.handle)
						.offset((align_to(size_of::<ViewMatrices>(), context.limits.ubo_min_alignment) * frame_index) as _)
						.range(size_of::<ViewMatrices>() as _)
					]),
				vk::WriteDescriptorSet::default()
					.dst_binding(1)
					.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
					.image_info(&[vk::DescriptorImageInfo::default()
						.sampler(sampler.handle)
						.image_view(texture.view)
						.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
					]),
			]
		);

		device.cmd_bind_pipeline(
			cmd,
			vk::PipelineBindPoint::GRAPHICS,
			pipeline.handle,
		);

		device.cmd_bind_index_buffer(
			cmd,
			mesh.index_buffer.handle,
			0,
			vk::IndexType::UINT16,
		);

		device.cmd_bind_vertex_buffers(
			cmd,
			0,
			&[mesh.vertex_buffer.handle],
			&[0],
		);

		device.cmd_draw_indexed(
			cmd,
			36,
			2,
			0,
			0,
			0,
		);

		///
		/// END RENDER PASS
		///

		context.exts.create_render_pass2.cmd_end_render_pass2(
			cmd,
			&vk::SubpassEndInfo::default(),
		);
	}
}
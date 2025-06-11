use std::sync::Arc;
use ash::vk;
use azart_gfx_utils::Msaa;
use bevy::math::UVec2;
use bevy::prelude::*;
use either::{for_both, Either};
use crate::{rhi_label, GpuContext, GpuContextHandle};
use crate::buffer::Buffer;
use crate::graphics_pipeline::GraphicsPipeline;
use crate::render::camera::{Camera, XrCamera, XrView};
use crate::render::plugin::{align_to, CubeMesh, Sampler, Texture, ViewMatrices, ViewMatricesBuffer};
use crate::render::xr::{XrSession, XrSpace};
use openxr as xr;
use crate::render::gpu_scene::{GlobalUbo, GpuScene, RenderFrameIndex};
use crate::render::swapchain::Swapchain;
use crate::render_settings::RenderSettings;

#[derive(Resource, Deref, DerefMut)]
pub(crate) struct XrFrameState(pub(crate) xr::FrameState);

pub fn update_xr_camera_transform(
	mut cameras: Query<&mut XrCamera>,
	session: Res<XrSession>,
	frame_state: Res<XrFrameState>,
	space: Res<XrSpace>,
) {
	let Ok(mut camera) = cameras.get_single_mut() else {
		return;
	};

	let (view_flags, views) = session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, frame_state.predicted_display_time, &*space).unwrap();
	let views: [_; 2] = views.try_into().unwrap_or_else(|views: Vec<xr::View>| panic!("There are {} views instead of the presumed 2!", views.len()));

	camera.views = views.map(|xr::View { pose: xr::Posef { position: pos, orientation: rot }, fov }| {
		let pos = Vec3::new(pos.x, pos.y, pos.z);
		let rot = Quat::from_xyzw(rot.x, rot.y, rot.z, rot.w);

		XrView {
			pos,
			rot,
			fov,
		}
	});
}

pub fn update_global_ubo_xr(
	cx: Res<GpuContextHandle>,
	mut cameras: Query<(&Transform, &mut XrCamera)>,
	mut scene: ResMut<GpuScene>,
	frame_index: Res<RenderFrameIndex>,
	settings: Res<RenderSettings>,
) {
	let Ok((transform, mut camera)) = cameras.get_single_mut() else {
		return;
	};

	unsafe {
		const Z_FAR: f32 = 1000.0;
		const Z_NEAR: f32 = 0.025;

		let data = scene.global_ubo.allocation.mapped_ptr().unwrap().cast::<u8>();
		let ubo = data.add(align_to(size_of::<GlobalUbo>(), cx.limits.ubo_min_align) * (frame_index.0 % (settings.frames_in_flight as u64)) as usize).cast::<GlobalUbo>().as_mut();
		ubo.views = camera.views.map(|XrView { pos, rot, fov: xr::Fovf { angle_right: r, angle_left: l, angle_up: u, angle_down: d } }| {
			// world -> eye
			let view = (transform.compute_matrix() * Mat4::from_rotation_translation(rot, pos)).inverse();

			let (r, l, u, d) = (r.tan(), l.tan(), u.tan(), d.tan());
			let (w, h) = (r - l, d - u);

			let proj = Mat4::from_cols_array_2d(&[
				[2.0 / w, 0.0, 0.0, 0.0],
				[0.0, 2.0 / h, 0.0, 0.0],
				[(r + l) / w, (u + d) / h, -(Z_FAR + Z_NEAR) / (Z_FAR - Z_NEAR), -1.0],
				[0.0, 0.0, -(Z_FAR * (Z_NEAR + Z_NEAR)) / (Z_FAR - Z_NEAR), 0.0],
			]);

			proj * view
		});
	}
}

pub fn update_global_ubo(
	cx: Res<GpuContextHandle>,
	mut cameras: Query<(&Transform, &Camera)>,
	mut scene: ResMut<GpuScene>,
	swapchain: Query<&Swapchain>,
	frame_index: Res<RenderFrameIndex>,
	settings: Res<RenderSettings>,
) {
	let (Ok((transform, camera)), Ok(swapchain)) = (cameras.get_single(), swapchain.get_single()) else {
		return;
	};

	unsafe {
		const Z_FAR: f32 = 1000.0;
		const Z_NEAR: f32 = 0.025;

		let data = scene.global_ubo.allocation.mapped_ptr().unwrap().cast::<u8>();
		let ubo = data.add(align_to(size_of::<GlobalUbo>(), cx.limits.ubo_min_align) * (frame_index.0 % (settings.frames_in_flight as u64)) as usize).cast::<GlobalUbo>().as_mut();
		let mut proj = Mat4::perspective_rh(camera.fov.to_radians(), swapchain.resolution.x as f32 / swapchain.resolution.y as f32, Z_NEAR, Z_FAR);
		proj.y_axis.y *= -1.0;// Vulkan has inverted Y compared to OpenGL.
		ubo.views[0] = proj * transform.compute_matrix().inverse();
	}
}

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
						.offset((align_to(size_of::<ViewMatrices>(), context.limits.ubo_min_align) * frame_index) as _)
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
			if mesh.index_buffer.size() > u16::MAX as usize * size_of::<u16>() {
			  vk::IndexType::UINT32
			} else {
			  vk::IndexType::UINT16
			},
		);

		device.cmd_bind_vertex_buffers(
			cmd,
			0,
			&[mesh.vertex_buffer.handle],
			&[0],
		);

		device.cmd_draw_indexed(
			cmd,
			(mesh.index_buffer.size() / size_of::<u32>()) as u32,
			1,
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

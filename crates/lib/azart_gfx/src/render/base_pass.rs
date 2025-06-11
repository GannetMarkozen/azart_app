use std::ops::{Deref, DerefMut};
use std::slice;
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
use crate::render::plugin::{align_to, BasePass, CubeMesh, PbrPipeline, Sampler, Texture, ViewMatrices, ViewMatricesBuffer};
use crate::render::xr::{XrFrameStream, XrFrameWaiter, XrSession, XrSpace, XrSwapchain};
use openxr as xr;
use crate::render::gpu_scene::{GlobalUbo, GpuScene};
use crate::render::swapchain::Swapchain;
use crate::render_settings::RenderSettings;

/// Some when rendering.
/// @HACK: Transient resource. Only really needed during rendering. This just allows it to persist
/// without off by one frame errors with inserting and removing resources.
#[derive(Resource, Deref, DerefMut)]
pub(crate) struct XrFrameState(pub(crate) Option<xr::FrameState>);

pub fn update_xr_camera_transform(
	mut cameras: Query<&mut XrCamera>,
	session: Res<XrSession>,
	frame_state: Res<XrFrameState>,
	space: Res<XrSpace>,
) {
	let (Ok(mut camera), Some(frame_state)) = (cameras.get_single_mut(), frame_state.0.as_ref()) else {
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
	swapchain: Res<XrSwapchain>,
	settings: Res<RenderSettings>,
) {
	let Ok((transform, mut camera)) = cameras.get_single_mut() else {
		return;
	};

	unsafe {
		const Z_FAR: f32 = 1000.0;
		const Z_NEAR: f32 = 0.025;

		let data = scene.global_ubo.allocation.mapped_ptr().unwrap().cast::<u8>();
		let ubo = data.add(align_to(size_of::<GlobalUbo>(), cx.limits.ubo_min_align) * swapchain.current_frame_index).cast::<GlobalUbo>().as_mut();
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

			let mat = proj * view;

			mat.to_cols_array_2d()
		});
	}
}

pub fn update_global_ubo(
	cx: Res<GpuContextHandle>,
	mut cameras: Query<(&Transform, &Camera)>,
	mut scene: ResMut<GpuScene>,
	swapchain: Query<&Swapchain>,
	settings: Res<RenderSettings>,
) {
	let (Ok((transform, camera)), Ok(swapchain)) = (cameras.get_single(), swapchain.get_single()) else {
		return;
	};

	unsafe {
		const Z_FAR: f32 = 1000.0;
		const Z_NEAR: f32 = 0.025;

		let data = scene.global_ubo.allocation.mapped_ptr().unwrap().cast::<u8>();
		let ubo = data.add(align_to(size_of::<GlobalUbo>(), cx.limits.ubo_min_align) * swapchain.current_frame_index).cast::<GlobalUbo>().as_mut();
		let mut proj = Mat4::perspective_rh(camera.fov.to_radians(), swapchain.resolution.x as f32 / swapchain.resolution.y as f32, Z_NEAR, Z_FAR);
		proj.y_axis.y *= -1.0;// Vulkan has inverted Y compared to OpenGL.
		let mat = proj * transform.compute_matrix();
		ubo.views[0] = mat.to_cols_array_2d();
	}
}

pub(crate) fn begin_base_pass_xr(
	mut commands: Commands,
	cx: Res<GpuContextHandle>,
	session: Res<XrSession>,
	mut frame_waiter: ResMut<XrFrameWaiter>,
	mut frame_stream: ResMut<XrFrameStream>,
	mut out_frame_state: ResMut<XrFrameState>,
	mut swapchain: ResMut<XrSwapchain>,
	space: Res<XrSpace>,
) {
	#[cfg(target_os = "android")]
	{
		let ctx = ndk_context::android_context();
		let vm = unsafe { jni::JavaVM::from_raw(ctx.vm().cast()) }.unwrap();
		let env = vm.attach_current_thread().unwrap();
	}

	///
	/// BEGIN XR
	///

	let frame_state = match frame_waiter.wait() {
		Ok(frame_state) => frame_state,
		Ok(xr::FrameState { should_render: false, .. }) | Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => {
			warn!("Session not running!");
			return;
		},
		Err(e) => panic!("Failed to wait frame: {e}"),
	};

	frame_stream.begin().expect("Failed to begin frame stream!");
	let image_index = swapchain.handle.acquire_image().expect("Failed to acquire next image in swapchain!") as usize;
	let frame_index = (swapchain.current_frame_index + 1) % swapchain.frames.len();
	swapchain.handle.wait_image(xr::Duration::INFINITE).expect("Failed to wait image!");

	swapchain.current_frame_buffer_index = image_index;
	swapchain.current_frame_index = frame_index;

	let (view_flags, views) = session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, frame_state.predicted_display_time, &space).unwrap();
	assert_eq!(views.len(), 2, "There must be 2 views!");
	let views = [views[0], views[1]];

	///
	/// BEGIN VK
	///

	let frame = &swapchain.frames[frame_index];
	let swapchain_frame_buffer = &swapchain.swapchain_frame_buffers[image_index];

	unsafe {
		cx.device.wait_for_fences(slice::from_ref(&frame.in_flight_fence), true, u64::MAX).expect("Failed to wait for frame fences!");
		cx.device.reset_fences(slice::from_ref(&frame.in_flight_fence)).unwrap();
	}

	// Begin command buffer.
	unsafe {
		cx.device.reset_command_pool(frame.graphics_cmd_pool, vk::CommandPoolResetFlags::empty()).unwrap();
		cx.device.begin_command_buffer(frame.graphics_cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
	};

	*out_frame_state = XrFrameState(Some(frame_state));
}

pub(crate) fn end_base_pass_xr(
	cx: Res<GpuContextHandle>,
	mut swapchain: ResMut<XrSwapchain>,
	space: Res<XrSpace>,
	mut frame_stream: ResMut<XrFrameStream>,
	frame_state: Res<XrFrameState>,
	camera: Query<&XrCamera>,
) {
	///
	///	END VK
	///

	let Ok(camera) = camera.get_single() else {
		return;
	};

	let frame = swapchain.frame_in_flight();
	let cmd = frame.graphics_cmd;

	unsafe {
		cx.device.end_command_buffer(cmd).unwrap();

		let queue = cx.device.get_device_queue(cx.queue_families.graphics, 0);
		let submit_info = vk::SubmitInfo::default()
			.command_buffers(slice::from_ref(&cmd));

		cx.device.queue_submit(queue, slice::from_ref(&submit_info), frame.in_flight_fence).expect("Failed to submit command buffer!");
	}

	///
	/// END XR
	///

	swapchain.handle.release_image().expect("Failed to release image!");

	let rect = xr::Rect2Di {
		offset: xr::Offset2Di { x: 0, y: 0 },
		extent: xr::Extent2Di {
			width: swapchain.resolution.x as _,
			height: swapchain.resolution.y as _,
		},
	};

	/*let composition_projection_views: [_; 2] = std::array::from_fn(|i| {
		let xr::View { pose, fov } = views[i];
		xr::CompositionLayerProjectionView::new()
			.pose(pose)
			.fov(fov)
			.sub_image(xr::SwapchainSubImage::new()
				.swapchain(&swapchain.handle)// Ignore the borrow checker.
				.image_array_index(i as _)
				.image_rect(rect)
			)
	});*/

	let composition_projection_views: [_; 2] = std::array::from_fn(|i| {
		let XrView { pos, rot, fov } = camera.views[i];
		xr::CompositionLayerProjectionView::new()
			.pose(xr::Posef {
				orientation: xr::Quaternionf { x: rot.x, y: rot.y, z: rot.z, w: rot.w },
				position: xr::Vector3f { x: pos.x, y: pos.y, z: pos.z },
			})
			.fov(fov)
			.sub_image(xr::SwapchainSubImage::new()
				.swapchain(&swapchain.handle)
				.image_array_index(i as _)
				.image_rect(rect)
			)
	});

	let layer_projection = xr::CompositionLayerProjection::new()
		.space(&space)// Ignore the borrow checker.
		.views(&composition_projection_views);

	let result = frame_stream.end(frame_state.unwrap().predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[&*layer_projection]);
	match result {
		Ok(()) => {},
		Err(xr::sys::Result::ERROR_POSE_INVALID) => println!("Pose invalid!: {:?}", camera.views),
		Err(e) => panic!("Failed to end frame: {e}"),
	}
}

pub(crate) fn begin_base_pass(
	mut commands: Commands,
	cx: Res<GpuContextHandle>,
	mut swapchains: Query<&mut Swapchain>,
	settings: Res<RenderSettings>,
) {
	// @TODO: Allow multiple swapchains to exist to render to different windows!
	let Ok(mut swapchain) = swapchains.get_single_mut() else {
		return;
	};

	///
	/// WAIT FOR FRAME TO FINISH
	///
	let frame_index = swapchain.current_frame_index;
	let frame = &swapchain.frames[frame_index];
	let (image_index, suboptimal) = unsafe {
		cx.device.wait_for_fences(slice::from_ref(&frame.in_flight_fence), true, u64::MAX).expect("Failed to wait for frame fences!");

		let result = cx.exts.swapchain.acquire_next_image(swapchain.handle, u64::MAX, frame.image_available_semaphore, vk::Fence::null());
		assert_ne!(result, Err(vk::Result::SUBOPTIMAL_KHR));

		let (image_index, suboptimal) = match result {
			Ok(result) => result,
			Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => todo!("Recreate swapchain!"),
			Err(e) => panic!("Failed to acquire swapchain image!: {e}"),
		};

		cx.device.reset_fences(slice::from_ref(&frame.in_flight_fence)).unwrap();

		swapchain.current_frame_index = frame_index;//(frame_index + 1) % settings.frames_in_flight;
		swapchain.current_frame_buffer_index = image_index as _;

		(image_index as usize, suboptimal)
	};

	// Update frame with the latest frame (the one we just acquired).
	let frame = swapchain.frame_in_flight();

	let cmd = unsafe {
		cx.device.reset_command_pool(frame.graphics_cmd_pool, vk::CommandPoolResetFlags::empty()).unwrap();
		let cmd = frame.graphics_cmd;

		cx.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

		cmd
	};
}

pub(crate) fn end_base_pass(
	cx: Res<GpuContextHandle>,
	mut swapchains: Query<&mut Swapchain>,
	settings: Res<RenderSettings>,
) {
	// @TODO: Allow multiple swapchains!
	let Ok(mut swapchain) = swapchains.get_single_mut() else {
		return;
	};

	let frame = swapchain.frame_in_flight();
	let cmd = frame.graphics_cmd;
	let image_index = swapchain.current_frame_buffer_index;

	let queue = unsafe { cx.device.get_device_queue(cx.queue_families.graphics, 0) };

	// Submit queue.
	unsafe {
		cx.device.end_command_buffer(cmd).unwrap();

		let submit_info = vk::SubmitInfo::default()
			.command_buffers(slice::from_ref(&cmd))
			.wait_semaphores(slice::from_ref(&frame.image_available_semaphore))// @NOTE: Might need to be on image buffer instead?
			.wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
			.signal_semaphores(slice::from_ref(&frame.render_finished_semaphore));

		cx.device.queue_submit(queue, slice::from_ref(&submit_info), frame.in_flight_fence).expect("Failed to submit command buffer!");
	}

	// Queue present.
	unsafe {
		let image_indices = [image_index as _];
		let present_info = vk::PresentInfoKHR::default()
			.swapchains(slice::from_ref(&swapchain.handle))
			.image_indices(&image_indices)
			.wait_semaphores(slice::from_ref(&frame.render_finished_semaphore));

		let result = cx.exts.swapchain.queue_present(queue, &present_info);
		assert_ne!(result, Err(vk::Result::SUBOPTIMAL_KHR));

		if /*suboptimal ||*/ matches!(result, Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR)) {
			todo!("Handle");
		}
	}

	/*swapchain.current_frame_index = (frame_index + 1) % swapchain.frames.len();
	swapchain.current_frame_buffer_index = image_index;*/

	swapchain.current_frame_index = (swapchain.current_frame_index + 1) % swapchain.frames.len();
}

pub fn base_pass(
	cx: Res<GpuContextHandle>,
	scene: Res<GpuScene>,
	mesh: Res<CubeMesh>,
	pipeline: Res<PbrPipeline>,
	xr_swapchain: Option<Res<XrSwapchain>>,
	settings: Res<RenderSettings>,
	sampler: Res<Sampler>,
	texture: Res<Texture>,
	render_pass: Res<BasePass>,
	swapchains: Query<&Swapchain>,
) {
	let Some((frame_index, resolution, cmd, frame_buffer)) = xr_swapchain
		.map(|swapchain| (swapchain.current_frame_index, swapchain.resolution, swapchain.frame_in_flight().graphics_cmd, swapchain.swapchain_frame_buffers[swapchain.current_frame_buffer_index].frame_buffer))
		.or_else(|| swapchains.get_single()
			.ok()
			.map(|swapchain| (swapchain.current_frame_index, swapchain.resolution, swapchain.frame_in_flight().graphics_cmd, swapchain.swapchain_frame_buffers[swapchain.current_frame_buffer_index].frame_buffer))
		) else
	{
		return;
	};

	rhi_label!("base_pass", &cx, &cmd);

	unsafe {
		///
		/// BEGIN RENDER PASS
		///

		let clear_values = match settings.msaa {
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

		cx.exts.create_render_pass2.cmd_begin_render_pass2(
			cmd,
			&vk::RenderPassBeginInfo::default()
				.render_pass(render_pass.handle)
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

		cx.device.cmd_set_viewport(
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

		cx.device.cmd_set_scissor(
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

		cx.exts.push_descriptor.cmd_push_descriptor_set(
			cmd,
			vk::PipelineBindPoint::GRAPHICS,
			pipeline.layout,
			0,
			&[
				vk::WriteDescriptorSet::default()
					.dst_binding(0)
					.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
					.buffer_info(&[vk::DescriptorBufferInfo::default()
						.buffer(scene.global_ubo.handle)
						.offset((align_to(size_of::<GlobalUbo>(), cx.limits.ubo_min_align) * frame_index) as _)
						.range(size_of::<GlobalUbo>() as _)
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

		cx.device.cmd_bind_pipeline(
			cmd,
			vk::PipelineBindPoint::GRAPHICS,
			pipeline.handle,
		);

		cx.device.cmd_bind_index_buffer(
			cmd,
			mesh.index_buffer.handle,
			0,
			if mesh.index_buffer.size() > u16::MAX as usize * size_of::<u16>() {
				vk::IndexType::UINT32
			} else {
				vk::IndexType::UINT16
			},
		);

		cx.device.cmd_bind_vertex_buffers(
			cmd,
			0,
			&[mesh.vertex_buffer.handle],
			&[0],
		);

		cx.device.cmd_draw_indexed(
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

		cx.exts.create_render_pass2.cmd_end_render_pass2(
			cmd,
			&vk::SubpassEndInfo::default(),
		);
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

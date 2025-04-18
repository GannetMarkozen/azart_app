use std::ffi::CString;
use std::sync::Arc;
use azart_utils::debug_string::DebugString;
use openxr as xr;
use ash::vk;
use ash::vk::Handle;
use azart_gfx_utils::{Format, Msaa};
use bevy::math::UVec2;
use crate::GpuContext;
use crate::xr::{XrInstance, XrSession};

pub struct Swapchain {
	name: DebugString,
	pub(crate) handle: xr::Swapchain<xr::Vulkan>,
	pub(crate) frame_waiter: xr::FrameWaiter,
	pub(crate) frame_stream: xr::FrameStream<xr::Vulkan>,
	pub(crate) frames: Box<[FrameState]>,// Encapsulates swapchain images and all "frame in flight" state.
	current_image_index: usize,
	extent: UVec2,
	format: Format,
	context: Arc<GpuContext>,
}

impl Swapchain {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		frame_waiter: xr::FrameWaiter,
		frame_stream: xr::FrameStream<xr::Vulkan>,
		create_info: &SwapchainCreateInfo,
	) -> Self {
		todo!("Remove");
		let &SwapchainCreateInfo { xr, session, .. } = create_info;

		let (format, extent) = {
			assert!(xr.instance.enumerate_view_configurations(xr.hmd).unwrap().contains(&xr::ViewConfigurationType::PRIMARY_STEREO),
							"Primary stereo view configuration not supported!");

			let view_configs = xr.instance.enumerate_view_configuration_views(xr.hmd, xr::ViewConfigurationType::PRIMARY_STEREO).unwrap();
			assert!(view_configs.len() >= 2, "Primary stereo view configuration must have at least two views!");
			assert_eq!(view_configs[0], view_configs[1], "Primary stereo view configuration must have the same view configuration!");

			let view_config = view_configs[0];

			let formats = session.enumerate_swapchain_formats().unwrap();
			let format = formats
				.iter()
				.map(|&x| Format::from(vk::Format::from_raw(x as _)))
				.find(|&format| {
					let target_format = create_info.format.unwrap_or(Format::RgbaU8Srgb);
					format == target_format
				})
				.unwrap_or_else(|| Format::from(vk::Format::from_raw(formats[0] as _)));

			(format, UVec2::new(view_config.recommended_image_rect_width, view_config.recommended_image_rect_height))
		};

		let swapchain = {
			let create_info = xr::SwapchainCreateInfo::<xr::Vulkan> {
				create_flags: xr::SwapchainCreateFlags::EMPTY,
				usage_flags: create_info.usage,
				format: <_ as Into<vk::Format>>::into(format).as_raw() as _,
				sample_count: 1,
				width: extent.x,
				height: extent.y,
				face_count: 1,
				array_size: 2,// Stereo rendering.
				mip_count: 1,
			};

			session.create_swapchain(&create_info).expect("Failed to create OpenXR swapchain!")
		};

		let frames = swapchain
			.enumerate_images()
			.expect("Failed to enumerate OpenXR swapchain images!")
			.into_iter()
			.enumerate()
			.map(|(i, image)| {
				let image = vk::Image::from_raw(image);

				let image_view = {
					let create_info = vk::ImageViewCreateInfo::default()
						.image(image)
						.view_type(vk::ImageViewType::TYPE_2D)
						.format(format.into())
						.subresource_range(vk::ImageSubresourceRange::default()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.base_mip_level(0)
							.level_count(1)
							.base_array_layer(0)
							.layer_count(1)
						);

					unsafe { context.device.create_image_view(&create_info, None) }.expect("Failed to create image view!")
				};

				let graphics_command_pool = {
					let create_info = vk::CommandPoolCreateInfo::default()
						.queue_family_index(context.queue_families.graphics)
						.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

					unsafe { context.device.create_command_pool(&create_info, None) }.expect("Failed to create command pool!")
				};

				let graphics_command_buffer = {
					let create_info = vk::CommandBufferAllocateInfo::default()
						.command_pool(graphics_command_pool)
						.level(vk::CommandBufferLevel::PRIMARY)
						.command_buffer_count(1);

					unsafe { context.device.allocate_command_buffers(&create_info) }.expect("Failed to allocate command buffers!")[0]
				};

				let in_flight_fence = {
					let create_info = vk::FenceCreateInfo::default()
						.flags(vk::FenceCreateFlags::SIGNALED);

					unsafe { context.device.create_fence(&create_info, None) }.expect("Failed to create fence!")
				};

				let (image_available_semaphore, render_finished_semaphore) = {
					let create_info = vk::SemaphoreCreateInfo::default();

					unsafe {
						(
							context.device.create_semaphore(&create_info, None).expect("Failed to create semaphore!"),
							context.device.create_semaphore(&create_info, None).expect("Failed to create semaphore!"),
						)
					}
				};

				#[cfg(debug_assertions)]
				unsafe {
					context.set_debug_name(format!("{name}_image[{i}]").as_str(), image);
					context.set_debug_name(format!("{name}_image_view[{i}]").as_str(), image_view);
					context.set_debug_name(format!("{name}_graphics_command_pool[{i}]").as_str(), graphics_command_pool);
					context.set_debug_name(format!("{name}_graphics_command_buffer[{i}]").as_str(), graphics_command_buffer);
					context.set_debug_name(format!("{name}_in_flight_fence[{i}]").as_str(), in_flight_fence);
					context.set_debug_name(format!("{name}_image_available_semaphore[{i}]").as_str(), image_available_semaphore);
				}

				FrameState {
					image,
					image_view,
					graphics_command_pool,
					graphics_command_buffer,
					in_flight_fence,
					image_available_semaphore,
					render_finished_semaphore,
				}
			})
			.collect::<Box<_>>();

		/*context.immediate_cmd(context.queue_families.graphics, |cmd| {
			let barriers = frames
				.iter()
				.map(|&FrameState { image, .. }| vk_sync::ImageBarrier {
					previous_accesses: &[],
					previous_layout: vk_sync::ImageLayout::Optimal,
					next_accesses: &[vk_sync::AccessType::TransferRead],
					next_layout: vk_sync::ImageLayout::Optimal,
					discard_contents: true,
					src_queue_family_index: context.queue_families.graphics,
					dst_queue_family_index: context.queue_families.graphics,
					image,
					range: vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.base_mip_level(0)
						.base_array_layer(0)
						.level_count(vk::REMAINING_MIP_LEVELS)
						.layer_count(vk::REMAINING_ARRAY_LAYERS),
				})
				.collect::<Vec<_>>();

			context.pipeline_barrier(cmd, None, &[], &barriers);
		});*/

		/*context.immediate_cmd(context.queue_families.graphics, |cmd| {
			let barriers = frames
				.iter()
				.map(|&FrameState { image, .. }| vk::ImageMemoryBarrier::default()
					.src_access_mask(vk::AccessFlags::empty())
					.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
					.old_layout(vk::ImageLayout::UNDEFINED)
					.new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.image(image)
					.subresource_range(vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.level_count(vk::REMAINING_MIP_LEVELS)
						.layer_count(vk::REMAINING_ARRAY_LAYERS)
					)
				)
				.collect::<Vec<_>>();

			unsafe {
				context.device.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::BOTTOM_OF_PIPE,
					vk::PipelineStageFlags::TOP_OF_PIPE,
					vk::DependencyFlags::empty(),
					&[],
					&[],
					&barriers,
				);
			}
		});*/

		Self {
			name,
			handle: swapchain,
			frame_waiter,
			frame_stream,
			frames,
			current_image_index: 0,
			extent,
			format,
			context,
		}
	}

	pub fn acquire_next_image(&mut self) -> usize {
		let frame_state = self.frame_waiter.wait().expect("Failed to wait!");
		println!("Frame state {frame_state:?}");

		self.handle.wait_image(xr::Duration::INFINITE).expect("Failed to wait for next swapchain image!");
		self.current_image_index = self.handle.acquire_image().expect("Failed to acquire next swapchain image!") as usize;

		self.current_image_index
	}

	#[inline(always)]
	pub fn current_frame_index(&self) -> usize {
		self.current_image_index
	}

	#[inline(always)]
	pub fn extent(&self) -> UVec2 {
		self.extent
	}

	#[inline(always)]
	pub fn format(&self) -> Format {
		self.format
	}
}

impl Drop for Swapchain {
	fn drop(&mut self) {
		self.context.wait_idle();

		for frame in self.frames.iter() {
			unsafe {
				self.context.device.destroy_image_view(frame.image_view, None);
				self.context.device.destroy_command_pool(frame.graphics_command_pool, None);
				self.context.device.destroy_fence(frame.in_flight_fence, None);
				self.context.device.destroy_semaphore(frame.image_available_semaphore, None);
				self.context.device.destroy_semaphore(frame.render_finished_semaphore, None);
			}
		}
	}
}

pub struct SwapchainCreateInfo<'a> {
	pub xr: &'a XrInstance,
	pub session: &'a xr::Session<xr::Vulkan>,
	pub usage: xr::SwapchainUsageFlags,
	pub format: Option<Format>,
}

pub struct FrameState {
	pub image: vk::Image,
	pub image_view: vk::ImageView,
	pub graphics_command_pool: vk::CommandPool,
	pub graphics_command_buffer: vk::CommandBuffer,
	pub in_flight_fence: vk::Fence,
	pub image_available_semaphore: vk::Semaphore,
	pub render_finished_semaphore: vk::Semaphore,
}
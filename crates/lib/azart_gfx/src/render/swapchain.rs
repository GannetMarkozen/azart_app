use std::{mem, slice};
use std::sync::Arc;
use ash::vk;
use azart_gfx_utils::{Format, Msaa};
use azart_utils::dbgfmt;
use azart_utils::debug_string::DebugString;
use bevy::log::warn;
use bevy::math::UVec2;
use bevy::prelude::*;
use bevy::window::PresentMode;
use either::{for_both, Either};
use gpu_allocator::MemoryLocation;
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use crate::{GpuContext, Image, ImageCreateInfo, MipCount};
use crate::render_pass::RenderPass;

#[derive(Component)]
pub struct Swapchain {
	name: DebugString,
	context: Arc<GpuContext>,
	render_pass: Arc<RenderPass>,
	pub handle: vk::SwapchainKHR,
	pub surface: vk::SurfaceKHR,
	pub frames: Box<[Frame]>,
	pub swapchain_frame_buffers: Box<[SwapchainFrameBuffer]>,
	pub current_frame_index: usize,
	pub current_frame_buffer_index: usize,
	pub format: Format,
	pub color_space: vk::ColorSpaceKHR,
	pub resolution: UVec2,
}

impl Swapchain {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		render_pass: Arc<RenderPass>,
		display: DisplayHandle,
		window: WindowHandle,
		desc: &SwapchainDesc,
	) -> Self {
		// Hard-coded for now.
		const DEPTH_FORMAT: Format = Format::DF32;

		let surface = unsafe { ash_window::create_surface(&context.entry, &context.instance, display.as_raw(), window.as_raw(), None) }.unwrap();

		let (format, color_space) = {
			let available_formats = unsafe { context.exts.surface.get_physical_device_surface_formats(context.physical_device, surface) }.unwrap();
			assert!(!available_formats.is_empty());

			let selected_format = desc.format.unwrap_or(Format::RgbaU8Srgb);
			available_formats
				.iter()
				.map(|&x| (Format::from(x.format), x.color_space))
				.find(|&(format, color_space)| format == selected_format && (desc.color_space == None || desc.color_space == Some(color_space)))
				.unwrap_or_else(|| {
					warn!("Could not surface format combination {:?} {:?}! Using first available format: {:?} {:?}!", selected_format, desc.color_space, available_formats[0].format, available_formats[0].color_space);
					(available_formats[0].format.into(), available_formats[0].color_space)
				})
		};

		let capabilities = unsafe { context.exts.surface.get_physical_device_surface_capabilities(context.physical_device, surface) }.unwrap();
		let resolution = {
			let mut extent = match desc.resolution {
				Some(extent) => UVec2 {
					x: extent.x.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
					y: extent.y.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
				},
				None => UVec2 {
					x: capabilities.current_extent.width,
					y: capabilities.current_extent.height,
				},
			};

			if matches!(capabilities.current_transform, vk::SurfaceTransformFlagsKHR::ROTATE_90 | vk::SurfaceTransformFlagsKHR::ROTATE_270) {
				mem::swap(&mut extent.x, &mut extent.y);
			}

			extent
		};

		let swapchain = {
			assert!(desc.frames_in_flight > 0, "Image count must be greater than 0!");

			let present_modes = unsafe { context.exts.surface.get_physical_device_surface_present_modes(context.physical_device, surface) }.unwrap();
			assert!(!present_modes.is_empty());

			let present_mode = match present_modes.contains(&Self::vk_present_mode(desc.present_mode)) {
				true => Self::vk_present_mode(desc.present_mode),
				false => {
					error!("Failed to find requested present mode {:?}! Using first available mode: {:?}", desc.present_mode, present_modes[0]);
					present_modes[0]
				},
			};

			let create_info = vk::SwapchainCreateInfoKHR::default()
				.surface(surface)
				.min_image_count(desc.frames_in_flight as _)
				.image_format(format.into())
				.image_color_space(color_space)
				.image_extent(vk::Extent2D {
					width: resolution.x,
					height: resolution.y,
				})
				.image_array_layers(1)
				.image_usage(desc.usage)
				.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
				.pre_transform(capabilities.current_transform)
				.present_mode(present_mode)
				.clipped(true)
				.composite_alpha(match capabilities.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
					true => vk::CompositeAlphaFlagsKHR::OPAQUE,
					false => vk::CompositeAlphaFlagsKHR::from_raw(capabilities.supported_composite_alpha.as_raw() & capabilities.supported_composite_alpha.as_raw().wrapping_neg()),// Selects first set bit.
				})
				.old_swapchain(vk::SwapchainKHR::null());

			unsafe { context.exts.swapchain.create_swapchain(&create_info, None) }.unwrap()
		};

		let frames = (0..desc.frames_in_flight)
			.map(|i| {
				let graphics_cmd_pool = {
					let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
						.queue_family_index(context.queue_families.graphics);

					unsafe { context.device.create_command_pool(&create_info, None) }.expect("Failed to create command pool!")
				};

				let graphics_cmd = {
					let create_info = vk::CommandBufferAllocateInfo::default()
						.level(vk::CommandBufferLevel::PRIMARY)
						.command_pool(graphics_cmd_pool)
						.command_buffer_count(1);

					unsafe { context.device.allocate_command_buffers(&create_info) }.expect("Failed to allocate command buffer!")[0]
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
					context.set_debug_name(format!("{name}_graphics_cmd_pool[{i}]").as_str(), graphics_cmd_pool);
					context.set_debug_name(format!("{name}_in_flight_fence[{i}]").as_str(), in_flight_fence);
					context.set_debug_name(format!("{name}_image_available_semaphore[{i}]").as_str(), image_available_semaphore);
				}

				Frame {
					graphics_cmd_pool,
					graphics_cmd,
					in_flight_fence,
					image_available_semaphore,
					render_finished_semaphore,
				}
			})
			.collect::<Box<_>>();

		let images = unsafe { context.exts.swapchain.get_swapchain_images(swapchain) }
			.unwrap()
			.into_iter()
			.enumerate()
			.map(|(i, image)| {
				let image_view = {
					let create_info = vk::ImageViewCreateInfo::default()
						.image(image)
						.view_type(vk::ImageViewType::TYPE_2D_ARRAY)
						.format(format.into())
						.subresource_range(vk::ImageSubresourceRange::default()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.level_count(vk::REMAINING_MIP_LEVELS)
							.layer_count(vk::REMAINING_ARRAY_LAYERS)
						);

					unsafe { context.device.create_image_view(&create_info, None) }.expect("Failed to create image view!")
				};

				let msaa_color_attachment = match desc.msaa {
					Msaa::None => None,
					msaa => Some(Image::new(dbgfmt!("msaa_color_attachment[{i}]"), Arc::clone(&context), &ImageCreateInfo {
						resolution,
						mip_count: MipCount::None,
						format,
						usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
						initial_layout: vk::ImageLayout::UNDEFINED,
						tiling: vk::ImageTiling::OPTIMAL,
						msaa,
						layers: 1,
						memory: MemoryLocation::GpuOnly,
					})),
				};

				let depth_attachment = Image::new(dbgfmt!("depth_attachment[{i}]"), Arc::clone(&context), &ImageCreateInfo {
					resolution,
					mip_count: MipCount::None,
					format: DEPTH_FORMAT.into(),
					usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,// Transient if we don't care to read this buffer after rendering.
					initial_layout: vk::ImageLayout::UNDEFINED,
					tiling: vk::ImageTiling::OPTIMAL,
					msaa: desc.msaa,
					layers: 1,
					memory: MemoryLocation::GpuOnly,
				});

				let frame_buffer = {
					// Resolve attachment at index 2. Only required if MSAA is enabled. If MSAA is enabled the swapchain image
					// will be the resolve image, else it will be the color render target.
					let attachments = match &msaa_color_attachment {
						Some(msaa_color_attachment) => Either::Left([
							image_view,
							depth_attachment.view,
							msaa_color_attachment.view,
						]),
						None => Either::Right([
							image_view,
							depth_attachment.view,
						]),
					};

					let create_info = vk::FramebufferCreateInfo::default()
						.render_pass(render_pass.handle)
						.width(resolution.x)
						.height(resolution.y)
						.attachments(for_both!(&attachments, a => a.as_slice()))
						.layers(1);// Must be 1 even with multiview.

					unsafe { context.device.create_framebuffer(&create_info, None) }.expect("Failed to create frame buffer!")
				};

				#[cfg(debug_assertions)]
				unsafe {
					context.set_debug_name(format!("swapchain_image[{i}]").as_str(), image);
					context.set_debug_name(format!("swapchain_image_view[{i}]").as_str(), image_view);
					context.set_debug_name(format!("swapchain_frame_buffer[{i}]").as_str(), frame_buffer);
				}

				SwapchainFrameBuffer {
					image,
					image_view,
					msaa_color_attachment,
					depth_attachment: mem::ManuallyDrop::new(depth_attachment),
					frame_buffer,
				}
			})
			.collect::<Box<_>>();

		let present_queue_family = {
			let surface_ext = ash::khr::surface::Instance::new(&context.entry, &context.instance);
			unsafe { context.instance.get_physical_device_queue_family_properties(context.physical_device) }
				.into_iter()
				.enumerate()
				.filter(|&(i, _)| unsafe { surface_ext.get_physical_device_surface_support(context.physical_device, i as u32, surface) }.unwrap())
				.max_by(|(_, x), (_, y)| x.queue_count.cmp(&y.queue_count))
				.expect("Failed to find a queue family that supports the surface!")
				.0 as u32
		};

		#[cfg(debug_assertions)]
		unsafe {
			context.set_debug_name(format!("{name}_surface").as_str(), surface);
			context.set_debug_name(name.as_str(), swapchain);
		}

		Self {
			name,
			context,
			render_pass,
			handle: swapchain,
			surface,
			frames,
			swapchain_frame_buffers: images,
			current_frame_index: 0,
			current_frame_buffer_index: 0,
			format,
			color_space,
			resolution,
		}
	}

	#[inline(always)]
	pub fn frame_in_flight(&self) -> &Frame {
		&self.frames[self.current_frame_index]
	}

	#[inline]
	const fn vk_present_mode(present_mode: PresentMode) -> vk::PresentModeKHR {
		match present_mode {
			PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
			PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
			PresentMode::Fifo => vk::PresentModeKHR::FIFO,
			PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
			_ => vk::PresentModeKHR::FIFO,
		}
	}
}

impl Drop for Swapchain {
	fn drop(&mut self) {
		self.context.wait_idle();

		unsafe {
			for frame in self.frames.iter() {
				self.context.device.destroy_semaphore(frame.image_available_semaphore, None);
				self.context.device.destroy_semaphore(frame.render_finished_semaphore, None);
				self.context.device.destroy_fence(frame.in_flight_fence, None);
				self.context.device.destroy_command_pool(frame.graphics_cmd_pool, None);
			}

			for frame_buffer in self.swapchain_frame_buffers.iter_mut() {
				drop(frame_buffer.msaa_color_attachment.take());
				mem::ManuallyDrop::drop(&mut frame_buffer.depth_attachment);
				self.context.device.destroy_image_view(frame_buffer.image_view, None);
				self.context.device.destroy_framebuffer(frame_buffer.frame_buffer, None);
			}

			self.context.exts.swapchain.destroy_swapchain(self.handle, None);
			self.context.exts.surface.destroy_surface(self.surface, None);
		}
	}
}

pub struct Frame {
	pub graphics_cmd_pool: vk::CommandPool,
	pub graphics_cmd: vk::CommandBuffer,
	pub in_flight_fence: vk::Fence,
	pub image_available_semaphore: vk::Semaphore,
	pub render_finished_semaphore: vk::Semaphore,
}

pub struct SwapchainFrameBuffer {
	pub image: vk::Image,
	pub image_view: vk::ImageView,
	pub msaa_color_attachment: Option<Image>,
	pub depth_attachment: mem::ManuallyDrop<Image>,
	pub frame_buffer: vk::Framebuffer,
}

pub struct SwapchainDesc {
	pub resolution: Option<UVec2>,
	pub present_mode: PresentMode,
	pub format: Option<Format>,
	pub color_space: Option<vk::ColorSpaceKHR>,
	pub msaa: Msaa,
	pub usage: vk::ImageUsageFlags,
	pub frames_in_flight: usize,
}

impl Default for SwapchainDesc {
	#[inline]
	fn default() -> Self {
		Self {
			resolution: None,
			present_mode: PresentMode::Fifo,
			format: None,
			color_space: None,
			msaa: Msaa::None,
			usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
			frames_in_flight: 2,
		}
	}
}

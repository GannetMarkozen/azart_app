use std::{mem, slice};
use azart_utils::debug_string::DebugString;
use std::sync::Arc;
use bevy::prelude::*;
use ash::vk;
use openxr as xr;
use bevy::window::PresentMode;
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use crate::context::GpuContext;
use crate::render_settings::DisplayMode;

// Represents everything required to submit commands and render to a surface.
#[derive(Component)]
pub struct Swapchain {
	name: DebugString,
	pub(crate) handle: vk::SwapchainKHR,
	pub(crate) surface: vk::SurfaceKHR,
	frames: Box<[FrameState]>,
	images: Box<[SwapchainImage]>,
	pub current_frame_index: FrameIndex,
	pub current_image_index: SwapchainImageIndex,
	pub present_queue_family: u32,
	extent: UVec2,
	format: vk::Format,
	color_space: vk::ColorSpaceKHR,
	present_mode: PresentMode,
	image_usage_flags: vk::ImageUsageFlags,
	current_transform: vk::SurfaceTransformFlagsKHR,
	pub(crate) context: Arc<GpuContext>,
}

impl Swapchain {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		display_handle: DisplayHandle,
		window_handle: WindowHandle,
		create_info: &SwapchainCreateInfo,
	) -> Self {
		let surface = unsafe { ash_window::create_surface(&context.entry, &context.instance, display_handle.as_raw(), window_handle.as_raw(), None).unwrap() };

		let (format, color_space) = {
			let available_formats = unsafe { context.extensions.surface.get_physical_device_surface_formats(context.physical_device, surface) }.unwrap();
			assert!(!available_formats.is_empty());

			let selected_format = create_info.format.unwrap_or(vk::Format::B8G8R8A8_SRGB);
			available_formats
				.iter()
				.map(|&x| (x.format, x.color_space))
				.find(|&(format, color_space)| format == selected_format && (create_info.color_space == None || create_info.color_space == Some(color_space)))
				.unwrap_or_else(|| {
					warn!("Could not surface format combination {:?} {:?}! Using first available format: {:?} {:?}!", selected_format, create_info.color_space, available_formats[0].format, available_formats[0].color_space);
					(available_formats[0].format, available_formats[0].color_space)
				})
		};

		let capabilities = unsafe { context.extensions.surface.get_physical_device_surface_capabilities(context.physical_device, surface) }.unwrap();
		let extent = {
			let mut extent = match create_info.extent {
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
			assert!(create_info.image_count > 0, "Image count must be greater than 0!");

			let create_info = vk::SwapchainCreateInfoKHR::default()
				.surface(surface)
				.min_image_count(create_info.image_count)
				.image_format(format)
				.image_color_space(color_space)
				.image_extent(vk::Extent2D {
					width: extent.x,
					height: extent.y,
				})
				.image_array_layers(1)
				.image_usage(create_info.usage)
				.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
				.pre_transform(capabilities.current_transform)
				.present_mode(Self::vk_present_mode(create_info.present_mode))
				.clipped(true)
				.composite_alpha(match capabilities.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
					true => vk::CompositeAlphaFlagsKHR::OPAQUE,
					false => vk::CompositeAlphaFlagsKHR::from_raw(capabilities.supported_composite_alpha.as_raw() & capabilities.supported_composite_alpha.as_raw().wrapping_neg()),// Selects first set bit.
				})
				.old_swapchain(vk::SwapchainKHR::null());

			unsafe { context.extensions.swapchain.create_swapchain(&create_info, None) }.unwrap()
		};
		
		let frames = (0..create_info.image_count)
			.map(|i| {
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
					context.set_debug_name(format!("{name}_graphics_command_pool[{i}]").as_str(), graphics_command_pool);
					context.set_debug_name(format!("{name}_graphics_command_buffer[{i}]").as_str(), graphics_command_buffer);
					context.set_debug_name(format!("{name}_in_flight_fence[{i}]").as_str(), in_flight_fence);
					context.set_debug_name(format!("{name}_image_available_semaphore[{i}]").as_str(), image_available_semaphore);
				}

				FrameState {
					graphics_command_pool,
					graphics_command_buffer,
					in_flight_fence,
					image_available_semaphore,
					render_finished_semaphore,
				}
			})
			.collect::<Box<_>>();

		let images = unsafe { context.extensions.swapchain.get_swapchain_images(swapchain) }
			.unwrap()
			.into_iter()
			.enumerate()
			.map(|(i, image)| {
				let image_view = {
					let create_info = vk::ImageViewCreateInfo::default()
						.image(image)
						.view_type(vk::ImageViewType::TYPE_2D)
						.format(format)
						.subresource_range(vk::ImageSubresourceRange::default()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.base_mip_level(0)
							.level_count(1)
							.base_array_layer(0)
							.layer_count(1)
						);

					unsafe { context.device.create_image_view(&create_info, None) }.expect("Failed to create image view!")
				};
				
				#[cfg(debug_assertions)]
				unsafe {
					context.set_debug_name(format!("{name}_image[{i}]").as_str(), image);
					context.set_debug_name(format!("{name}_image_view[{i}]").as_str(), image_view);
				}
				
				SwapchainImage {
					image,
					image_view,
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
			handle: swapchain,
			surface,
			frames,
			images,
			current_frame_index: FrameIndex(0),
			current_image_index: SwapchainImageIndex(0),
			present_queue_family,
			extent,
			format,
			color_space,
			present_mode: create_info.present_mode,
			image_usage_flags: create_info.usage,
			current_transform: capabilities.current_transform,
			context,
		}
	}

	// Returns the new frame index (reflects the value of self.current_frame_index) and whether the surface is suboptimal for the swapchain (the swapchain should be recreated but it's still renderable to).
	pub fn acquire_next_image(&mut self) -> (FrameIndex, SwapchainImageIndex, bool) {
		
		let start = std::time::Instant::now();
		
		// Wait for frame to finish rendering.
		unsafe { self.context.device.wait_for_fences(slice::from_ref(&self.frames[self.current_frame_index.0].in_flight_fence), true, u64::MAX) }.unwrap();
		
		let start = std::time::Instant::now();

		let result = unsafe { self.context.extensions.swapchain.acquire_next_image(self.handle, u64::MAX, self.frames[self.current_frame_index.0].image_available_semaphore, vk::Fence::null()) };

		assert_ne!(result, Err(vk::Result::SUBOPTIMAL_KHR));

		let (index, suboptimal) = match result {
			Ok((index, suboptimal)) => (index, suboptimal),
			Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
				info!("Swapchain is out of date during acquire_next_image. Recreating...");

				// Wait for the GPU to go idle before doing anything.
				// @NOTE: This is very expensive...
				self.context.wait_idle();

				self.recreate(None, None);

				unsafe { self.context.extensions.swapchain.acquire_next_image(self.handle, u64::MAX, self.frames[self.current_frame_index.0].image_available_semaphore, vk::Fence::null()) }
					.expect("Failed to acquire next image event after recreating swapchain!")
			},
			Err(e) => panic!("Failed to acquire next image. Error: {:?}", e),
		};

		// Reset fence only after successfully acquiring the next image.
		unsafe { self.context.device.reset_fences(slice::from_ref(&self.frames[self.current_frame_index.0].in_flight_fence)) }.unwrap();
		
		self.current_frame_index = FrameIndex((self.current_frame_index.0 + 1) % self.frames.len());
		self.current_image_index = SwapchainImageIndex(index as usize);

		(self.current_frame_index, self.current_image_index, suboptimal)
	}

	// If extent is None it will be resized to match the surface. Returns whether a resize actually occured or not (no resize if the extent matches the current extent).
	pub fn recreate(&mut self, extent: Option<UVec2>, present_mode: Option<PresentMode>) -> bool {
		assert_ne!(self.surface, vk::SurfaceKHR::null());

		let capabilities = unsafe { self.context.extensions.surface.get_physical_device_surface_capabilities(self.context.physical_device, self.surface) }.unwrap();
		let extent = {
			let mut extent = match extent {
				Some(extent) => UVec2 {
					x: extent.x.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
					y: extent.y.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
				},
				None => UVec2 {
					x: capabilities.current_extent.width.max(1),
					y: capabilities.current_extent.height.max(1),
				}
			};

			if matches!(capabilities.current_transform, vk::SurfaceTransformFlagsKHR::ROTATE_90 | vk::SurfaceTransformFlagsKHR::ROTATE_270) {
				mem::swap(&mut extent.x, &mut extent.y);
			}

			extent
		};

		if self.extent == extent || matches!(present_mode, Some(x) if x != self.present_mode) {
			return false;
		};

		// Update self's present_mode.
		if let Some(x) = present_mode {
			self.present_mode = x;
		}

		// Wait for all work being done to finish before recreation.
		// NOTE: This can be avoided by keeping around the old swapchain for the duration it's used.
		let frame_in_flight_fences = self
			.frames
			.iter()
			.map(|frame| frame.in_flight_fence)
			.collect::<Vec<_>>();

		trace!("Waiting for all fences to finish before recreating swapchain...");
		unsafe { self.context.device.wait_for_fences(&frame_in_flight_fences, true, u64::MAX) }.unwrap();

		// Create a new swapchain replacing the old one.
		let new_swapchain = {
			let create_info = vk::SwapchainCreateInfoKHR::default()
				.surface(self.surface)
				.min_image_count(self.images.len() as u32)
				.image_format(self.format)
				.image_color_space(self.color_space)
				.image_extent(vk::Extent2D {
					width: extent.x,
					height: extent.y,
				})
				.image_array_layers(1)
				.image_usage(self.image_usage_flags)
				.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
				.pre_transform(capabilities.current_transform)
				.present_mode(Self::vk_present_mode(self.present_mode))
				.clipped(true)
				.composite_alpha(match capabilities.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
					true => vk::CompositeAlphaFlagsKHR::OPAQUE,
					false => vk::CompositeAlphaFlagsKHR::from_raw(capabilities.supported_composite_alpha.as_raw() & capabilities.supported_composite_alpha.as_raw().wrapping_neg()),// Selects first set bit.
				})
				.old_swapchain(self.handle);

			unsafe { self.context.extensions.swapchain.create_swapchain(&create_info, None) }.expect("failed to create swapchain!")
		};

		let images = unsafe { self.context.extensions.swapchain.get_swapchain_images(new_swapchain) }.unwrap();
		assert_eq!(images.len(), self.images.len());

		// Destroy image views for old swapchain before destroying the old swapchain.
		for i in 0..self.images.len() {
			let frame = &mut self.images[i];

			// Destroy image view for old swapchain image.
			unsafe { self.context.device.destroy_image_view(frame.image_view, None) };
			
			// Assign new image and image views.
			frame.image = images[i];
			frame.image_view = {
				let create_info = vk::ImageViewCreateInfo::default()
					.image(images[i])
					.view_type(vk::ImageViewType::TYPE_2D)
					.format(self.format)
					.subresource_range(vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.base_mip_level(0)
						.level_count(1)
						.base_array_layer(0)
						.layer_count(1)
					);

				unsafe { self.context.device.create_image_view(&create_info, None) }.unwrap()
			};
		}

		// Destroy old swapchain.
		unsafe { self.context.extensions.swapchain.destroy_swapchain(self.handle, None); }

		self.handle = new_swapchain;
		self.extent = extent;
		self.current_frame_index = FrameIndex(0);
		self.current_image_index = SwapchainImageIndex(0);
		self.current_transform = capabilities.current_transform;

		true
	}

	#[inline(always)]
	const fn vk_present_mode(present_mode: PresentMode) -> vk::PresentModeKHR {
		match present_mode {
			PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
			PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
			PresentMode::Fifo => vk::PresentModeKHR::FIFO,
			PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
			_ => vk::PresentModeKHR::FIFO,
		}
	}

	pub fn context(&self) -> &GpuContext {
		&*self.context
	}
	
	pub fn images(&self) -> &[SwapchainImage] {
		&self.images
	}

	pub fn frames(&self) -> &[FrameState] {
		&self.frames
	}

	pub fn format(&self) -> vk::Format {
		self.format
	}
	
	pub fn color_space(&self) -> vk::ColorSpaceKHR {
		self.color_space
	}
	
	pub fn extent(&self) -> UVec2 {
		self.extent
	}
	
	pub fn present_mode(&self) -> PresentMode {
		self.present_mode
	}

	#[inline(always)]
	pub fn current_transform(&self) -> vk::SurfaceTransformFlagsKHR {
		self.current_transform }
}

impl Drop for Swapchain {
	fn drop(&mut self) {
		unsafe {
			let frame_in_flight_fences = self
				.frames
				.iter()
				.map(|frame| frame.in_flight_fence)
				.collect::<Vec<_>>();
			
			let device = &self.context.device;
			device.wait_for_fences(&frame_in_flight_fences, true, u64::MAX).unwrap();
			
			for image in self.images.iter() {
				device.destroy_image_view(image.image_view, None);
			}
			
			for frame in self.frames.iter() {
				device.free_command_buffers(frame.graphics_command_pool, slice::from_ref(&frame.graphics_command_buffer));
				device.destroy_command_pool(frame.graphics_command_pool, None);
				
				device.destroy_fence(frame.in_flight_fence, None);
				device.destroy_semaphore(frame.image_available_semaphore, None);
				device.destroy_semaphore(frame.render_finished_semaphore, None);
			}
			
			self.context.extensions.swapchain.destroy_swapchain(self.handle, None);
			self.context.extensions.surface.destroy_surface(self.surface, None);
		}
	}
}

pub struct SwapchainCreateInfo {
	pub extent: Option<UVec2>,
	pub image_count: u32,
	pub present_mode: PresentMode,
	pub format: Option<vk::Format>,
	pub color_space: Option<vk::ColorSpaceKHR>,
	pub usage: vk::ImageUsageFlags,
	pub display_mode: Option<DisplayMode>,// If None, will be selected by whether OpenXR is active or not.
}

impl Default for SwapchainCreateInfo {
	fn default() -> Self {
		Self {
			extent: None,
			image_count: 2,
			present_mode: PresentMode::Fifo,
			format: None,
			color_space: None,
			usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
			display_mode: None,
		}
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FrameIndex(pub usize);

impl From<usize> for FrameIndex {
	fn from(i: usize) -> Self {
		Self(i)
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SwapchainImageIndex(pub usize);

impl From<usize> for SwapchainImageIndex {
	fn from(i: usize) -> Self {
		Self(i)
	}
}

pub struct FrameState {
	pub graphics_command_pool: vk::CommandPool,
	pub graphics_command_buffer: vk::CommandBuffer,
	pub in_flight_fence: vk::Fence,
	pub image_available_semaphore: vk::Semaphore,
	pub render_finished_semaphore: vk::Semaphore,
}

pub struct SwapchainImage {
	pub image: vk::Image,
	pub image_view: vk::ImageView,
}

enum SwapchainState<'a> {
	Standard(SwapchainStandardState),
	Xr(SwapchainXrState<'a>),
}

struct SwapchainStandardState {
	handle: vk::SwapchainKHR,
	images: Box<[vk::Image]>,
}

struct SwapchainXrState<'a> {
	handle: xr::Swapchain<xr::Vulkan>,
	images: Box<[xr::SwapchainSubImage<'a, xr::Vulkan>]>,
}
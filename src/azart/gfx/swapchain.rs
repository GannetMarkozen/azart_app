use std::ffi::CString;
use std::num::NonZero;
use std::sync::Arc;
use crate::azart::gfx::context::GpuContext;
use ash::vk;
use bevy::math::UVec2;
use bevy::prelude::*;
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use cstr::cstr;
use crate::azart::utils::debug_string::DebugString;

pub struct Swapchain {
	name: DebugString,
	pub(crate) handle: vk::SwapchainKHR,
	pub(crate) surface: vk::SurfaceKHR,
	pub(crate) images: Vec<vk::Image>,
	pub(crate) image_views: Vec<vk::ImageView>,
	extent: UVec2,
	format: vk::Format,
	color_space: vk::ColorSpaceKHR,
	present_mode: vk::PresentModeKHR,
	pub(crate) context: Arc<GpuContext>,
}

impl Swapchain {
	pub fn new(name: DebugString, context: Arc<GpuContext>, display_handle: DisplayHandle, window_handle: WindowHandle, create_info: &SwapchainCreateInfo) -> Self {
		assert!(create_info.image_count > 0, "Invalid image count {}!", create_info.image_count);

		unsafe {
			let surface = ash_window::create_surface(&context.entry, &context.instance, display_handle.as_raw(), window_handle.as_raw(), None).expect("failed to create surface!");

			let (format, color_space) = match create_info.format {
				Some(vk::SurfaceFormatKHR { format, color_space }) => (format, color_space),
				None => {
					let formats = context.extensions.surface.get_physical_device_surface_formats(context.physical_device, surface).expect("Failed to get surface formats!");
					assert!(!formats.is_empty(), "No surface formats available!");

					let first_format = formats[0];
					let vk::SurfaceFormatKHR { format, color_space } = formats
						.into_iter()
						.find(|&x| x.format == vk::Format::B8G8R8A8_SRGB)
						.unwrap_or(first_format);

					(format, color_space)
				},
			};

			let extent = {
				let capabilities = context.extensions.surface.get_physical_device_surface_capabilities(context.physical_device, surface).expect("failed to get surface capabilities!");
				match create_info.extent {
					Some(extent) => UVec2 {
						x: extent.x.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
						y: extent.y.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
					},
					None => UVec2 {
						x: capabilities.current_extent.width,
						y: capabilities.current_extent.height,
					},
				}
			};

			let swapchain = {
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
					.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
					.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
					.pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
					.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
					.present_mode(create_info.present_mode)
					.clipped(true)
					.old_swapchain(vk::SwapchainKHR::null());

				context.extensions.swapchain.create_swapchain(&create_info, None).expect("failed to create swapchain!")
			};

			let images = context.extensions.swapchain.get_swapchain_images(swapchain).expect("failed to get swapchain images!");
			let image_views = images
				.iter()
				.map(|&x| {
					let create_info = vk::ImageViewCreateInfo::default()
						.image(x)
						.view_type(vk::ImageViewType::TYPE_2D)
						.format(format)
						.subresource_range(vk::ImageSubresourceRange::default()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.base_mip_level(0)
							.level_count(1)
							.base_array_layer(0)
							.layer_count(1)
						);

					context.device.create_image_view(&create_info, None).expect("Failed to create image view!")
				})
				.collect::<Vec<_>>();
			
			#[cfg(debug_assertions)]
			{
				context.set_debug_name_for_gpu_resource(format!("{name}_surface").as_str(), surface);
				context.set_debug_name_for_gpu_resource(name.as_str(), swapchain);
				
				assert_eq!(images.len(), image_views.len());
				for i in 0..images.len() {
					context.set_debug_name_for_gpu_resource(format!("{name}_image[{i}]").as_str(), images[i]);
					context.set_debug_name_for_gpu_resource(format!("{name}_image_view[{i}]").as_str(), image_views[i]);
				}
			}

			Self {
				name,
				handle: swapchain,
				surface,
				images,
				image_views,
				extent,
				format,
				color_space,
				present_mode: create_info.present_mode,
				context,
			}
		}
	}

	pub fn acquire_next_image(&mut self, semaphore: Option<vk::Semaphore>, fence: Option<vk::Fence>, timeout: Option<u64>) -> Result<(usize, bool), vk::Result> {
		unsafe {
			match self.context.extensions.swapchain.acquire_next_image(self.handle, timeout.unwrap_or(u64::MAX), semaphore.unwrap_or(vk::Semaphore::null()), fence.unwrap_or(vk::Fence::null())) {
				Ok((index, optimal)) => Ok((index as usize, optimal)),
				Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
					info!("Swapchain out of date! Recreating...");
					
					self.resize(None);
					self.acquire_next_image(semaphore, fence, timeout)
				},
				Err(e) => Err(e),
			}
		}
	}

	pub fn image(&self, index: usize) -> vk::Image {
		self.images[index]
	}

	pub fn image_view(&self, index: usize) -> vk::ImageView {
		self.image_views[index]
	}

	// Returns false if no resize occured.
	pub fn resize(&mut self, extent: Option<UVec2>) -> bool {
		assert_ne!(self.surface, vk::SurfaceKHR::null(), "Swapchain has no surface!");

		unsafe {
			let capabilities = self.context.extensions.surface.get_physical_device_surface_capabilities(self.context.physical_device, self.surface).expect("Failed to get surface capabilities!");

			assert!(capabilities.current_extent.width != u32::MAX && capabilities.current_extent.height != u32::MAX, "Unhandled!");

			let new_extent = match extent {
				Some(extent) => UVec2 {
					x: extent.x.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
					y: extent.y.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
				},
				None => UVec2 {
					x: capabilities.max_image_extent.width.max(1),
					y: capabilities.max_image_extent.height.max(1),
				}
			};


			// Extents match. No need to resize.
			if self.extent == new_extent {
				return false;
			}

			// Create a new swapchain replacing the old one.
			let new_handle = {
				let create_info = vk::SwapchainCreateInfoKHR::default()
					.surface(self.surface)
					.min_image_count(self.image_views.len() as u32)
					.image_format(self.format)
					.image_color_space(self.color_space)
					.image_extent(vk::Extent2D {
						width: new_extent.x,
						height: new_extent.y,
					})
					.image_array_layers(1)
					.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
					.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
					.pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
					.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
					.present_mode(self.present_mode)
					.clipped(true)
					.old_swapchain(self.handle);

				self.context.extensions.swapchain.create_swapchain(&create_info, None).expect("failed to create swapchain!")
			};

			// Destroy image views. They will be recreated with the new swapchain.
			for &image_view in self.image_views.iter() {
				self.context.device.destroy_image_view(image_view, None);
			}

			// Destroy old swapchain.
			self.context.extensions.swapchain.destroy_swapchain(self.handle, None);

			// Set new swapchain.
			self.handle = new_handle;

			// Assign images and create image views for new swapchain.
			self.images = self.context.extensions.swapchain.get_swapchain_images(self.handle).unwrap();
			self.image_views = self.images
				.iter()
				.map(|&x| {
					let create_info = vk::ImageViewCreateInfo::default()
						.image(x)
						.view_type(vk::ImageViewType::TYPE_2D)
						.format(self.format)
						.subresource_range(vk::ImageSubresourceRange::default()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.base_mip_level(0)
							.level_count(1)
							.base_array_layer(0)
							.layer_count(1)
						);

					self.context.device.create_image_view(&create_info, None).expect("Failed to create image view!")
				})
				.collect();

			self.extent = new_extent;

			#[cfg(debug_assertions)]
			{
				let name = &self.name;
				self.context.set_debug_name_for_gpu_resource(name.as_str(), self.handle);

				assert_eq!(self.images.len(), self.image_views.len());
				for i in 0..self.images.len() {
					self.context.set_debug_name_for_gpu_resource(format!("{name}_image[{i}]").as_str(), self.images[i]);
					self.context.set_debug_name_for_gpu_resource(format!("{name}_image_view[{i}]").as_str(), self.image_views[i]);
				}
			}

			true
		}
	}
	
	pub fn recreate(&mut self) {
		self.resize(None);
	}
	
	pub const fn format(&self) -> vk::Format {
		self.format
	}
	
	pub const fn extent(&self) -> UVec2 {
		self.extent
	}
}

impl Drop for Swapchain {
	fn drop(&mut self) {
		unsafe {
			for image_view in self.image_views.iter() {
				self.context.device.destroy_image_view(*image_view, None);
			}

			self.context.extensions.swapchain.destroy_swapchain(self.handle, None);
			self.context.extensions.surface.destroy_surface(self.surface, None);
		}
	}
}

pub struct SwapchainCreateInfo {
	pub extent: Option<UVec2>,// Set to None and it will be selected for you based off of the surface.
	pub image_count: u32,
	pub format: Option<vk::SurfaceFormatKHR>,// If none is selected one will be selected for you
	pub present_mode: vk::PresentModeKHR,
}

impl Default for SwapchainCreateInfo {
	fn default() -> Self {
		Self {
			extent: None,
			image_count: 2,
			format: None,
			present_mode: vk::PresentModeKHR::FIFO,
		}
	}
}
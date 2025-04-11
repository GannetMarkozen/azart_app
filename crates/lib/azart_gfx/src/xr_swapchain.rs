use std::sync::Arc;
use azart_utils::debug_string::DebugString;
use openxr as xr;
use ash::vk;
use ash::vk::Handle;
use azart_gfx_utils::{Format, MsaaCount};
use bevy::math::UVec2;
use crate::GpuContext;
use crate::xr::{XrInstance, XrSession};

pub struct Swapchain {
	name: DebugString,
	pub(crate) handle: xr::Swapchain<xr::Vulkan>,
	pub(crate) session: xr::Session<xr::Vulkan>,
	pub(crate) frame_waiter: xr::FrameWaiter,
	pub(crate) frame_stream: xr::FrameStream<xr::Vulkan>,
	pub(crate) images: Box<[SwapchainImage]>,
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
		let &SwapchainCreateInfo { xr, session, .. } = create_info;

		let (format, extent) = {
			let view_configs = xr.instance.enumerate_view_configuration_views(xr.hmd, xr::ViewConfigurationType::PRIMARY_STEREO).unwrap();
			println!("view_configs: {view_configs:?}");

			let view_config = view_configs[0];

			let formats = session.handle.enumerate_swapchain_formats().unwrap();
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
				create_flags: xr::SwapchainCreateFlags::PROTECTED_CONTENT,
				usage_flags: create_info.usage,
				format: <_ as Into<vk::Format>>::into(format).as_raw() as _,
				sample_count: create_info.msaa.as_u32(),
				width: extent.x,
				height: extent.y,
				face_count: 1,
				array_size: 1,
				mip_count: 1,
			};

			session.handle.create_swapchain(&create_info).expect("Failed to create OpenXR swapchain!")
		};

		#[cfg(debug_assertions)]
		unsafe {
			context.set_debug_name(name.as_str(), vk::SwapchainKHR::from_raw(swapchain.as_raw().into_raw()));
		}

		let images = swapchain
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

		Self {
			name,
			handle: swapchain,
			frame_waiter,
			frame_stream,
			session: session.handle.clone(),
			images,
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
}

impl Drop for Swapchain {
	fn drop(&mut self) {
		for &SwapchainImage { image_view, .. } in &self.images {
			unsafe { self.context.device.destroy_image_view(image_view, None) };
		}
	}
}

pub struct SwapchainCreateInfo<'a> {
	pub xr: &'a XrInstance,
	pub session: &'a XrSession,
	pub usage: xr::SwapchainUsageFlags,
	pub format: Option<Format>,
	pub msaa: MsaaCount,
}

pub struct SwapchainImage {
	pub image: vk::Image,
	pub image_view: vk::ImageView,
}
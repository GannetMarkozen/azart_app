use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::sync::Arc;
use ash::vk;
use bevy::math::UVec2;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use crate::azart::gfx::GpuContext;
use crate::azart::utils::debug_string::DebugString;

pub struct Image {
	pub(crate) handle: vk::Image,
	pub(crate) view: vk::ImageView,
	pub(crate) allocation: ManuallyDrop<Allocation>,
	pub(crate) resolution: UVec2,
	pub(crate) format: vk::Format,
	pub(crate) usage: vk::ImageUsageFlags,
	pub(crate) layout: vk::ImageLayout,
	context: Arc<GpuContext>,
}

impl Image {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		create_info: &ImageCreateInfo,
	) -> Self {
		let max_mip_levels = create_info.resolution.max_element().ilog2() + 1;
		let mip_levels = match create_info.mip_count {
			Some(x) => x.get().min(max_mip_levels),
			None => max_mip_levels,
		};

		let image = {
			let flags = if create_info.array_layers == 1 { vk::ImageCreateFlags::empty() } else { vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE };

			assert!(create_info.resolution.x > 0 && create_info.resolution.y > 0, "Resolution was {:?}! Must be greater than 0!", create_info.resolution);

			assert!(
				!matches!(unsafe { context.instance.get_physical_device_image_format_properties(context.physical_device, create_info.format, vk::ImageType::TYPE_2D, create_info.tiling, create_info.usage, flags) },
				Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)),
				"Image \"{name}\" is not supported with: format: {:?}, tiling: {:?}, usage: {:?}.", create_info.format, create_info.tiling, create_info.usage
			);

			let create_info = vk::ImageCreateInfo::default()
				.image_type(vk::ImageType::TYPE_2D)
				.extent(vk::Extent3D::default()
					.width(create_info.resolution.x)
					.height(create_info.resolution.y)
					.depth(1)
				)
				.mip_levels(mip_levels)
				.array_layers(1)
				.format(create_info.format)
				.flags(flags)
				.samples(vk::SampleCountFlags::TYPE_1)
				.tiling(create_info.tiling)
				.usage(create_info.usage)
				.sharing_mode(vk::SharingMode::EXCLUSIVE)
				.initial_layout(create_info.initial_layout);

			unsafe { context.device.create_image(&create_info, None) }.expect("failed to create image!")
		};

		let allocation = {
			let create_info = AllocationCreateDesc {
				name: name.as_str(),
				requirements: unsafe { context.device.get_image_memory_requirements(image) },
				location: create_info.memory,
				linear: true,
				allocation_scheme: AllocationScheme::GpuAllocatorManaged,
			};

			context.allocator.lock().unwrap().allocate(&create_info).expect("failed to allocate image memory!")
		};

		unsafe { context.device.bind_image_memory(image, allocation.memory(), allocation.offset()) }.expect("failed to bind image memory!");

		let image_view = {
			let create_info = vk::ImageViewCreateInfo::default()
				.image(image)
				.view_type(if create_info.array_layers == 1 { vk::ImageViewType::TYPE_2D } else { vk::ImageViewType::TYPE_2D_ARRAY })
				.format(create_info.format)
				.subresource_range(vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_mip_level(0)
					.level_count(mip_levels)
					.base_array_layer(0)
					.layer_count(create_info.array_layers)
				);

			unsafe { context.device.create_image_view(&create_info, None) }.expect("failed to create image view!")
		};
		
		#[cfg(debug_assertions)]
		unsafe {
			context.set_debug_name_for_gpu_resource(name.as_str(), image);
			context.set_debug_name_for_gpu_resource(format!("{name}_view").as_str(), image_view);
		}

		Self {
			handle: image,
			view: image_view,
			allocation: ManuallyDrop::new(allocation),
			resolution: create_info.resolution,
			format: create_info.format,
			usage: create_info.usage,
			layout: create_info.initial_layout,
			context,
		}
	}
}

impl Drop for Image {
	fn drop(&mut self) {
		unsafe {
			self.context.device.destroy_image_view(self.view, None);
			self.context.dealloc(ManuallyDrop::take(&mut self.allocation));
			self.context.device.destroy_image(self.handle, None);
		}
	}
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct ImageCreateInfo {
	pub resolution: UVec2,
	pub mip_count: Option<NonZero<u32>>,// If None the mip count will be decided based on the image extent.
	pub format: vk::Format,
	pub usage: vk::ImageUsageFlags,
	pub initial_layout: vk::ImageLayout,
	pub tiling: vk::ImageTiling,
	pub array_layers: u32,
	pub memory: MemoryLocation,
}

impl Default for ImageCreateInfo {
	fn default() -> Self {
		Self {
			resolution: UVec2::new(1, 1),
			mip_count: Some(NonZero::new(1).unwrap()),
			format: vk::Format::R8G8B8A8_SRGB,
			usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
			initial_layout: vk::ImageLayout::UNDEFINED,
			tiling: vk::ImageTiling::LINEAR,
			array_layers: 1,
			memory: MemoryLocation::GpuOnly,
		}
	}
}
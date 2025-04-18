use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::sync::Arc;
use ash::vk;
use bevy::math::UVec2;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use crate::GpuContext;
use azart_gfx_utils::{Msaa, GpuResource, Format};
use azart_utils::debug_string::DebugString;
use bevy::reflect::Reflect;

pub struct Image {
	name: DebugString,
	pub(crate) handle: vk::Image,
	pub(crate) view: vk::ImageView,
	pub(crate) allocation: ManuallyDrop<Allocation>,
	pub(crate) resolution: UVec2,
	layers: u32,
	mips: u32,
	pub(crate) format: Format,
	pub(crate) usage: vk::ImageUsageFlags,
	pub(crate) layout: vk::ImageLayout,
	pub(crate) msaa: Msaa,
	context: Arc<GpuContext>,
}

impl Image {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		create_info: &ImageCreateInfo,
	) -> Self {
		assert!(create_info.resolution.x > 0 && create_info.resolution.y > 0, "Resolution was {:?}! Must be greater than 0!", create_info.resolution);
		assert!(create_info.layers > 0, "Layers was {:?}! Must be greater than 0!", create_info.layers);

		let max_mip_levels = create_info.resolution.max_element().ilog2() + 1;
		let mip_levels = match create_info.mip_count {
			MipCount::None => 1,
			MipCount::Max => max_mip_levels,
			MipCount::Custom(mip_levels) => mip_levels.clamp(1, max_mip_levels),
		};

		let image = {
			assert!(
				!matches!(unsafe { context.instance.get_physical_device_image_format_properties(context.physical_device, create_info.format.into(), vk::ImageType::TYPE_2D, create_info.tiling, create_info.usage, vk::ImageCreateFlags::empty()) },
				Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)),
				"Image \"{name}\" is not supported with: {:?}", create_info,
			);

			let create_info = vk::ImageCreateInfo::default()
				.image_type(vk::ImageType::TYPE_2D)
				.extent(vk::Extent3D::default()
					.width(create_info.resolution.x)
					.height(create_info.resolution.y)
					.depth(1)
				)
				.mip_levels(mip_levels)
				.array_layers(create_info.layers)
				.format(create_info.format.into())
				.flags(vk::ImageCreateFlags::empty())
				.samples(create_info.msaa.as_vk_sample_count())
				.tiling(create_info.tiling)
				.usage(create_info.usage)
				.sharing_mode(vk::SharingMode::EXCLUSIVE)
				.initial_layout(create_info.initial_layout);

			unsafe { context.device.create_image(&create_info, None) }.expect("failed to create image!")
		};

		let allocation = {
			let (requirements, dedicated_allocation) = {
				let info = vk::ImageMemoryRequirementsInfo2::default()
					.image(image);

				let mut requirements = vk::MemoryRequirements2::default();
				let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
				requirements.push_next(&mut dedicated_requirements);

				unsafe { context.device.get_image_memory_requirements2(&info, &mut requirements); }
				(requirements.memory_requirements, dedicated_requirements.prefers_dedicated_allocation == vk::TRUE)
			};

			let allocation_scheme = match dedicated_allocation {
				true => AllocationScheme::DedicatedImage(image),
				false => AllocationScheme::GpuAllocatorManaged,
			};

			let create_info = AllocationCreateDesc {
				name: name.as_str(),
				requirements,
				location: create_info.memory,
				linear: create_info.tiling == vk::ImageTiling::LINEAR,
				allocation_scheme,
			};

			context.alloc(&create_info)
		};

		unsafe { context.device.bind_image_memory(image, allocation.memory(), allocation.offset()) }.expect("failed to bind image memory!");

		let image_view = {
			let create_info = vk::ImageViewCreateInfo::default()
				.image(image)
				.view_type(if create_info.layers == 1 { vk::ImageViewType::TYPE_2D } else { vk::ImageViewType::TYPE_2D_ARRAY })
				.format(create_info.format.into())
				.subresource_range(vk::ImageSubresourceRange::default()
					.aspect_mask(match create_info.format.into() {
						vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D16_UNORM => vk::ImageAspectFlags::DEPTH,
						vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
						_ => vk::ImageAspectFlags::COLOR,
					})
					.base_mip_level(0)
					.level_count(mip_levels)
					.base_array_layer(0)
					.layer_count(create_info.layers)
				);

			unsafe { context.device.create_image_view(&create_info, None) }.expect("failed to create image view!")
		};
		
		#[cfg(debug_assertions)]
		unsafe {
			context.set_debug_name(name.as_str(), image);
			context.set_debug_name(format!("{name}_view").as_str(), image_view);
		}

		Self {
			name,
			handle: image,
			view: image_view,
			allocation: ManuallyDrop::new(allocation),
			resolution: create_info.resolution,
			format: create_info.format,
			layers: create_info.layers,
			mips: mip_levels,
			usage: create_info.usage,
			layout: create_info.initial_layout,
			msaa: create_info.msaa,
			context,
		}
	}
	
	#[inline(always)]
	pub fn name(&self) -> &DebugString {
		&self.name
	}

	#[inline(always)]
	pub fn layers(&self) -> u32 {
		self.layers
	}

	#[inline(always)]
	pub fn mips(&self) -> u32 {
		self.mips
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

impl GpuResource for Image {}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct ImageCreateInfo {
	pub resolution: UVec2,
	pub mip_count: MipCount,// If None the mip count will be decided based on the image extent.
	pub format: Format,
	pub usage: vk::ImageUsageFlags,
	pub initial_layout: vk::ImageLayout,
	pub tiling: vk::ImageTiling,
	pub msaa: Msaa,
	pub layers: u32,
	pub memory: MemoryLocation,
}

impl Default for ImageCreateInfo {
	fn default() -> Self {
		Self {
			resolution: UVec2::new(1, 1),
			mip_count: MipCount::None,
			format: Format::RgbaU8Srgb,
			usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
			initial_layout: vk::ImageLayout::UNDEFINED,
			tiling: vk::ImageTiling::OPTIMAL,
			msaa: Msaa::None,
			layers: 1,
			memory: MemoryLocation::GpuOnly,
		}
	}
}

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash, Debug, Reflect)]
pub enum MipCount {
	#[default]
	None,// No mips.
	Max,// Max mips based on resolution.
	Custom(u32),// User-defined will be clamped from 1..Max.
}
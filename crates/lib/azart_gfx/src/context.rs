use std::ffi::{c_char, CStr, CString, OsStr};
use std::io::{BufReader, Read};
use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::ops::{Deref, Range};
use std::{mem, ptr, slice};
use std::any::TypeId;
use std::path::Path;
use std::sync::{Arc, Mutex};
use ash::vk;
use ash::vk::Handle;
use bevy::log::{error, info, warn};
use bevy::math::UVec2;
use bevy::prelude::{FromReflect, PartialReflect, Resource};
use bevy::tasks::IoTaskPool;
use bevy::utils::HashSet;
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings, MemoryLocation};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use thiserror::Error;
use crate::buffer::Buffer;
use crate::Image;
use azart_gfx_utils::misc::{MsaaCount, ShaderPath};
use azart_gfx_utils::spirv::Spirv;
use azart_utils::debug_string::{DebugString, dbgfmt};
use azart_utils::io;
use bevy::asset::ron;
use bevy::reflect::serde::{ReflectDeserializer, TypedReflectDeserializer};
use bevy::reflect::TypeRegistry;
use serde::de::DeserializeSeed;
use vk_sync::*;
use crate::xr::XrInstance;

pub struct GpuContext {
	pub entry: ash::Entry,
	pub instance: ash::Instance,
	pub physical_device: vk::PhysicalDevice,
	pub device: ash::Device,
	pub queue_families: QueueFamilies,
	pub extensions: Extensions,
	pub capabilities: Capabilities,
	pub xr: Option<XrInstance>,// Only valid if running with OpenXR (not necessarily playing with VR).
	pub(crate) allocator: ManuallyDrop<Mutex<Allocator>>,
}

impl GpuContext {
	#[cfg(debug_assertions)]
	const INSTANCE_LAYER_NAMES: [&'static CStr; 1] = [
		c"VK_LAYER_KHRONOS_validation",
	];
	#[cfg(not(debug_assertions))]
	const INSTANCE_LAYER_NAMES: [&'static CStr; 0] = [];

	#[cfg(debug_assertions)]
	const INSTANCE_EXTENSION_NAMES: [&'static CStr; 1] = [
		ash::ext::debug_utils::NAME
	];
	#[cfg(not(debug_assertions))]
	const INSTANCE_EXTENSION_NAMES: [&'static CStr; 0] = [];

	// List of core extensions.
	const DEVICE_EXTENSION_NAMES: [&'static CStr; 12] = [
		ash::ext::descriptor_indexing::NAME,
		ash::khr::buffer_device_address::NAME,
		ash::khr::push_descriptor::NAME,
		ash::ext::extended_dynamic_state::NAME,
		ash::khr::swapchain::NAME,
		ash::khr::multiview::NAME,
		ash::khr::draw_indirect_count::NAME,
		ash::khr::shader_draw_parameters::NAME,
		ash::khr::create_renderpass2::NAME,
		ash::khr::timeline_semaphore::NAME,
		ash::khr::fragment_shading_rate::NAME,
		ash::ext::fragment_density_map::NAME,
	];

	pub fn new(extensions: &[&CStr], xr: Option<XrInstance>) -> Self {
		unsafe {
			let entry = ash::Entry::load().expect("failed to create entry point for vulkan!");
			let instance = {
				let layer_names = {
					let available_layers = entry
						.enumerate_instance_layer_properties()
						.unwrap()
						.into_iter()
						.map(|x| CStr::from_ptr(x.layer_name.as_ptr()).to_owned())
						.collect::<HashSet<_>>();

					Self::INSTANCE_LAYER_NAMES
						.iter()
						.filter(|&&x| {
							let available = available_layers.contains(x);

							#[cfg(debug_assertions)]
							if !available {
								warn!("Layer {} is unavailable! Removing from list of layers", x.to_str().unwrap());
							}

							available
						})
						.map(|x| x.as_ptr())
						.collect::<Vec<_>>()
				};

				let extensions = {
					let available_extensions = [None]
						.into_iter()
						.chain(layer_names
							.iter()
							.map(|&x| Some(CStr::from_ptr(x)))
						)
						.flat_map(|x| entry.enumerate_instance_extension_properties(x).unwrap())
						.map(|x| CStr::from_ptr(x.extension_name.as_ptr()).to_owned())
						.collect::<HashSet<_>>();

					Self::INSTANCE_EXTENSION_NAMES
						.iter()
						.chain(extensions.iter())
						.filter(|&&x| {
							let available = available_extensions.contains(x);

							#[cfg(debug_assertions)]
							if !available {
								warn!("Extension {} is unavailable! Removing from list of instance extensions", x.to_str().unwrap());
							}

							available
						})
						.map(|&x| x.as_ptr())
						.collect::<Vec<_>>()
				};

				const API_VERSION: u32 = vk::make_api_version(0, 1, 1, 0);

				let app_info = vk::ApplicationInfo::default()
					.api_version(API_VERSION)
					.application_name(c"azart game")
					.engine_name(c"azart engine")
					.engine_version(vk::make_api_version(0, 1, 0, 0));

				let create_info = vk::InstanceCreateInfo::default()
					.enabled_layer_names(&layer_names)
					.enabled_extension_names(&extensions)
					.application_info(&app_info);

				match &xr {
					Some(xr) => {
						#[cfg(debug_assertions)]
						{
							const XR_API_VERSION: openxr::Version = openxr::Version::new(1, 0, 0);
							let requirements = xr.instance.graphics_requirements::<openxr::Vulkan>(xr.hmd).expect("Failed to get OpenXR graphics requirements for Vulkan!");

							assert!(XR_API_VERSION >= requirements.min_api_version_supported, "API version {:?} is less than min API version supported {:?} for OpenXR!", XR_API_VERSION, requirements.min_api_version_supported);
							assert!(XR_API_VERSION <= requirements.max_api_version_supported, "API version {:?} is greater than max API version supported {:?} for OpenXR!", XR_API_VERSION, requirements.max_api_version_supported);
						}

						let instance = xr.instance.create_vulkan_instance(
							xr.hmd,
							mem::transmute(entry.static_fn().get_instance_proc_addr),
							&create_info as *const _ as *const _,
						).unwrap().unwrap();

						ash::Instance::load(&entry.static_fn(), vk::Instance::from_raw(instance as _))
					}
					None => entry.create_instance(&create_info, None).expect("Failed to create Vulkan instance!")
				}
			};

			let physical_device = match &xr {
				Some(xr) => {
					let physical_device = xr.instance.vulkan_graphics_device(
						xr.hmd,
						instance.handle().as_raw() as *const _,
					).unwrap();

					vk::PhysicalDevice::from_raw(physical_device as _)
				},
				None => instance
					.enumerate_physical_devices()
					.unwrap()
					.into_iter()
					.map(|physical_device| (instance.get_physical_device_properties(physical_device), physical_device))
					.filter(|&(x, _)| x.api_version >= vk::make_api_version(0, 1, 1, 0) && matches!(x.device_type, vk::PhysicalDeviceType::DISCRETE_GPU | vk::PhysicalDeviceType::INTEGRATED_GPU))
					.max_by(|&(a, _), &(b, _)| a.device_type.cmp(&b.device_type))
					.map(|(_, x)| x)
					.expect("Failed to find a suitable physical device!")
			};

			let queue_families = {
				let queue_family_props = instance.get_physical_device_queue_family_properties(physical_device);

				let select_queue_family = |flags: vk::QueueFlags| {
					queue_family_props
						.iter()
						.enumerate()
						.filter(|&(_, x)| x.queue_flags & flags == flags)
						.max_by(|&(_, a), (_, b)| a.queue_count.cmp(&b.queue_count))
						.unwrap_or_else(|| panic!("failed to find suitable queue family with flags {:?}", flags))
						.0 as u32
				};
				
				QueueFamilies {
					graphics: select_queue_family(vk::QueueFlags::GRAPHICS),
					compute: select_queue_family(vk::QueueFlags::COMPUTE),
					transfer: select_queue_family(vk::QueueFlags::TRANSFER),
				}
			};

			let device = {
				let available_extensions = instance
					.enumerate_device_extension_properties(physical_device)
					.unwrap()
					.into_iter()
					.map(|x| CStr::from_ptr(x.extension_name.as_ptr()).to_owned())
					.collect::<HashSet<_>>();

				let extensions = Self::DEVICE_EXTENSION_NAMES
					.iter()
					.filter(|&&x| {
						let available = available_extensions.contains(x);

						#[cfg(debug_assertions)]
						if !available {
							warn!("Extension {} is not available. Removing from list of extensions", x.to_str().unwrap());
						}

						available
					})
					.map(|&x| x.as_ptr())
					.collect::<Vec<_>>();

				let queue_create_infos = [
						queue_families.graphics,
						queue_families.compute,
						queue_families.transfer,
					]
					.into_iter()
					.collect::<HashSet<_>>()// collect into a hash set so that only unique queue families are constructed
					.into_iter()
					.map(|x| vk::DeviceQueueCreateInfo::default()
						.queue_family_index(x)
						.queue_priorities(&[1.0])
					)
					.collect::<Vec<_>>();

				let available_features = instance.get_physical_device_features(physical_device);
				let mut features = vk::PhysicalDeviceFeatures::default();

				macro_rules! enable_feature {
					($feature:ident) => {
						if available_features.$feature == vk::TRUE {
							features.$feature = vk::TRUE;
						} else {
							warn!("Feature {} is not available. Removing from list of features", stringify!($feature));
						}
					}
				}

				enable_feature!(multi_draw_indirect);
				enable_feature!(multi_viewport);
				enable_feature!(shader_storage_buffer_array_dynamic_indexing);
				enable_feature!(shader_storage_image_array_dynamic_indexing);
				enable_feature!(shader_uniform_buffer_array_dynamic_indexing);
				enable_feature!(shader_storage_image_array_dynamic_indexing);
				enable_feature!(shader_storage_buffer_array_dynamic_indexing);

				let mut buffer_device_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
					.buffer_device_address(true);

				let create_info = vk::DeviceCreateInfo::default()
					.enabled_extension_names(&extensions)
					.queue_create_infos(&queue_create_infos)
					.enabled_features(&features)
					.push_next(&mut buffer_device_address_features);

				match &xr {
					Some(xr) => {
						let device = xr.instance.create_vulkan_device(
							xr.hmd,
							mem::transmute(entry.static_fn().get_instance_proc_addr),
							physical_device.as_raw() as _,
							&create_info as *const _ as *const _,
						).unwrap().unwrap();

						ash::Device::load(&instance.fp_v1_0(), vk::Device::from_raw(device as _))
					},
					None => instance.create_device(physical_device, &create_info, None).expect("Failed to create logical device!"),
				}
			};
			
			let extensions = Extensions {
				swapchain: ash::khr::swapchain::Device::new(&instance, &device),
				surface: ash::khr::surface::Instance::new(&entry, &instance),
				draw_indirect_count: ash::khr::draw_indirect_count::Device::new(&instance, &device),
				#[cfg(debug_assertions)]
				debug_utils: ash::ext::debug_utils::Device::new(&instance, &device),
			};

			let allocator = {
				#[cfg(debug_assertions)]
				let debug_settings = AllocatorDebugSettings {
					log_leaks_on_shutdown: true,
					store_stack_traces: true,
					..Default::default()
				};

				#[cfg(not(debug_assertions))]
				let debug_settings = AllocatorDebugSettings::default();

				let create_info = AllocatorCreateDesc {
					instance: instance.clone(),
					device: device.clone(),
					physical_device,
					debug_settings,
					buffer_device_address: true,
					allocation_sizes: AllocationSizes::default(),
				};

				Allocator::new(&create_info).unwrap()
			};

			let capabilities = Capabilities {
				max_msaa: {
					let props = instance.get_physical_device_properties(physical_device);
					let mask = props.limits.framebuffer_color_sample_counts & props.limits.framebuffer_depth_sample_counts;
					if mask.contains(vk::SampleCountFlags::TYPE_8) {
						MsaaCount::Sample8
					} else if mask.contains(vk::SampleCountFlags::TYPE_4) {
						MsaaCount::Sample4
					} else if mask.contains(vk::SampleCountFlags::TYPE_2) {
						MsaaCount::Sample2
					} else {
						MsaaCount::None
					}
				},
			};

			Self {
				entry,
				instance,
				physical_device,
				device,
				queue_families,
				extensions,
				capabilities,
				xr,
				allocator: ManuallyDrop::new(Mutex::new(allocator)),
			}
		}
	}

	pub fn alloc(&self, create_info: &AllocationCreateDesc) -> Allocation {
		self.allocator
			.lock()
			.unwrap()
			.allocate(&create_info)
			.unwrap_or_else(|e| panic!("Failed to allocate memory for {} with create_info: {create_info:?}. Error: {:?}", create_info.name, e))
	}

	pub fn dealloc(&self, allocation: Allocation) {
		self.allocator
			.lock()
			.unwrap()
			.free(allocation)
			.unwrap_or_else(|e| panic!("Failed to free memory. Error: {:?}", e));
	}

	pub fn pipeline_barrier(
		&self,
		cmd: vk::CommandBuffer,
		global_barrier: Option<GlobalBarrier>,
		buffer_barriers: &[BufferBarrier],
		image_barriers: &[ImageBarrier],
	) {
		let mut src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
		let mut dst_stage_mask = vk::PipelineStageFlags::BOTTOM_OF_PIPE;

		let memory_barrier = global_barrier.map(|global_barrier| {
			let (src_mask, dst_mask, barrier) = get_memory_barrier(&global_barrier);
			src_stage_mask |= src_mask;
			dst_stage_mask |= dst_mask;
			barrier
		});

		let memory_barriers = match &memory_barrier {
			Some(memory_barrier) => slice::from_ref(memory_barrier),
			None => &[],
		};

		let buffer_barriers = buffer_barriers
			.iter()
			.map(|buffer_barrier| {
				let (src_mask, dst_mask, barrier) = get_buffer_memory_barrier(buffer_barrier);
				src_stage_mask |= src_mask;
				dst_stage_mask |= dst_mask;
				barrier
			})
			.collect::<Vec<_>>();

		let image_barriers = image_barriers
			.iter()
			.map(|image_barrier| {
				let (src_mask, dst_mask, barrier) = get_image_memory_barrier(image_barrier);
				src_stage_mask |= src_mask;
				dst_stage_mask |= dst_mask;
				barrier
			})
			.collect::<Vec<_>>();

		unsafe {
			self.device.cmd_pipeline_barrier(
				cmd,
				src_stage_mask,
				dst_stage_mask,
				vk::DependencyFlags::empty(),
				&memory_barriers,
				&buffer_barriers,
				&image_barriers,
			);
		}
	}

	// @TODO: Hard-coded and inflexible + over-generalized. Making a RenderGraph should alleviate this problem.
	pub unsafe fn cmd_transition_image_layout(
		&self,
		cmd: vk::CommandBuffer,
		image: vk::Image,
		format: vk::Format,
		old_layout: vk::ImageLayout,
		new_layout: vk::ImageLayout,
		mips: Range<u32>,
		layers: Range<u32>,
	) {
		use vk::Format;
		use vk::ImageLayout;
		use vk::AccessFlags;
		use vk::PipelineStageFlags;
		use vk::ImageAspectFlags;

		assert_ne!(mips.len(), 0, "Mip count must be greater than 0!");
		
		let (
			src_access_mask,
			dst_access_mask,
			src_stage,
			dst_stage,
		) = match (old_layout, new_layout) {
			(ImageLayout::UNDEFINED, ImageLayout::TRANSFER_SRC_OPTIMAL | ImageLayout::TRANSFER_DST_OPTIMAL) => (
				AccessFlags::empty(),
				AccessFlags::TRANSFER_WRITE,
				PipelineStageFlags::TOP_OF_PIPE,
				PipelineStageFlags::TRANSFER,
			),
			(ImageLayout::TRANSFER_DST_OPTIMAL, ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
				AccessFlags::TRANSFER_WRITE,
				AccessFlags::SHADER_READ,
				PipelineStageFlags::TRANSFER,
				PipelineStageFlags::FRAGMENT_SHADER,
			),
			(ImageLayout::TRANSFER_DST_OPTIMAL, ImageLayout::PRESENT_SRC_KHR) => (
				AccessFlags::TRANSFER_WRITE,
				AccessFlags::empty(),
				PipelineStageFlags::TRANSFER,
				PipelineStageFlags::BOTTOM_OF_PIPE,
			),
			(ImageLayout::UNDEFINED, ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
				AccessFlags::empty(),
				AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | AccessFlags::INPUT_ATTACHMENT_READ,
				PipelineStageFlags::TOP_OF_PIPE,
				PipelineStageFlags::EARLY_FRAGMENT_TESTS,
			),
			_ => panic!("Unhandled layout transition {:?} to {:?}", old_layout, new_layout),
		};

		let aspect_mask = match (new_layout, format) {
			(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, Format::D32_SFLOAT_S8_UINT | Format::D24_UNORM_S8_UINT) => ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
			(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, _) => ImageAspectFlags::DEPTH,
			_ => ImageAspectFlags::COLOR,
		};

		let barrier = vk::ImageMemoryBarrier::default()
			.src_access_mask(src_access_mask)
			.dst_access_mask(dst_access_mask)
			.old_layout(old_layout)
			.new_layout(new_layout)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(image)
			.subresource_range(vk::ImageSubresourceRange::default()
				.aspect_mask(aspect_mask)
				.base_mip_level(mips.start)
				.level_count(mips.len() as u32)
				.base_array_layer(layers.start)
				.layer_count(layers.len() as u32)
			);

		unsafe { self.device.cmd_pipeline_barrier(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[], &[barrier]); }
	}
	
	// @NOTE: Insanely unoptimized atm.
	pub fn immediate_cmd<T>(&self, queue_family: u32, closure: impl FnOnce(vk::CommandBuffer) -> T) -> T {
		assert!(queue_family == self.queue_families.graphics || queue_family == self.queue_families.compute || queue_family == self.queue_families.transfer);
		
		let command_pool = {
			let create_info = vk::CommandPoolCreateInfo::default()
				.queue_family_index(self.queue_families.graphics);
			
			unsafe { self.device.create_command_pool(&create_info, None).expect("Failed to create command pool!") }
		};
		
		let cmd = {
			let create_info = vk::CommandBufferAllocateInfo::default()
				.command_pool(command_pool)
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1);
			
			unsafe { self.device.allocate_command_buffers(&create_info) }.expect("Failed to allocate command buffer!")[0]
		};
		
		// Begin.
		unsafe {
			let begin_info = vk::CommandBufferBeginInfo::default()
				.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
			
			self.device.begin_command_buffer(cmd, &begin_info).expect("Failed to begin command buffer!");
		}
		
		let result = closure(cmd);
		
		// End
		unsafe { self.device.end_command_buffer(cmd) }.unwrap();
		
		let one_time_submit_fence = {
			let create_info = vk::FenceCreateInfo::default();
			unsafe { self.device.create_fence(&create_info, None).expect("Failed to create fence!") }
		};
		
		// Upload to queue.
		{
			let submit_info = vk::SubmitInfo::default()
				.command_buffers(slice::from_ref(&cmd));
			
			let queue = unsafe { self.device.get_device_queue(queue_family, 0) };
			
			unsafe { self.device.queue_submit(queue, slice::from_ref(&submit_info), one_time_submit_fence) }.expect("Failed to submit command buffer!");
		};
		
		// Wait for all queued commands to finish.
		unsafe { self.device.wait_for_fences(slice::from_ref(&one_time_submit_fence), true, u64::MAX) }.unwrap();
		
		// Wait for submit to complete.
		unsafe {
			self.device.destroy_fence(one_time_submit_fence, None);
			self.device.free_command_buffers(command_pool, slice::from_ref(&cmd));
			self.device.destroy_command_pool(command_pool, None);
		}
		
		result
	}
	
	pub fn upload_buffer<T>(&self, buffer: &Buffer, closure: impl FnOnce(&mut [u8]) -> T) -> T {
		let staging_buffer = {
			let create_info = vk::BufferCreateInfo::default()
				.usage(vk::BufferUsageFlags::TRANSFER_SRC)
				.size(buffer.size() as u64)
				.queue_family_indices(slice::from_ref(&self.queue_families.transfer))
				.sharing_mode(vk::SharingMode::EXCLUSIVE);

			unsafe { self.device.create_buffer(&create_info, None) }.unwrap()
		};

		let allocation = {
			let (requirements, dedicated_allocation) = {
				let info = vk::BufferMemoryRequirementsInfo2::default()
					.buffer(staging_buffer);

				let mut requirements = vk::MemoryRequirements2::default();
				let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
				requirements.push_next(&mut dedicated_requirements);

				unsafe { self.device.get_buffer_memory_requirements2(&info, &mut requirements); }
				(requirements.memory_requirements, dedicated_requirements.prefers_dedicated_allocation == vk::TRUE)
			};

			let allocation_scheme = match dedicated_allocation {
				true => AllocationScheme::DedicatedBuffer(staging_buffer),
				false => AllocationScheme::GpuAllocatorManaged,
			};

			let name = dbgfmt!("staging_buffer_{}", buffer.name().as_str());
			let create_info = AllocationCreateDesc {
				name: name.as_str(),
				requirements,
				location: MemoryLocation::CpuToGpu,
				linear: true,
				allocation_scheme,
			};

			self.alloc(&create_info)
		};

		unsafe { self.device.bind_buffer_memory(staging_buffer, allocation.memory(), allocation.offset()) }.expect("Failed to bind buffer memory!");
		
		let memory = unsafe {
			let memory = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
			slice::from_raw_parts_mut(memory, buffer.size())
		};
		
		let result = closure(memory);
		
		self.immediate_cmd(self.queue_families.transfer, |cmd| {
			let regions = [
				vk::BufferCopy::default()
					.src_offset(0)
					.dst_offset(0)
					.size(buffer.size() as u64),
			];
			
			unsafe { self.device.cmd_copy_buffer(cmd, staging_buffer, buffer.handle, &regions); }
		});
		
		unsafe {
			self.device.destroy_buffer(staging_buffer, None);
			self.dealloc(allocation);
		}
		
		result
	}

	// If the mips are not explicitly stated, all elements will be uploaded to.
	// @TODO: Layers.
	// @TODO: Optimize.
	pub fn upload_image<T>(&self, image: &Image, mips: Option<Range<u32>>, closure: impl FnOnce(&mut [&mut [u8]]) -> T) -> T {
		assert!(image.usage.contains(vk::ImageUsageFlags::TRANSFER_DST), "Image {} must be created with TRANSFER_DST usage flag!", image.name());
		assert_eq!(image.msaa, MsaaCount::None, "Can not upload Msaa target image with sample count {:?}", image.msaa);

		let mips = mips.unwrap_or(0..image.mips());

		let staging_buffer = {
			let pixel_count = mips
				.clone()
				.map(|mip| (image.resolution.x >> mip).max(1) * (image.resolution.y >> mip).max(1))
				.sum::<u32>();

			let create_info = vk::BufferCreateInfo::default()
				.usage(vk::BufferUsageFlags::TRANSFER_SRC)
				.size(pixel_count as vk::DeviceSize * image.format.block_size() as vk::DeviceSize)
				.sharing_mode(vk::SharingMode::EXCLUSIVE);

			unsafe { self.device.create_buffer(&create_info, None) }.unwrap()
		};

		let allocation = {
			let (requirements, dedicated_allocation) = {
				let info = vk::BufferMemoryRequirementsInfo2::default()
					.buffer(staging_buffer);

				let mut requirements = vk::MemoryRequirements2::default();
				let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
				requirements.push_next(&mut dedicated_requirements);

				unsafe { self.device.get_buffer_memory_requirements2(&info, &mut requirements); }
				(requirements.memory_requirements, dedicated_requirements.prefers_dedicated_allocation == vk::TRUE)
			};

			let allocation_scheme = match dedicated_allocation {
				true => AllocationScheme::DedicatedBuffer(staging_buffer),
				false => AllocationScheme::GpuAllocatorManaged,
			};

			let name = dbgfmt!("staging_buffer_{}", image.name().as_str());
			let create_info = AllocationCreateDesc {
				name: name.as_str(),
				requirements,
				location: MemoryLocation::CpuToGpu,
				linear: true,
				allocation_scheme,
			};

			self.alloc(&create_info)
		};

		unsafe { self.device.bind_buffer_memory(staging_buffer, allocation.memory(), allocation.offset()) }.expect("Failed to bind buffer memory!");

		let mut mips_memory = {
			let mut memory = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
			mips
				.clone()
				.map(|mip| {
					let size = (image.resolution.x >> mip).max(1) * (image.resolution.y >> mip).max(1) * image.format.block_size() as u32;
					let slice = unsafe { slice::from_raw_parts_mut(memory, size as usize) };
					memory = unsafe { memory.add(size as usize) };
					slice
				})
				.collect::<Vec<_>>()
		};

		let result = closure(&mut mips_memory);

		self.immediate_cmd(self.queue_families.transfer, |cmd| {
			unsafe { self.cmd_transition_image_layout(cmd, image.handle, image.format.into(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, mips.clone(), 0..1); };

			let mut offset = 0;
			let regions = mips
				.clone()
				.map(|mip| {
					let width = (image.resolution.x >> mip).max(1);
					let height = (image.resolution.y >> mip).max(1);
					let size = width * height * image.format.block_size() as u32;

					let region = vk::BufferImageCopy::default()
						.buffer_offset(offset)
						.buffer_row_length(0)
						.buffer_image_height(0)
						.image_subresource(vk::ImageSubresourceLayers::default()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.mip_level(mip)
							.base_array_layer(0)
							.layer_count(1)
						)
						.image_offset(vk::Offset3D { x: 0, y: 0, z: 0})
						.image_extent(vk::Extent3D { width, height, depth: 1 });

					offset += size as vk::DeviceSize;

					region
				})
				.collect::<Vec<_>>();

			unsafe { self.device.cmd_copy_buffer_to_image(cmd, staging_buffer, image.handle, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions); }

			unsafe { self.cmd_transition_image_layout(cmd, image.handle, image.format.into(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, mips, 0..1); }
		});

		unsafe {
			self.device.destroy_buffer(staging_buffer, None);
			self.dealloc(allocation);
		}

		result
	}
	
	pub fn create_shader_module(&self, path: &ShaderPath) -> Result<ShaderModule, ShaderModuleError> {
		// Load Spirv struct from file.
		let spirv = {
			let file_contents = io::read(&path.0).unwrap_or_else(|e| panic!("Failed to read shader file {path:?}: {e}"));

			let mut registry = TypeRegistry::new();
			registry.register::<Spirv>();

			let deserializer = ReflectDeserializer::new(&registry);

			let deserialized_value = deserializer.deserialize(
				&mut ron::Deserializer::from_bytes(&file_contents).unwrap()
			).unwrap();

			<Spirv as FromReflect>::from_reflect(&*deserialized_value).unwrap()
		};

		let shader_module = {
			let create_info = vk::ShaderModuleCreateInfo::default()
				.code(&spirv.code);

			unsafe { self.device.create_shader_module(&create_info, None) }.expect("Failed to create shader module!")
		};

		Ok(ShaderModule {
			handle: shader_module,
			spirv,
			context: &self,
		})
	}

	pub fn wait_idle(&self) {
		unsafe { self.device.device_wait_idle().expect("Failed to wait for device to become idle!"); }
	}

	pub fn max_msaa_samples(&self) -> MsaaCount {
		let props = unsafe { self.instance.get_physical_device_properties(self.physical_device) };
		let sample_count = props.limits.framebuffer_color_sample_counts & props.limits.framebuffer_depth_sample_counts;
		let sample_count = vk::SampleCountFlags::from_raw(1 << sample_count.as_raw().trailing_zeros());// Use the smallest available sample count.
		sample_count.into()
	}

	// Handle must be valid and ensure this is only called from a single thread per-object!
	#[cfg(debug_assertions)]
	pub unsafe fn set_debug_name_cstr(&self, name: &CStr, handle: impl vk::Handle) {
		let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
			.object_name(name)
			.object_handle(handle);

		//unsafe { self.extensions.debug_utils.set_debug_utils_object_name(&name_info) }.expect("Failed to set debug name!");
		let result = unsafe { self.extensions.debug_utils.set_debug_utils_object_name(&name_info) };
		if let Err(e) = result {
			error!("Failed to set debug name {}!: {e}", name.to_str().unwrap_or("<invalid>"));
		}
	}

	// Handle must be valid and ensure this is only called from a single thread per-object!
	#[cfg(debug_assertions)]
	pub unsafe fn set_debug_name(&self, name: &str, handle: impl vk::Handle) {
		let name = CString::new(name).unwrap_or_else(|_| panic!("Failed to convert string \"{name}\" to CString!"));
		unsafe { self.set_debug_name_cstr(name.as_c_str(), handle); }
	}
}

impl Drop for GpuContext {
	fn drop(&mut self) {
		unsafe {
			ManuallyDrop::drop(&mut self.allocator);
			self.device.destroy_device(None);
			self.instance.destroy_instance(None);
		}
	}
}

pub struct QueueFamilies {
	pub graphics: u32,
	pub compute: u32,
	pub transfer: u32,
}

pub struct Extensions {
	pub swapchain: ash::khr::swapchain::Device,
	pub surface: ash::khr::surface::Instance,
	pub draw_indirect_count: ash::khr::draw_indirect_count::Device,
	#[cfg(debug_assertions)]
	pub debug_utils: ash::ext::debug_utils::Device,
}

#[derive(Debug)]
pub struct Capabilities {
	pub max_msaa: MsaaCount,
}

// Bevy Resource that holds an Arc<GpuContext>.
#[derive(Resource)]
pub struct GpuContextHandle(pub Arc<GpuContext>);

impl GpuContextHandle {
	pub const fn new(context: Arc<GpuContext>) -> Self {
		Self(context)
	}
}

impl Deref for GpuContextHandle {
	type Target = Arc<GpuContext>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

pub struct ShaderModule<'a> {
	pub(crate) handle: vk::ShaderModule,
	pub spirv: Spirv,
	pub(crate) context: &'a GpuContext,
}

impl Drop for ShaderModule<'_> {
	fn drop(&mut self) {
		unsafe { self.context.device.destroy_shader_module(self.handle, None); }
	}
}

#[derive(Debug, Error)]
pub enum ShaderModuleError {
	#[error("Invalid path {0}!")]
	InvalidPath(String),
	#[error("Permission denied to access file path {0}!")]
	PermissionDenied(String),
	#[error("Failed to read shader module {0}! Error {1}")]
	UnknownFileError(String, std::io::Error),
	#[error("No .spv suffix on path!")]
	NotSpv,
}
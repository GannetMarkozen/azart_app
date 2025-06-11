use std::ffi::{c_char, c_void, CStr, CString, OsStr};
use std::io::{BufReader, Read};
use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::ops::{Deref, Range};
use std::{mem, ptr, slice};
use std::any::TypeId;
use std::borrow::Cow;
use std::cell::Cell;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use ash::vk;
use ash::vk::Handle;
use bevy::log::{debug, error, info, warn};
use bevy::math::UVec2;
use bevy::prelude::{Deref, FromReflect, PartialReflect, Resource};
use bevy::tasks::IoTaskPool;
use bevy::utils::HashSet;
use derivative::Derivative;
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings, MemoryLocation};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use thiserror::Error;
use crate::buffer::Buffer;
use crate::Image;
use azart_gfx_utils::misc::{Msaa, ShaderPath};
use azart_gfx_utils::spirv::Spirv;
use azart_utils::debug_string::{DebugString, dbgfmt};
use azart_utils::io;
use bevy::asset::ron;
use bevy::reflect::serde::{ReflectDeserializer, TypedReflectDeserializer};
use bevy::reflect::TypeRegistry;
use serde::de::DeserializeSeed;
use vk_sync::*;
use openxr as xr;
use crate::render::xr::XrInstance;

pub struct GpuContext {
	pub entry: ash::Entry,
	pub instance: ash::Instance,
	pub physical_device: vk::PhysicalDevice,
	pub device: ash::Device,
	#[cfg(debug_assertions)]
	pub dbg_messenger: vk::DebugUtilsMessengerEXT,
	pub queue_families: QueueFamilies,
	pub exts: Extensions,
	pub limits: Limits,
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
		ash::ext::debug_utils::NAME,
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
		ash::khr::draw_indirect_count::NAME,
		ash::khr::create_renderpass2::NAME,
		ash::khr::timeline_semaphore::NAME,
		ash::khr::fragment_shading_rate::NAME,
		ash::ext::fragment_density_map::NAME,
		ash::khr::create_renderpass2::NAME,
		ash::khr::uniform_buffer_standard_layout::NAME,// For ubo scalar layout.
	];

	pub fn new(extensions: &[&CStr], xr: Option<(&xr::Instance, xr::SystemId)>) -> Self {
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

				#[cfg(debug_assertions)]
				let mut debug_messenger_ext = layer_names
					.iter()
					.any(|&name| unsafe { CStr::from_ptr(name) } == c"VK_LAYER_KHRONOS_validation")
					.then(|| vk::DebugUtilsMessengerCreateInfoEXT::default()
						.message_severity(
							vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
								| vk::DebugUtilsMessageSeverityFlagsEXT::INFO
								| vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
								| vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
						)
						.message_type(
							vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
								| vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
								| vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
						)
						.pfn_user_callback(Some(dbg_messenger_callback))
					);

				let mut create_info = vk::InstanceCreateInfo::default()
					.enabled_layer_names(&layer_names)
					.enabled_extension_names(&extensions)
					.application_info(&app_info);

				#[cfg(debug_assertions)]
				if let Some(debug_messenger_ext) = &mut debug_messenger_ext {
					create_info = create_info.push_next(debug_messenger_ext);
				}

				match xr {
					Some((xr, hmd)) => {
						const XR_API_VERSION: xr::Version = xr::Version::new(1, 0, 0);

						// @NOTE: MUST call this before creating a session otherwise you will get esoteric errors. Regardless of if you use the result.
						let requirements = xr.graphics_requirements::<xr::Vulkan>(hmd).expect("Failed to get OpenXR graphics requirements for Vulkan!");

						assert!(XR_API_VERSION >= requirements.min_api_version_supported, "API version {:?} is less than min API version supported {:?} for OpenXR!", XR_API_VERSION, requirements.min_api_version_supported);
						assert!(XR_API_VERSION <= requirements.max_api_version_supported, "API version {:?} is greater than max API version supported {:?} for OpenXR!", XR_API_VERSION, requirements.max_api_version_supported);

						let instance = xr.create_vulkan_instance(
							hmd,
							mem::transmute(entry.static_fn().get_instance_proc_addr),
							&create_info as *const _ as *const _,
						).unwrap().unwrap();

						ash::Instance::load(&entry.static_fn(), vk::Instance::from_raw(instance as _))
					}
					None => entry.create_instance(&create_info, None).expect("Failed to create Vulkan instance!")
				}
			};

			let physical_device = match xr {
				Some((xr, hmd)) => {
					let physical_device = xr.vulkan_graphics_device(
						hmd,
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

			#[cfg(debug_assertions)]
			let (instance_debug_utils, debug_messenger) = {
				let instance_debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
				let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
					.message_severity(
						vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
							| vk::DebugUtilsMessageSeverityFlagsEXT::INFO
							| vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
							| vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
					)
					.message_type(
						vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
							| vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
							| vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
					)
					.pfn_user_callback(Some(dbg_messenger_callback));

				let messenger = unsafe { instance_debug_utils.create_debug_utils_messenger(&create_info, None) }.unwrap();

				(instance_debug_utils, messenger)
			};

			let (device, buffer_device_address_enabled) = {
				let available_extensions = instance
					.enumerate_device_extension_properties(physical_device)
					.unwrap()
					.into_iter()
					.map(|x| CStr::from_ptr(x.extension_name.as_ptr()).to_owned())
					.collect::<HashSet<_>>();

				println!("Available extensions: {available_extensions:?}");

				let exts = Self::DEVICE_EXTENSION_NAMES
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

				let (available_features_1_0, available_features_1_1) = {
					let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::default();
					let mut features2 = vk::PhysicalDeviceFeatures2::default()
						.push_next(&mut features_1_1);

					instance.get_physical_device_features2(physical_device, &mut features2);

					(features2.features, features_1_1)
				};

				let mut features_1_0 = vk::PhysicalDeviceFeatures::default();
				let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::default();

				macro_rules! enable_feature_1_0 {
					($feature:ident) => {
						if available_features_1_0.$feature == vk::TRUE {
							features_1_0.$feature = vk::TRUE;
						} else {
							warn!("Physical device feature (1.0) {} is not available. Removing from list of features", stringify!($feature));
						}
					}
				}

				macro_rules! enable_feature_1_1 {
					($feature:ident) => {
						if available_features_1_1.$feature == vk::TRUE {
							features_1_1.$feature = vk::TRUE;
							true
						} else {
							error!("Physical device feature (1.1) {} is not available. Removing from list of features", stringify!($feature));
							false
						}
					}
				}

				enable_feature_1_0!(multi_draw_indirect);
				enable_feature_1_0!(multi_viewport);
				enable_feature_1_0!(shader_storage_buffer_array_dynamic_indexing);
				enable_feature_1_0!(shader_storage_image_array_dynamic_indexing);
				enable_feature_1_0!(shader_uniform_buffer_array_dynamic_indexing);
				enable_feature_1_0!(shader_storage_image_array_dynamic_indexing);
				enable_feature_1_0!(shader_storage_buffer_array_dynamic_indexing);

				enable_feature_1_1!(multiview);
				enable_feature_1_1!(shader_draw_parameters);

				let buffer_device_address_enabled = exts
					.iter()
					.any(|&ext| unsafe { CStr::from_ptr(ext) } == ash::ext::buffer_device_address::NAME);

				let mut buffer_device_address_features = buffer_device_address_enabled.then(|| vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
					.buffer_device_address(true)
				);

				let mut create_info = vk::DeviceCreateInfo::default()
					.push_next(&mut features_1_1)
					.enabled_features(&features_1_0)
					.enabled_extension_names(&exts)
					.queue_create_infos(&queue_create_infos);

				if let Some(buffer_device_address_features) = &mut buffer_device_address_features {
					create_info = create_info.push_next(buffer_device_address_features);
				}

				let device = match xr {
					Some((xr, hmd)) => {
						let device = xr.create_vulkan_device(
							hmd,
							mem::transmute(entry.static_fn().get_instance_proc_addr),
							physical_device.as_raw() as _,
							&create_info as *const _ as *const _,
						)
							.unwrap()
							.unwrap();

						ash::Device::load(&instance.fp_v1_0(), vk::Device::from_raw(device as _))
					},
					None => instance.create_device(physical_device, &create_info, None).expect("Failed to create logical device!"),
				};

				(device, buffer_device_address_enabled)
			};

			let exts = Extensions {
				swapchain: ash::khr::swapchain::Device::new(&instance, &device),
				surface: ash::khr::surface::Instance::new(&entry, &instance),
				draw_indirect_count: ash::khr::draw_indirect_count::Device::new(&instance, &device),
				create_render_pass2: ash::khr::create_renderpass2::Device::new(&instance, &device),
				push_descriptor: ash::khr::push_descriptor::Device::new(&instance, &device),
				#[cfg(debug_assertions)]
				instance_debug_utils,
				#[cfg(debug_assertions)]
				device_debug_utils: ash::ext::debug_utils::Device::new(&instance, &device),
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
					buffer_device_address: buffer_device_address_enabled,// Will be forcefully disabled by RenderDoc!
					allocation_sizes: AllocationSizes::default(),
				};

				Allocator::new(&create_info).unwrap()
			};

			let limits = {
				let props = instance.get_physical_device_properties(physical_device);
				Limits {
					max_msaa: {
						let mask = props.limits.framebuffer_color_sample_counts & props.limits.framebuffer_depth_sample_counts;
						if mask.contains(vk::SampleCountFlags::TYPE_8) {
							Msaa::x8
						} else if mask.contains(vk::SampleCountFlags::TYPE_4) {
							Msaa::x4
						} else if mask.contains(vk::SampleCountFlags::TYPE_2) {
							Msaa::x2
						} else {
							Msaa::None
						}
					},
					ubo_max_size: props.limits.max_uniform_buffer_range as _,
					ubo_min_align: props.limits.min_uniform_buffer_offset_alignment as _,
					ssbo_max_size: props.limits.max_storage_buffer_range as _,
					ssbo_min_align: props.limits.min_storage_buffer_offset_alignment as _,
					push_constants_max_size: props.limits.max_push_constants_size as _,
				}
			};

			Self {
				entry,
				instance,
				physical_device,
				device,
				#[cfg(debug_assertions)]
				dbg_messenger: debug_messenger,
				queue_families,
				exts,
				limits,
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

		let cmd_pool = {
			let create_info = vk::CommandPoolCreateInfo::default()
				.queue_family_index(self.queue_families.graphics);

			unsafe { self.device.create_command_pool(&create_info, None).expect("Failed to create command pool!") }
		};

		let cmd = {
			let create_info = vk::CommandBufferAllocateInfo::default()
				.command_pool(cmd_pool)
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
			self.device.free_command_buffers(cmd_pool, slice::from_ref(&cmd));
			self.device.destroy_command_pool(cmd_pool, None);
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
		assert_eq!(image.msaa, Msaa::None, "Can not upload Msaa target image with sample count {:?}", image.msaa);

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

	pub fn max_msaa_samples(&self) -> Msaa {
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
		let result = unsafe { self.exts.device_debug_utils.set_debug_utils_object_name(&name_info) };
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
		self.wait_idle();

		unsafe {
			ManuallyDrop::drop(&mut self.allocator);

			self.device.destroy_device(None);

			#[cfg(debug_assertions)]
			self.exts.instance_debug_utils.destroy_debug_utils_messenger(self.dbg_messenger, None);

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
	pub create_render_pass2: ash::khr::create_renderpass2::Device,
	pub push_descriptor: ash::khr::push_descriptor::Device,
	#[cfg(debug_assertions)]
	pub instance_debug_utils: ash::ext::debug_utils::Instance,
	#[cfg(debug_assertions)]
	pub device_debug_utils: ash::ext::debug_utils::Device,
}

#[derive(Debug)]
pub struct Limits {
	pub max_msaa: Msaa,
	pub ubo_max_size: usize,
	pub ubo_min_align: usize,
	pub ssbo_max_size: usize,
	pub ssbo_min_align: usize,
	pub push_constants_max_size: usize,
}

// Bevy Resource that holds an Arc<GpuContext>.
#[derive(Resource, Deref)]
pub struct GpuContextHandle(pub Arc<GpuContext>);

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

#[must_use = "RAII object"]
pub struct RhiLabel<'a> {
	context: &'a GpuContext,
	cmd: &'a vk::CommandBuffer,
}

impl<'a> RhiLabel<'a> {
	#[inline]
	pub fn new(name: &str, context: &'a GpuContext, cmd: &'a vk::CommandBuffer) -> Self {
		#[cfg(debug_assertions)]
		unsafe {
			let name = CString::new(name).unwrap();

			context.exts.device_debug_utils.cmd_begin_debug_utils_label(
				*cmd,
				&vk::DebugUtilsLabelEXT::default()
					.label_name(name.as_c_str())
					.color([1.0, 0.0, 1.0, 1.0]),
			);
		}

		Self {
			context,
			cmd,
		}
	}
}

#[cfg(debug_assertions)]
impl Drop for RhiLabel<'_> {
	#[inline]
	fn drop(&mut self) {
		unsafe { self.context.exts.device_debug_utils.cmd_end_debug_utils_label(*self.cmd); }
	}
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! rhi_label {
	($label:expr, $context:expr, $cmd:expr) => {
		let _label = $crate::RhiLabel::new($label, $context, $cmd);
	};
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules ! rhi_label {
	($($arg:tt)*) => {}
}

thread_local! {
	// If this value is > 0. Vulkan debug messages will be disabled.
	#[cfg(debug_assertions)]
	static IGNORE_CALLBACK_DEPTH: Cell<usize> = Cell::new(0);
}

// While this object is alive, don't broadcast rhi messages.
#[must_use = "You must bind this value for scoped semantics!"]
pub struct RhiHush {
	_phantom: PhantomData<std::rc::Rc<()>>,// Forces !Send and !Copy.
}

impl RhiHush {
	#[inline]
	pub fn new() -> Self {
		#[cfg(debug_assertions)]
		IGNORE_CALLBACK_DEPTH.set(IGNORE_CALLBACK_DEPTH.get() + 1);

		Self {
			_phantom: PhantomData,
		}
	}
}

#[cfg(debug_assertions)]
impl Drop for RhiHush {
	#[inline]
	fn drop(&mut self) {
		IGNORE_CALLBACK_DEPTH.set(IGNORE_CALLBACK_DEPTH.get() - 1);
	}
}

#[macro_export]
macro_rules! rhi_hush {
	() => {
		#[cfg(debug_assertions)]
		let _guard = $crate::RhiHush::new();
	};
}

unsafe extern "system" fn dbg_messenger_callback(
	severity: vk::DebugUtilsMessageSeverityFlagsEXT,
	msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
	data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
	user_data: *mut c_void,
) -> vk::Bool32 {
	#[cfg(debug_assertions)]
	if IGNORE_CALLBACK_DEPTH.get() > 0 {
		return vk::FALSE;
	}

	assert_ne!(data, ptr::null());
	let data = unsafe { &*data };

	let msg_id = if data.p_message_id_name.is_null() { "<null>" } else { unsafe { CStr::from_ptr(data.p_message_id_name) }.to_str().unwrap() };
	let msg = if data.p_message.is_null() { "<null>" } else { unsafe { CStr::from_ptr(data.p_message) }.to_str().unwrap() };

	const IGNORE_MSGS: LazyLock<HashSet<&str>> = LazyLock::new(|| [
		"BestPractices-vkCreateDevice-physical-device-features-not-retrieved",// Because get_physical_device_features is not called but get_physical_device_features2 is so this can be ignored.
		"VUID-VkImageCreateInfo-pNext-01443",// openxr::FrameStream::end triggers this from swapchain image. Just ignore.
	].into());

	if IGNORE_MSGS.contains(msg_id) {
		return vk::FALSE;
	}

	#[derive(Debug)]
	pub enum MsgType {
		General,
		Validation,
		Performance,
	}

	#[derive(Debug)]
	pub enum Severity {
		Verbose,
		Info,
		Warning,
		Error,
	}

	let msg_type = match msg_type.as_raw().is_power_of_two() {
		true => msg_type,
		false => vk::DebugUtilsMessageTypeFlagsEXT::from_raw(1u32 << (32 - msg_type.as_raw().leading_zeros())),
	};

	let msg_type = match msg_type {
		vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => MsgType::General,
		vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => MsgType::Validation,
		vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => MsgType::Performance,
		_ => unreachable!("Unknown message type {msg_type:?}!"),
	};

	let severity = match severity {
		vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Severity::Verbose,
		vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Severity::Info,
		vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Severity::Warning,
		vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Severity::Error,
		_ => unreachable!("Unknown severity {severity:?}!"),
	};

	let backtrace = std::backtrace::Backtrace::capture().to_string();
	let module = match backtrace.contains("openxr::") {
		true => return vk::FALSE,//"openxr",// TMP.
		false => "application",
	};

	let msg = format!("VK[{msg_type:?}][{severity:?}][{msg_id}]: {msg}\nmodule: {module}\n");

	match severity {
		Severity::Verbose => debug!("{msg}"),
		Severity::Info => info!("{msg}"),
		Severity::Warning => warn!("{msg}"),
		Severity::Error => error!("{msg}"),
	}

	vk::FALSE
}

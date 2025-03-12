use std::ffi::{c_char, CStr, CString};
use std::io::Read;
use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use ash::vk;
use bevy::log::warn;
use bevy::prelude::Resource;
use bevy::tasks::IoTaskPool;
use bevy::utils::HashSet;
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings};
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use thiserror::Error;
use crate::azart::gfx::misc::ShaderPath;

pub struct GpuContext {
	pub entry: ash::Entry,
	pub instance: ash::Instance,
	pub physical_device: vk::PhysicalDevice,
	pub device: ash::Device,
	pub queue_families: QueueFamilies,
	pub extensions: Extensions,
	pub(crate) allocator: ManuallyDrop<Mutex<Allocator>>,
}

impl GpuContext {
	#[cfg(debug_assertions)]
	const INSTANCE_LAYER_NAMES: [&'static CStr; 1] = [c"VK_LAYER_KHRONOS_validation"];
	#[cfg(not(debug_assertions))]
	const INSTANCE_LAYER_NAMES: [&'static CStr; 0] = [];

	#[cfg(debug_assertions)]
	const INSTANCE_EXTENSION_NAMES: [&'static CStr; 1] = [ash::ext::debug_utils::NAME];
	#[cfg(not(debug_assertions))]
	const INSTANCE_EXTENSION_NAMES: [&'static CStr; 0] = [];

	const DEVICE_EXTENSION_NAMES: [&'static CStr; 8] = [
		ash::ext::descriptor_indexing::NAME,
		ash::ext::extended_dynamic_state::NAME,
		ash::khr::buffer_device_address::NAME,
		ash::khr::swapchain::NAME,
		ash::khr::multiview::NAME,
		ash::khr::draw_indirect_count::NAME,
		ash::khr::create_renderpass2::NAME,
		ash::khr::fragment_shading_rate::NAME,
	];

	pub fn new(extensions: &[&CStr]) -> Self {
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

				let app_info = vk::ApplicationInfo::default()
					.api_version(vk::make_api_version(0, 1, 1, 0))
					.application_name(c"azart game")
					.engine_name(c"azart engine")
					.engine_version(vk::make_api_version(0, 1, 0, 0));

				let create_info = vk::InstanceCreateInfo::default()
					.enabled_layer_names(&layer_names)
					.enabled_extension_names(&extensions)
					.application_info(&app_info);

				entry.create_instance(&create_info, None).expect("failed to create vulkan instance!")
			};

			let (physical_device, queue_families) = {
				let physical_devices = instance.enumerate_physical_devices().unwrap();
				let props = physical_devices
					.iter()
					.map(|&x| instance.get_physical_device_properties(x))
					.collect::<Vec<_>>();

				let result = props
					.iter()
					.enumerate()
					.filter(|&(_, x)| x.api_version >= vk::make_api_version(0, 1, 1, 0) && matches!(x.device_type, vk::PhysicalDeviceType::DISCRETE_GPU | vk::PhysicalDeviceType::INTEGRATED_GPU))
					.max_by(|&(_, a), (_, b)| a.device_type.cmp(&b.device_type))
					.expect("failed to find a suitable physical device!")
					.0;

				let physical_device = physical_devices[result];
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
				
				let queue_families = QueueFamilies {
					graphics: select_queue_family(vk::QueueFlags::GRAPHICS),
					compute: select_queue_family(vk::QueueFlags::COMPUTE),
					transfer: select_queue_family(vk::QueueFlags::TRANSFER),
				};

				(physical_device, queue_families)
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

				instance.create_device(physical_device, &create_info, None).expect("failed to create logical device!")
			};
			
			let extensions = Extensions {
				swapchain: ash::khr::swapchain::Device::new(&instance, &device),
				surface: ash::khr::surface::Instance::new(&entry, &instance),
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

			Self {
				entry,
				instance,
				physical_device,
				device,
				queue_families,
				extensions,
				allocator: ManuallyDrop::new(Mutex::new(allocator)),
			}
		}
	}

	pub fn alloc(&self, create_info: &gpu_allocator::vulkan::AllocationCreateDesc) -> Allocation {
		self.allocator.lock().unwrap().allocate(&create_info).expect("Failed to allocate memory!")
	}

	pub fn dealloc(&self, allocation: Allocation) {
		self.allocator.lock().unwrap().free(allocation).expect("Failed to deallocate allocation!");
	}

	pub unsafe fn cmd_transition_image_layout(
		&self,
		cmd: vk::CommandBuffer,
		image: vk::Image,
		format: vk::Format,
		old_layout: vk::ImageLayout,
		new_layout: vk::ImageLayout,
		mip_count: u32,
	) {
		use vk::Format;
		use vk::ImageLayout;
		use vk::AccessFlags;
		use vk::PipelineStageFlags;
		use vk::ImageAspectFlags;

		assert_ne!(mip_count, 0, "Mip count must be greater than 0!");
		
		let (
			src_access_mask,
			dst_access_mask,
			src_stage,
			dst_stage
		) = match (old_layout, new_layout) {
			(ImageLayout::UNDEFINED, ImageLayout::TRANSFER_DST_OPTIMAL) => (
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
				.base_mip_level(0)
				.level_count(mip_count)
				.base_array_layer(0)
				.layer_count(1)
			);

		unsafe { self.device.cmd_pipeline_barrier(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[], &[barrier]); }
	}

	pub fn create_shader_module(&self, path: ShaderPath) -> Result<ShaderModule, ShaderModuleError> {
		#[cfg(debug_assertions)]
		if !path.as_str().ends_with(".spv") {
			return Err(ShaderModuleError::NotSpv);
		}

		let mut file = match std::fs::File::open(path.as_str()) {
			Ok(file) => file,
			Err(e) => return match e.kind() {
				std::io::ErrorKind::NotFound => Err(ShaderModuleError::InvalidPath(path.as_str().to_owned())),
				std::io::ErrorKind::PermissionDenied => Err(ShaderModuleError::PermissionDenied(path.as_str().to_owned())),
				_ => Err(ShaderModuleError::UnknownFileError(path.as_str().to_owned(), e)),
			},
		};

		let mut code = vec![];
		file.read_to_end(&mut code).unwrap();

		let code = bytemuck::cast_slice(&code);

		let shader_module = {
			let create_info = vk::ShaderModuleCreateInfo::default()
				.code(&code);

			unsafe { self.device.create_shader_module(&create_info, None) }.expect("Failed to create shader module!")
		};

		Ok(ShaderModule {
			handle: shader_module,
			context: &self,
		})
	}

	pub fn wait_idle(&self) {
		unsafe { self.device.device_wait_idle().expect("Failed to wait for device to become idle!"); }
	}

	// Handle must be valid and ensure this is only called from a single thread per-object!
	#[cfg(debug_assertions)]
	pub unsafe fn set_debug_name_cstr(&self, name: &CStr, handle: impl vk::Handle) {
		let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
			.object_name(name)
			.object_handle(handle);

		unsafe { self.extensions.debug_utils.set_debug_utils_object_name(&name_info) }.expect("Failed to set debug name!");
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
	#[cfg(debug_assertions)]
	pub debug_utils: ash::ext::debug_utils::Device,
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
	pub handle: vk::ShaderModule,
	pub(crate) context: &'a GpuContext,
}

impl<'a> Drop for ShaderModule<'a> {
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
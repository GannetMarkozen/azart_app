use ash::vk::Handle;
use bevy::prelude::*;
use crate::GpuContext;
use openxr as xr;

pub struct XrInstance {
	pub(crate) entry: xr::Entry,
	pub(crate) instance: xr::Instance,
	pub(crate) hmd: xr::SystemId,
}

impl XrInstance {
	pub fn new() -> Self {
		#[cfg(target_os = "windows")]
		let entry = unsafe { xr::Entry::load() }.unwrap_or_else(|e| {
			info!("Failed to load OpenXR entry! OpenXR SDK not found: {e}");
			xr::Entry::linked()
		});

		#[cfg(not(target_os = "windows"))]
		let entry = unsafe { xr::Entry::load() }.expect("Failed to load OpenXR entry!");

		#[cfg(target_os = "android")]
		entry.initialize_android_loader().expect("Failed to initialize Android loader for OpenXR!");

		let instance = {
			let app_info = xr::ApplicationInfo {
				application_name: "azart_app",
				application_version: 0,
				engine_name: "azart_engine",
				engine_version: 0,
				api_version: xr::Version::new(1, 0, 0),
			};

			let available_exts = entry.enumerate_extensions().expect("Failed to enumerate OpenXR extensions!");
			let available_layers = entry.enumerate_layers().expect("Failed to enumerate OpenXR layers!");

			let mut exts = xr::ExtensionSet::default();
			exts.khr_vulkan_enable2 = true;
			exts.ext_debug_utils = true;

			entry.create_instance(&app_info, &exts, &[]).expect("Failed to create OpenXR instance!")
		};

		let hmd = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY).expect("No HMD found!");

		Self {
			entry,
			instance,
			hmd,
		}
	}
}

#[derive(Resource)]
pub struct XrSession {
	pub(crate) handle: xr::Session<xr::Vulkan>,
	pub(crate) frame_waiter: xr::FrameWaiter,
	pub(crate) frame_stream: xr::FrameStream<xr::Vulkan>,
	pub(crate) entry: xr::Entry,
	pub(crate) instance: xr::Instance,
}

impl XrSession {
	// Can only fail if the GpuContext was not created with an XrInstance.
	pub fn new(context: &GpuContext) -> Result<Self, &'static str> {
		let Some(xr) = &context.xr else {
			return Err("XrInstance unavailable!");
		};

		let (session, frame_waiter, frame_stream) = {
			let create_info = xr::vulkan::SessionCreateInfo {
				instance: context.instance.handle().as_raw() as _,
				physical_device: context.physical_device.as_raw() as _,
				device: context.device.handle().as_raw() as _,
				queue_family_index: context.queue_families.graphics,
				queue_index: 0,
			};

			unsafe { xr.instance.create_session(xr.hmd, &create_info) }.expect("Failed to create OpenXR session!")
		};

		Ok(Self {
			handle: session,
			frame_waiter,
			frame_stream,
			entry: xr.entry.clone(),
			instance: xr.instance.clone(),
		})
	}
}
use std::{env, mem, slice};
use std::borrow::Cow;
use std::cell::Cell;
use std::ffi::{c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;
use bevy::prelude::*;
use crate::render_settings::{DisplayMode, RenderSettings};
use ash::vk;
use ash::vk::Handle;
use azart_gfx_utils::{Format, Msaa};
use azart_utils::dbgfmt;
use azart_utils::debug_string::DebugString;
use either::{for_both, Either};
use gpu_allocator::MemoryLocation;
use openxr as xr;
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use vk_sync::ImageBarrier;
use crate::{rhi_hush, GpuContext, Image, ImageCreateInfo, MipCount};
use crate::render_pass::RenderPass;
use crate::xr_swapchain::{Swapchain, SwapchainCreateInfo};

#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, States)]
pub enum XrState {
	#[default]
	Idle,
	Focused,
}

#[derive(Resource, Deref)]
pub struct XrInstance {
	context: Arc<GpuContext>,// Must outlive this.
	pub entry: mem::ManuallyDrop<xr::Entry>,
	#[deref]
	pub handle: mem::ManuallyDrop<xr::Instance>,
	pub hmd: xr::SystemId,
	pub exts: Extensions,
}

impl XrInstance {
	#[cfg(debug_assertions)]
	const LAYERS: [&'static str; 1] = ["XR_APILAYER_LUNARG_core_validation"];

	#[cfg(not(debug_assertions))]
	const LAYERS: [&'static str; 0] = [];

	// With OpenXR enabled the OpenXR instance must create the basic Vulkan objects!
	pub fn new(vk_exts: &[&CStr]) -> Self {
		#[cfg(target_os = "windows")]
		let entry = {
			// @TODO: This only works when developing locally. Make work in packaged builds.
			let path = {
				let mut path = Cow::Borrowed("runtime_libs/openxr_loader/x86_64/x64/bin/api_layers");
				while !Path::new(&*path).exists() {
					path = format!("../{path}").into();
				}
				path
			};
			let path = &*path;

			assert!(Path::new(path).exists(), "path: {path:?}");

			unsafe { env::set_var("XR_API_LAYER_PATH", path) }

			xr::Entry::linked()
		};

		#[cfg(not(target_os = "windows"))]
		let entry = unsafe { xr::Entry::load() }.expect("Failed to load OpenXR entry!");

		#[cfg(target_os = "android")]
		entry.initialize_android_loader().expect("Failed to initialize Android loader for OpenXR!");

		let (instance, exts) = {
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
			exts.ext_debug_utils = cfg!(debug_assertions);
			exts.fb_display_refresh_rate = cfg!(target_os = "android");// TODO: Implement.

			let layers = Self::LAYERS
				.iter()
				.filter_map(|&layer| {
					let available = available_layers.iter().any(|xr::ApiLayerProperties { layer_name, .. }| layer_name.as_str() == layer);

					#[cfg(debug_assertions)]
					if !available {
						println!("OpenXR layer {layer} is unavailable! Removing from list of layers.");
					}

					available.then(|| layer)
				})
				.collect::<Vec<_>>();

			let instance = entry.create_instance(&app_info, &exts, &layers).expect("Failed to create OpenXR instance!");

			let exts = Extensions {
				#[cfg(debug_assertions)]
				debug_utils: exts.ext_debug_utils.then(|| {
					let debug_utils = unsafe { xr::raw::DebugUtilsEXT::load(&entry, instance.as_raw()) }.expect("Failed to load OpenXR DebugUtilsEXT!");

					let create_info = xr::sys::DebugUtilsMessengerCreateInfoEXT {
						ty: xr::sys::DebugUtilsMessengerCreateInfoEXT::TYPE,
						next: std::ptr::null(),
						message_severities:
							xr::sys::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
								| xr::sys::DebugUtilsMessageSeverityFlagsEXT::INFO
								| xr::sys::DebugUtilsMessageSeverityFlagsEXT::WARNING
								| xr::sys::DebugUtilsMessageSeverityFlagsEXT::ERROR,
						message_types:
							xr::sys::DebugUtilsMessageTypeFlagsEXT::GENERAL
								| xr::sys::DebugUtilsMessageTypeFlagsEXT::VALIDATION
								| xr::sys::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
								| xr::sys::DebugUtilsMessageTypeFlagsEXT::CONFORMANCE,
						user_callback: Some(dbg_messenger_callback),
						user_data: std::ptr::null_mut(),
					};

					let mut messenger = default();
					let result = unsafe { (debug_utils.create_debug_utils_messenger)(instance.as_raw(), &create_info as *const _, &mut messenger) };
					match result {
						xr::sys::Result::SUCCESS => {},
						e => panic!("Failed to create debug messenger!: {e}"),
					}

					DebugUtils {
						fns: debug_utils,
						messenger,
					}
				}),
			};

			(instance, exts)
		};

		// @TODO: This can fail if there's not a VR headset present. Detect this and abort VR play.
		let hmd = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();
		let context = Arc::new(GpuContext::new(vk_exts, Some((&instance, hmd))));

		let this = Self {
			context,
			entry: mem::ManuallyDrop::new(entry),
			handle: mem::ManuallyDrop::new(instance),
			hmd,
			exts,
		};

		#[cfg(debug_assertions)]
		unsafe { this.set_debug_name("instance", this.handle.as_raw().into_raw(), xr::sys::ObjectType::INSTANCE); }

		this
	}

	pub fn create_session(&self, reference_space: xr::ReferenceSpaceType) -> Result<(XrSession, XrFrameWaiter, XrFrameStream, XrSpace), xr::sys::Result> {
		let create_info = xr::vulkan::SessionCreateInfo {
			instance: self.context.instance.handle().as_raw() as *const _,
			physical_device: self.context.physical_device.as_raw() as *const _,
			device: self.context.device.handle().as_raw() as *const _,
			queue_family_index: self.context.queue_families.graphics,
			queue_index: 0,
		};

		let (session, frame_waiter, frame_stream) = unsafe { self.handle.create_session(self.hmd, &create_info) }?;
		let space = session.create_reference_space(reference_space, xr::Posef::IDENTITY)?;

		#[cfg(debug_assertions)]
		unsafe {
			self.set_debug_name("session", session.as_raw().into_raw(), xr::sys::ObjectType::SESSION);
			self.set_debug_name("space", space.as_raw().into_raw(), xr::sys::ObjectType::SPACE);
		}

		Ok((
			XrSession {
				_context: Arc::clone(&self.context),
				handle: mem::ManuallyDrop::new(session),
				hmd: self.hmd,
			},
			XrFrameWaiter {
				_context: Arc::clone(&self.context),
				handle: mem::ManuallyDrop::new(frame_waiter),
			},
			XrFrameStream {
				_context: Arc::clone(&self.context),
				handle: mem::ManuallyDrop::new(frame_stream),
			},
			XrSpace {
				_context: Arc::clone(&self.context),
				handle: mem::ManuallyDrop::new(space),
			},
		))
	}

	#[cfg(debug_assertions)]
	pub unsafe fn set_debug_name(&self, name: &str, handle: u64, object_type: xr::sys::ObjectType) {
		let Some(debug_utils) = &self.exts.debug_utils else {
			return;
		};

		let object_name = unsafe { CString::new(name) }.unwrap();
		let name_info = xr::sys::DebugUtilsObjectNameInfoEXT {
			ty: xr::sys::DebugUtilsObjectNameInfoEXT::TYPE,
			next: std::ptr::null(),
			object_type,
			object_handle: handle,
			object_name: object_name.as_ptr(),
		};

		let result = unsafe { (debug_utils.set_debug_utils_object_name)(self.handle.as_raw(), &name_info as *const _) };
		if result != xr::sys::Result::SUCCESS {
			error!("Failed to name OpenXR object <{object_type:?}> {name}");
		}
	}

	#[inline(always)]
	pub fn context(&self) -> &Arc<GpuContext> {
		&self.context
	}
}

#[cfg(debug_assertions)]
impl Drop for XrInstance {
	fn drop(&mut self) {
		let Some(debug_utils) = &self.exts.debug_utils else {
			return;
		};

		let result = unsafe { (debug_utils.destroy_debug_utils_messenger)(debug_utils.messenger) };
		assert_eq!(result, xr::sys::Result::SUCCESS, "Failed to destroy debug messenger!");

		rhi_hush!();

		unsafe {
			mem::ManuallyDrop::drop(&mut self.handle);
			mem::ManuallyDrop::drop(&mut self.entry);
		}
	}
}

pub struct Extensions {
	#[cfg(debug_assertions)]
	pub debug_utils: Option<DebugUtils>,
}


#[derive(Deref, DerefMut)]
pub struct DebugUtils {
	#[deref]
	pub fns: xr::raw::DebugUtilsEXT,
	pub messenger: xr::sys::DebugUtilsMessengerEXT,
}

#[derive(Resource, Deref)]
pub struct XrSession {
	_context: Arc<GpuContext>,// In order to keep the context alive longer than the session.
	#[deref]
	pub handle: mem::ManuallyDrop<xr::Session<xr::Vulkan>>,
	pub hmd: xr::SystemId,
}

impl Drop for XrSession {
	fn drop(&mut self) {
		rhi_hush!();

		unsafe {
			mem::ManuallyDrop::drop(&mut self.handle);
		}
	}
}

#[derive(Resource, Deref, DerefMut)]
pub struct XrFrameWaiter {
	_context: Arc<GpuContext>,
	#[deref]
	pub handle: mem::ManuallyDrop<xr::FrameWaiter>,
}

impl Drop for XrFrameWaiter {
	fn drop(&mut self) {
		rhi_hush!();

		unsafe {
			mem::ManuallyDrop::drop(&mut self.handle);
		}
	}
}

#[derive(Resource, Deref, DerefMut)]
pub struct XrFrameStream {
	_context: Arc<GpuContext>,
	#[deref]
	pub handle: mem::ManuallyDrop<xr::FrameStream<xr::Vulkan>>,
}

impl Drop for XrFrameStream {
	fn drop(&mut self) {
		rhi_hush!();

		unsafe {
			mem::ManuallyDrop::drop(&mut self.handle);
		}
	}
}

#[derive(Resource, Deref)]
pub struct XrSpace {
	_context: Arc<GpuContext>,
	#[deref]
	pub handle: mem::ManuallyDrop<xr::Space>,
}

impl Drop for XrSpace {
	fn drop(&mut self) {
		rhi_hush!();

		unsafe {
			mem::ManuallyDrop::drop(&mut self.handle);
		}
	}
}

#[derive(Resource)]
pub struct XrSwapchain {
	name: DebugString,
	cx: Arc<GpuContext>,// Must be destroyed last.
	pub render_pass: Arc<RenderPass>,
	pub handle: mem::ManuallyDrop<xr::Swapchain<xr::Vulkan>>,
	pub frames: Box<[Frame]>,
	pub swapchain_frame_buffers: Box<[SwapchainFrameBuffer]>,
	pub current_frame_index: usize,
	pub current_frame_buffer_index: usize,
	pub resolution: UVec2,
	pub format: Format,
}

impl XrSwapchain {
	pub fn new(
		name: DebugString,
		cx: Arc<GpuContext>,
		render_pass: Arc<RenderPass>,
		session: &XrSession,
		desc: &XrSwapchainDesc,
	) -> Self {
		assert_ne!(desc.frames_in_flight, 0);

		// Hard-coded for multiview.
		const MULTIVIEW_ARRAY_COUNT: u32 = 2;
		const MULTIVIEW_MASK: u32 = 0b11;
		const DEPTH_FORMAT: Format = Format::DF32;

		let instance = session.instance();
		let (format, resolution) = {
			assert!(instance.enumerate_view_configurations(session.hmd).unwrap().contains(&xr::ViewConfigurationType::PRIMARY_STEREO),
							"Primary stereo view configuration not supported!"
			);

			let view_config = match instance.enumerate_view_configuration_views(session.hmd, xr::ViewConfigurationType::PRIMARY_STEREO).unwrap().as_slice() {
				&[a, b] if a == b => a,
				e => panic!("Expected two view configurations that are equivalent! Got: {e:?}!"),
			};

			let formats = session.enumerate_swapchain_formats().expect("Failed to enumerate OpenXR swapchain formats!");
			let format = formats
				.iter()
				.map(|&format| Format::from(vk::Format::from_raw(format as _)))
				.find(|&format| format == desc.fmt.unwrap_or(Format::RgbaU8Srgb))
				.unwrap_or_else(|| Format::from(vk::Format::from_raw(formats[0] as _)));

			(format, UVec2::new(view_config.recommended_image_rect_width, view_config.recommended_image_rect_height))
		};

		let swapchain = {
			let create_info = xr::SwapchainCreateInfo::<xr::Vulkan> {
				create_flags: xr::SwapchainCreateFlags::EMPTY,
				usage_flags: desc.usage,
				format: vk::Format::R8G8B8A8_SRGB.as_raw() as _,//<_ as Into<vk::Format>>::into(format).as_raw() as _,
				sample_count: 1,
				width: resolution.x,
				height: resolution.y,
				face_count: 1,
				array_size: MULTIVIEW_ARRAY_COUNT,
				mip_count: 1,
			};

			session.create_swapchain(&create_info).expect("Failed to create OpenXR swapchain!")
		};

		let swapchain_images = swapchain.enumerate_images().expect("Failed to enumerate OpenXR swapchain images!");

		/*desc.context.immediate_cmd(desc.context.queue_families.graphics, |cmd| {
			let barriers = swapchain_images
				.iter()
				.map(|&image| vk::Image::from_raw(image))
				.map(|image| vk::ImageMemoryBarrier::default()
					.src_access_mask(vk::AccessFlags::empty())
					.dst_access_mask(vk::AccessFlags::empty())
					.old_layout(vk::ImageLayout::PREINITIALIZED)
					.new_layout(vk::ImageLayout::UNDEFINED)
					.src_queue_family_index(desc.context.queue_families.graphics)
					.dst_queue_family_index(desc.context.queue_families.graphics)
					.image(image)
					.subresource_range(vk::ImageSubresourceRange::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.level_count(1)
						.layer_count(2)
					)
				)
				.collect::<Vec<_>>();

			unsafe {
				desc.context.device.cmd_pipeline_barrier(
					cmd,
					vk::PipelineStageFlags::BOTTOM_OF_PIPE,
					vk::PipelineStageFlags::TOP_OF_PIPE,
					vk::DependencyFlags::empty(),
					&[],
					&[],
					&barriers,
				);
			}
		});*/

		let swapchain_frame_buffers = swapchain_images
			.iter()
			.map(|&image| vk::Image::from_raw(image))
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
							.layer_count(MULTIVIEW_ARRAY_COUNT)
						);

					unsafe { cx.device.create_image_view(&create_info, None) }.expect("Failed to create image view!")
				};

				let msaa_color_attachment = match desc.msaa {
					Msaa::None => None,
					msaa => Some(Image::new(dbgfmt!("msaa_color_attachment[{i}]"), Arc::clone(&cx), &ImageCreateInfo {
						resolution,
						mip_count: MipCount::None,
						format,
						usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
						initial_layout: vk::ImageLayout::UNDEFINED,
						tiling: vk::ImageTiling::OPTIMAL,
						msaa,
						layers: MULTIVIEW_ARRAY_COUNT,
						memory: MemoryLocation::GpuOnly,
					})),
				};

				let depth_attachment = Image::new(dbgfmt!("depth_attachment[{i}]"), Arc::clone(&cx), &ImageCreateInfo {
					resolution,
					mip_count: MipCount::None,
					format: DEPTH_FORMAT.into(),
					usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
					initial_layout: vk::ImageLayout::UNDEFINED,
					tiling: vk::ImageTiling::OPTIMAL,
					msaa: desc.msaa,
					layers: MULTIVIEW_ARRAY_COUNT,
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

					unsafe { cx.device.create_framebuffer(&create_info, None) }.expect("Failed to create frame buffer!")
				};

				#[cfg(debug_assertions)]
				unsafe {
					cx.set_debug_name(format!("xr_swapchain_image[{i}]").as_str(), image);
					cx.set_debug_name(format!("xr_swapchain_image_view[{i}]").as_str(), image_view);
					cx.set_debug_name(format!("xr_swapchain_frame_buffer[{i}]").as_str(), frame_buffer);
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

		let frames_in_flight = desc.frames_in_flight.min(swapchain_frame_buffers.len() as _);

		let frames = (0..frames_in_flight)
			.map(|i| {
				let fence = {
					let create_info = vk::FenceCreateInfo::default()
						.flags(vk::FenceCreateFlags::SIGNALED);

					unsafe { cx.device.create_fence(&create_info, None) }.unwrap()
				};

				let graphics_cmd_pool = {
					let create_info = vk::CommandPoolCreateInfo::default()
						.flags(vk::CommandPoolCreateFlags::TRANSIENT)
						.queue_family_index(cx.queue_families.graphics);

					unsafe { cx.device.create_command_pool(&create_info, None) }.unwrap()
				};

				let graphics_cmd = {
					let create_info = vk::CommandBufferAllocateInfo::default()
						.level(vk::CommandBufferLevel::PRIMARY)
						.command_pool(graphics_cmd_pool)
						.command_buffer_count(1);

					unsafe { cx.device.allocate_command_buffers(&create_info) }.unwrap()[0]
				};

				#[cfg(debug_assertions)]
				unsafe {
					cx.set_debug_name(format!("frame_cmd_pool[{i}]").as_str(), graphics_cmd_pool);
					cx.set_debug_name(format!("frame_fence[{i}]").as_str(), fence);
				}

				Frame {
					graphics_cmd_pool,
					graphics_cmd,
					in_flight_fence: fence,
				}
			})
			.collect::<Box<_>>();

		Self {
			name,
			cx,
			render_pass,
			handle: mem::ManuallyDrop::new(swapchain),
			frames,
			swapchain_frame_buffers,
			current_frame_index: 0,
			current_frame_buffer_index: 0,
			resolution,
			format,
		}
	}

	#[inline(always)]
	pub fn resolution(&self) -> UVec2 {
		self.resolution
	}

	#[inline(always)]
	pub fn format(&self) -> Format {
		self.format
	}

	#[inline(always)]
	pub fn frame_in_flight(&self) -> &Frame {
		&self.frames[self.current_frame_index]
	}
}

impl Drop for XrSwapchain {
	fn drop(&mut self) {
		// Wait for GPU to go idle before destroying this.
		// Very expensive but this shouldn't happen often.
		self.cx.wait_idle();

		for frame in self.frames.iter() {
			unsafe {
				self.cx.device.destroy_fence(frame.in_flight_fence, None);
				self.cx.device.destroy_command_pool(frame.graphics_cmd_pool, None);
			}
		}

		for frame_buffer in self.swapchain_frame_buffers.iter_mut() {
			unsafe {
				drop(frame_buffer.msaa_color_attachment.take());
				mem::ManuallyDrop::drop(&mut frame_buffer.depth_attachment);
				self.cx.device.destroy_image_view(frame_buffer.image_view, None);
				self.cx.device.destroy_framebuffer(frame_buffer.frame_buffer, None);
			}
		}

		rhi_hush!();

		unsafe {
			mem::ManuallyDrop::drop(&mut self.handle);
		}
	}
}

pub struct XrSwapchainDesc {
	pub frames_in_flight: u32,
	pub msaa: Msaa,
	pub fmt: Option<Format>,
	pub usage: xr::SwapchainUsageFlags,
}

pub struct Frame {
	pub graphics_cmd_pool: vk::CommandPool,
	pub graphics_cmd: vk::CommandBuffer,
	pub in_flight_fence: vk::Fence,
}

pub struct SwapchainFrameBuffer {
	pub image: vk::Image,// Render target without msaa. Resolve image with msaa.
	pub image_view: vk::ImageView,
	pub msaa_color_attachment: Option<Image>,
	pub depth_attachment: mem::ManuallyDrop<Image>,
	pub frame_buffer: vk::Framebuffer,
}

thread_local! {
	// If this value is > 0. Vulkan debug messages will be disabled.
	#[cfg(debug_assertions)]
	static IGNORE_CALLBACK_DEPTH: Cell<usize> = Cell::new(0);
}

pub struct XrHush {
	_phantom: PhantomData<std::rc::Rc<()>>,// Forces !Send and !Copy.
}

impl XrHush {
	#[must_use]
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
impl Drop for XrHush {
	#[inline]
	fn drop(&mut self) {
		IGNORE_CALLBACK_DEPTH.set(IGNORE_CALLBACK_DEPTH.get() - 1);
	}
}

unsafe extern "system" fn dbg_messenger_callback(
	severity: xr::sys::DebugUtilsMessageSeverityFlagsEXT,
	msg_type: xr::sys::DebugUtilsMessageTypeFlagsEXT,
	data: *const xr::sys::DebugUtilsMessengerCallbackDataEXT,
	user_data: *mut c_void,
) -> xr::sys::Bool32 {
	#[cfg(debug_assertions)]
	if IGNORE_CALLBACK_DEPTH.get() > 0 {
		return xr::sys::FALSE;
	}

	assert_ne!(data, std::ptr::null());
	let data = unsafe { &*data };
	let fn_name = unsafe { CStr::from_ptr(data.function_name) }.to_str().unwrap();
	let msg = unsafe { CStr::from_ptr(data.message) }.to_str().unwrap();

	#[derive(Debug)]
	pub enum MsgType {
		General,
		Validation,
		Performance,
		Conformance,
	}

	#[derive(Debug)]
	pub enum Severity {
		Verbose,
		Info,
		Warning,
		Error,
	}

	let msg_type = match msg_type {
		xr::sys::DebugUtilsMessageTypeFlagsEXT::GENERAL => MsgType::General,
		xr::sys::DebugUtilsMessageTypeFlagsEXT::VALIDATION => MsgType::Validation,
		xr::sys::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => MsgType::Performance,
		xr::sys::DebugUtilsMessageTypeFlagsEXT::CONFORMANCE => MsgType::Conformance,
		_ => unreachable!("Unknown message type {msg_type:?}!"),
	};

	let severity = match severity {
		xr::sys::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Severity::Verbose,
		xr::sys::DebugUtilsMessageSeverityFlagsEXT::INFO => Severity::Info,
		xr::sys::DebugUtilsMessageSeverityFlagsEXT::WARNING => Severity::Warning,
		xr::sys::DebugUtilsMessageSeverityFlagsEXT::ERROR => Severity::Error,
		_ => unreachable!("Unknown severity {severity:?}!"),
	};

	let msg = format!("XR[{msg_type:?}][{severity:?}][{fn_name}]: {msg}\n");

	match severity {
		Severity::Verbose => debug!("{msg}"),
		Severity::Info => info!("{msg}"),
		Severity::Warning => warn!("{msg}"),
		Severity::Error => error!("{msg}"),
	}

	xr::sys::FALSE
}
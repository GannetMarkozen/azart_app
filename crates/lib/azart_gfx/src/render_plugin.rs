use std::ffi::{CStr, CString};
use std::mem::offset_of;
use std::num::NonZero;
use std::ops::Deref;
use std::{mem, slice};
use std::path::Path;
use std::slice::Windows;
use bevy::prelude::*;
use std::sync::Arc;
use bevy::window::{PresentMode, PrimaryWindow, RequestRedraw, WindowCreated, WindowEvent, WindowResized};
use bevy::winit::{WakeUp, WinitWindows};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use super::{GpuContext, GpuContextHandle, MipCount, SwapchainCreateInfo};
use super::Swapchain;
use ash::vk;
use bevy::reflect::{DynamicTypePath, TypeRegistry};
use image::GenericImageView;
use crate::graphics_pipeline::{GraphicsPipeline, GraphicsPipelineCreateInfo, VertexAttribute, VertexInput};
use crate::image::{Image, ImageCreateInfo};
use azart_utils::io;
use azart_gfx_utils::{Format, MsaaCount, ShaderPath, TriangleFillMode};
use azart_utils::debug_string::*;
use azart_gfx_utils::{shader_path};
use bevy::asset::io::VecReader;
use bevy::tasks::{block_on, ComputeTaskPool, ParallelSliceMut, TaskPool};
use either::Either;
use gpu_allocator::MemoryLocation;
use std140::repr_std140;
use crate::buffer::{Buffer, BufferCreateInfo};
use crate::render_settings::*;
use crate::xr::{XrInstance, XrSession, XrState};
use openxr as xr;
use vk_sync::*;
/*#[repr_std140]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vertex {
	pub pos: std140::vec3,
	pub uv: std140::vec2,
}

impl Vertex {
	#[inline]
	pub const fn new(pos: Vec3, uv: Vec2) -> Self {
		Self {
			pos: std140::vec3(pos.x, pos.y, pos.z),
			uv: std140::vec2(uv.x, uv.y),
		}
	}
}*/

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vertex {
	pub pos: Vec3,
	pub uv: Vec2,
}

impl Vertex {
	#[inline]
	pub const fn new(pos: Vec3, uv: Vec2) -> Self {
		Self {
			pos,
			uv,
		}
	}
}

#[repr_std140]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct ViewMatrices {
	pub model: std140::mat4x4,
	pub view: std140::mat4x4,
	pub proj: std140::mat4x4,
}

pub const CUBE_VERTICES: [Vertex; 24] = [
	// -------------------------
	// FRONT FACE (+Z)
	Vertex::new(Vec3::new(-1.0, 1.0,  1.0), Vec2::new(0.0, 0.0)), // 0
	Vertex::new(Vec3::new( 1.0, 1.0,  1.0), Vec2::new(1.0, 0.0)), // 1
	Vertex::new(Vec3::new( 1.0,  -1.0,  1.0), Vec2::new(1.0, 1.0)), // 2
	Vertex::new(Vec3::new(-1.0,  -1.0,  1.0), Vec2::new(0.0, 1.0)), // 3

	// -------------------------
	// RIGHT FACE (+X)
	Vertex::new(Vec3::new( 1.0, 1.0,  1.0), Vec2::new(0.0, 0.0)), // 4
	Vertex::new(Vec3::new( 1.0, 1.0, -1.0), Vec2::new(1.0, 0.0)), // 5
	Vertex::new(Vec3::new( 1.0,  -1.0, -1.0), Vec2::new(1.0, 1.0)), // 6
	Vertex::new(Vec3::new( 1.0,  -1.0,  1.0), Vec2::new(0.0, 1.0)), // 7

	// -------------------------
	// BACK FACE (-Z)
	Vertex::new(Vec3::new( 1.0, 1.0, -1.0), Vec2::new(0.0, 0.0)), // 8
	Vertex::new(Vec3::new(-1.0, 1.0, -1.0), Vec2::new(1.0, 0.0)), // 9
	Vertex::new(Vec3::new(-1.0,  -1.0, -1.0), Vec2::new(1.0, 1.0)), // 10
	Vertex::new(Vec3::new( 1.0,  -1.0, -1.0), Vec2::new(0.0, 1.0)), // 11

	// -------------------------
	// LEFT FACE (-X)
	Vertex::new(Vec3::new(-1.0, 1.0, -1.0), Vec2::new(0.0, 0.0)), // 12
	Vertex::new(Vec3::new(-1.0, 1.0,  1.0), Vec2::new(1.0, 0.0)), // 13
	Vertex::new(Vec3::new(-1.0,  -1.0,  1.0), Vec2::new(1.0, 1.0)), // 14
	Vertex::new(Vec3::new(-1.0,  -1.0, -1.0), Vec2::new(0.0, 1.0)), // 15

	// -------------------------
	// TOP FACE (+Y)
	Vertex::new(Vec3::new(-1.0,  1.0,  1.0), Vec2::new(0.0, 0.0)), // 16
	Vertex::new(Vec3::new( 1.0,  1.0,  1.0), Vec2::new(1.0, 0.0)), // 17
	Vertex::new(Vec3::new( 1.0,  1.0, -1.0), Vec2::new(1.0, 1.0)), // 18
	Vertex::new(Vec3::new(-1.0,  1.0, -1.0), Vec2::new(0.0, 1.0)), // 19

	// -------------------------
	// BOTTOM FACE (-Y)
	Vertex::new(Vec3::new(-1.0, -1.0, -1.0), Vec2::new(0.0, 0.0)), // 20
	Vertex::new(Vec3::new( 1.0, -1.0, -1.0), Vec2::new(1.0, 0.0)), // 21
	Vertex::new(Vec3::new( 1.0, -1.0,  1.0), Vec2::new(1.0, 1.0)), // 22
	Vertex::new(Vec3::new(-1.0, -1.0,  1.0), Vec2::new(0.0, 1.0)), // 23
];

/// 36 indices (12 triangles) referencing the 24 unique vertices above
pub const CUBE_INDICES: [u16; 36] = [
	// Front face
	0,  1,  2,
	0,  2,  3,
	// Right face
	4,  5,  6,
	4,  6,  7,
	// Back face
	8,  9,  10,
	8,  10, 11,
	// Left face
	12, 13, 14,
	12, 14, 15,
	// Top face
	16, 17, 18,
	16, 18, 19,
	// Bottom face
	20, 21, 22,
	20, 22, 23,
];

#[derive(Copy, Clone, Resource, Deref)]
pub struct PreviousXrState(pub xr::SessionState);


#[derive(Clone, Debug)]
pub struct RenderPlugin {
	pub settings: RenderSettings,
	pub display_mode: DisplayMode,
}

impl Default for RenderPlugin {
	fn default() -> Self {
		Self {
			settings: default(),
			display_mode: DisplayMode::Xr,
		}
	}
}

impl Plugin for RenderPlugin {
	fn build(&self, app: &mut App) {
		assert!(self.settings.frames_in_flight > 0);

		let event_loop = app.world().get_non_send_resource::<EventLoop<WakeUp>>().expect("event loop not found!");
		let extensions = ash_window::enumerate_required_extensions(event_loop.display_handle().unwrap().as_raw())
			.expect("Failed to enumerate required extensions!")
			.iter()
			.map(|&x| unsafe { CStr::from_ptr(x) })
			.collect::<Vec<_>>();

		let xr = (self.display_mode == DisplayMode::Xr).then(|| XrInstance::new());
		let context = Arc::new(GpuContext::new(&extensions, xr));

		let mut settings = self.settings.clone();

		// @NOTE: This should really only be launched when you want to play in VR.
		if let Ok(session) = XrSession::new(Arc::clone(&context)) {
			// We don't get to decide the number of frames in flight with OpenXR.
			settings.frames_in_flight = session.swapchain.frames.len();

			app.add_systems(PreUpdate, create_global_resources_once_for_xr);

			app.insert_resource(session);
			app.insert_state(XrState::Idle);
			app.insert_resource(PreviousXrState(xr::SessionState::IDLE));
		}

		app
			.insert_resource(GpuContextHandle::new(context))
			.insert_state(self.display_mode)
			.insert_resource(settings)
			.insert_state(self.display_mode)
			.add_systems(PreUpdate, create_swapchain_on_window_spawned);
			//.add_systems(PreUpdate, create_render_pass_on_primary_window_spawned);

		// TMP
		app
			.add_systems(Startup, init)
			.add_systems(Update, render.run_if(in_state(DisplayMode::Standard)))
			.add_systems(Update,
				(
					poll_xr_state,
					render_xr.run_if(in_state(XrState::Focused)),
				)
					.chain()
					.run_if(in_state(DisplayMode::Xr))
			);

		app
			.add_systems(Update, |query: Query<&Window, With<PrimaryWindow>>, mut events: EventReader<WindowEvent>| {
				for event in events.read() {
					//println!("WINDOW_EVENT: {event:?}");
				}
			});
	}
}

#[derive(Resource)]
pub struct IndirectArgs {
	pub args_buffer: Buffer,
	pub count_buffer: Buffer,
}

#[derive(Resource)]
pub struct Texture(pub Image);

#[derive(Resource)]
pub struct Model {
	pub name: DebugString,
	pub index_buffer: Buffer,
	pub vertex_buffer: Buffer,
	pub view_matrices_buffer: Buffer,
}

#[derive(Resource)]
pub struct BasePass {
	name: DebugString,
	render_pass: vk::RenderPass,
	frame_buffer: vk::Framebuffer,
	msaa_color_attachment: Option<Image>,// Only set if msaa is enabled. color_attachment is the resolve image.
	color_attachment: Image,
	depth_attachment: Image,
	context: Arc<GpuContext>,
}

impl BasePass {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		resolution: UVec2,
		layers: u32,
		msaa: MsaaCount,
	) -> Self {
		let msaa_color_attachment = match msaa.enabled() {
			true => Some({
				let create_info = ImageCreateInfo {
					resolution,
					format: Format::RgbaU8Norm,
					usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
					msaa,
					layers,
					..default()
				};

				Image::new(dbgfmt!("msaa_{name}_color_attachment"), Arc::clone(&context), &create_info)
			}),
			false => None,
		};

		let color_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: Format::RgbaU8Norm,
				usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
				msaa: MsaaCount::None,// Always 1. If Msaa is enabled this is the resolve image.
				layers,
				..default()
			};

			Image::new(dbgfmt!("{name}_color_attachment"), Arc::clone(&context), &create_info)
		};

		let depth_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: Format::DF32,
				usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
				msaa,
				layers,
				..default()
			};
			
			Image::new(dbgfmt!("{name}_depth_stencil"), Arc::clone(&context), &create_info)
		};

		let render_pass = {
			let attachments = match &msaa_color_attachment {
				Some(msaa_color_attachment) => vec![
					vk::AttachmentDescription2KHR::default()
						.format(msaa_color_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(msaa.as_vk_sample_count()),
					vk::AttachmentDescription2KHR::default()
						.format(depth_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(msaa.as_vk_sample_count()),
					vk::AttachmentDescription2KHR::default()
						.format(color_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::DONT_CARE)
						.samples(vk::SampleCountFlags::TYPE_1),
				],
				None => vec![
					vk::AttachmentDescription2KHR::default()
						.format(color_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)// TODO: Conditionally use DONT_CARE on certain platforms perchance.
						.samples(vk::SampleCountFlags::TYPE_1),
					vk::AttachmentDescription2KHR::default()
						.format(depth_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(vk::SampleCountFlags::TYPE_1),
				]
			};
			
			let color_attachment_ref = vk::AttachmentReference2KHR::default()
				.attachment(0)
				.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
				.aspect_mask(vk::ImageAspectFlags::COLOR);
			
			let depth_attachment_ref = vk::AttachmentReference2KHR::default()
				.attachment(1)
				.layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
				.aspect_mask(vk::ImageAspectFlags::DEPTH);

			// Only needed for Msaa.
			let resolve_color_attachment_ref = vk::AttachmentReference2KHR::default()
				.attachment(2)
				.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
				.aspect_mask(vk::ImageAspectFlags::COLOR);

			let color_attachments = match msaa.enabled() {
				true => vec![
					color_attachment_ref,
					resolve_color_attachment_ref,
				],
				false => vec![
					color_attachment_ref
				],
			};

			assert!((0..=32).contains(&layers), "Invalid number of layers: {layers}.");
			let view_mask = !0 >> (32 - layers);

			let subpass = match msaa.enabled() {
				true => vk::SubpassDescription2KHR::default()
					.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
					.color_attachments(&color_attachments)
					.resolve_attachments(slice::from_ref(&resolve_color_attachment_ref))
					.depth_stencil_attachment(&depth_attachment_ref)
					.view_mask(view_mask),
				false => vk::SubpassDescription2KHR::default()
					.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
					.color_attachments(&color_attachments)
					.depth_stencil_attachment(&depth_attachment_ref)
					.view_mask(view_mask),
			};
			
			let dependencies = vk::SubpassDependency2KHR::default()
				.src_subpass(vk::SUBPASS_EXTERNAL)
				.dst_subpass(0)
				.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
				.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
				.src_access_mask(vk::AccessFlags::empty())
				.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

			let create_info = vk::RenderPassCreateInfo2KHR::default()
				.attachments(&attachments)
				.subpasses(&slice::from_ref(&subpass))
				.dependencies(slice::from_ref(&dependencies))
				.correlated_view_masks(slice::from_ref(&view_mask));

			unsafe { context.extensions.create_render_pass2.create_render_pass2(&create_info, None) }.unwrap()
		};
		
		let frame_buffer = {
			let attachments = match &msaa_color_attachment {
				Some(msaa_color_attachment) => vec![
					msaa_color_attachment.view,
					depth_attachment.view,
					color_attachment.view,// color_attachment is the resolve image in this case.
				],
				None => vec![
					color_attachment.view,
					depth_attachment.view,
				],
			};
			
			let create_info = vk::FramebufferCreateInfo::default()
				.render_pass(render_pass)
				.width(resolution.x)
				.height(resolution.y)
				.attachments(&attachments)
				.layers(layers);
			
			unsafe { context.device.create_framebuffer(&create_info, None) }.unwrap()
		};
		
		Self {
			name,
			render_pass,
			frame_buffer,
			msaa_color_attachment,
			color_attachment,
			depth_attachment,
			context,
		}
	}

	// Must ensure the resources being destroyed are not in use.
	// TODO: Would be resolved with frame dependency reference-counting.
	pub unsafe fn resize(&mut self, resolution: UVec2) {
		self.context.device.destroy_framebuffer(self.frame_buffer, None);

		self.msaa_color_attachment = self.msaa_color_attachment.as_ref().map(|x| {
			let create_info = ImageCreateInfo {
				resolution,
				format: x.format,
				usage: x.usage,
				msaa: x.msaa,
				..default()
			};

			Image::new(x.name().clone(), Arc::clone(&self.context), &create_info)
		});

		self.color_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: self.color_attachment.format,
				usage: self.color_attachment.usage,
				msaa: MsaaCount::None,
				..default()
			};

			Image::new(self.color_attachment.name().clone(), Arc::clone(&self.context), &create_info)
		};

		self.depth_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: self.depth_attachment.format,
				usage: self.depth_attachment.usage,
				msaa: self.depth_attachment.msaa,
				..default()
			};

			Image::new(self.depth_attachment.name().clone(), Arc::clone(&self.context), &create_info)
		};

		let attachments = match &self.msaa_color_attachment {
			Some(msaa_color_attachment) => vec![
				msaa_color_attachment.view,
				self.depth_attachment.view,
				self.color_attachment.view,
			],
			None => vec![
				self.color_attachment.view,
				self.depth_attachment.view,
			],
		};

		let create_info = vk::FramebufferCreateInfo::default()
			.render_pass(self.render_pass)
			.width(resolution.x)
			.height(resolution.y)
			.attachments(&attachments)
			.layers(1);

		self.frame_buffer = self.context.device.create_framebuffer(&create_info, None).unwrap();
	}
}

impl Drop for BasePass {
	fn drop(&mut self) {
		unsafe {
			self.context.device.destroy_framebuffer(self.frame_buffer, None);
			self.context.device.destroy_render_pass(self.render_pass, None);
		}
	}
}

#[derive(Resource)]
pub struct BaseMaterial {
	pipeline: GraphicsPipeline,
	descriptor_pool: vk::DescriptorPool,
	descriptor_sets: Vec<vk::DescriptorSet>,// 1 descriptor set per frame-in-flight.
}

fn init(
	mut commands: Commands,
	mut query: Query<&mut Swapchain>,
) {

}

fn poll_xr_state(
	mut commands: Commands,
	mut previous_xr_state: ResMut<PreviousXrState>,
	xr_state: Res<State<XrState>>,
	session: Res<XrSession>,
	context: Res<GpuContextHandle>,
	mut exit: EventWriter<AppExit>,
) {
	let mut buffer = default();
	while let Some(event) = context.xr().unwrap().instance.poll_event(&mut buffer).expect("Failed to poll event.") {
		match event {
			xr::Event::SessionStateChanged(state) => {
				let state = state.state();

				match state {
					xr::SessionState::FOCUSED | xr::SessionState::READY => commands.set_state(XrState::Focused),
					xr::SessionState::IDLE | xr::SessionState::STOPPING | xr::SessionState::LOSS_PENDING => commands.set_state(XrState::Idle),
					_ => {}
				}

				match (previous_xr_state.0, state) {
					(xr::SessionState::STOPPING, xr::SessionState::IDLE) => continue,
					(_, xr::SessionState::IDLE) => {
						std::thread::sleep(std::time::Duration::from_millis(100));
						continue;
					},
					(xr::SessionState::IDLE | xr::SessionState::STOPPING, xr::SessionState::READY) => {
						info!("Beginning session!");
						session.handle.begin(xr::ViewConfigurationType::PRIMARY_STEREO).expect("Failed to begin session.");
					},
					(_, xr::SessionState::STOPPING | xr::SessionState::LOSS_PENDING) => {
						info!("Ending session...");
						session.handle.end().expect("Failed to end session.");
						info!("Session ended.");
					},
					(_, xr::SessionState::EXITING) => {
						info!("Exiting App!");
						//exit.send(AppExit::Success);
					},
					_ => {}
				}

				println!("STATE_CHANGE: {:?} to {:?}", previous_xr_state.0, state);

				previous_xr_state.0 = state;
			}
			_ => println!("Unknown event"),
		}
	}
}

/*fn render_xr(
	context: Res<GpuContextHandle>,
	mut session: ResMut<XrSession>,
	time: Res<Time>,
	mut base_pass: Option<ResMut<BasePass>>,
	base_material: Option<Res<BaseMaterial>>,
	model: Option<Res<Model>>,
	indirect_args: Option<Res<IndirectArgs>>,
	mut query: Query<(&Window, &mut Swapchain)>,
	settings: Res<RenderSettings>,
) {
	let (Some(mut base_pass), Some(base_material), Some(model), Some(indirect_args)) = (base_pass, base_material, model, indirect_args) else {
		return;
	};

	// Bind jni env to thread (optimization).
	// Ideally should only run once per thread.
	#[cfg(target_os = "android")]
	{
		let ctx = ndk_context::android_context();
		let vm = unsafe { jni::JavaVM::from_raw(ctx.vm().cast()) }.unwrap();
		let env = vm.attach_current_thread_as_daemon();
	}

	let predicted_display_time = match session.swapchain.frame_waiter.wait() {
		Ok(xr::FrameState { predicted_display_time, .. }) => predicted_display_time,
		Err(xr::sys::Result::ERROR_SESSION_NOT_READY | xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => return,// Session not running. Don't render.
		Err(e) => panic!("Failed to wait for frame: {e}"),
	};

	//let xr::FrameState { predicted_display_time, .. } = session.swapchain.frame_waiter.wait().expect("Failed to wait for frame.");
	session.swapchain.frame_stream.begin().expect("Failed to begin frame.");

	unsafe {
		let frame = &session.swapchain.frames[session.swapchain.current_frame_index()];
		context.device.wait_for_fences(slice::from_ref(&frame.in_flight_fence), true, u64::MAX).unwrap();
		context.device.reset_fences(slice::from_ref(&frame.in_flight_fence)).unwrap();
	}

	let previous_frame_index = session.swapchain.current_frame_index();
	let previous_frame = &session.swapchain.frames[previous_frame_index];

	let frame_index = session.swapchain.handle.acquire_image().expect("Failed to acquire image.") as usize;
	session.swapchain.handle.wait_image(xr::Duration::INFINITE).expect("Failed to wait for image.");


	// Render...
	let (flags, views) = session.handle.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, predicted_display_time, &session.space).expect("Failed to locate views.");

	//println!("Flags: {flags:?}");

	/*for (i, &xr::View { pose, fov }) in views.iter().enumerate() {
		println!("view[{i}]: pose: {pose:?}, fov: {fov:?}");
	}*/

	unsafe {
		let frame = &session.swapchain.frames[frame_index];

		context.device.reset_command_buffer(frame.graphics_command_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
		context.device.begin_command_buffer(frame.graphics_command_buffer, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

		/*record(
			frame.graphics_command_buffer,
			&context,
			&base_pass,
			&base_material,
			frame.image,
			session.swapchain.format().into(),
			session.swapchain.extent(),
			&model,
			&indirect_args,
			frame_index,
			frame_index,
			&settings,
		);*/

		// RECORD
		{
			let cmd = frame.graphics_command_buffer;

			//context.cmd_transition_image_layout(cmd, frame.image, session.swapchain.format().into(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, 0..1, 0..2);
			context.pipeline_barrier(
				cmd,
				None,
				&[],
				&[
					ImageBarrier {
						previous_accesses: &[],
						next_accesses: &[AccessType::TransferWrite],
						previous_layout: ImageLayout::Optimal,
						next_layout: ImageLayout::Optimal,
						discard_contents: true,
						src_queue_family_index: context.queue_families.graphics,
						dst_queue_family_index: context.queue_families.graphics,
						image: frame.image,
						range: vk::ImageSubresourceRange::default()
							.level_count(1)
							.layer_count(2)
							.aspect_mask(vk::ImageAspectFlags::COLOR),
					},
				]
			);

			context.device.cmd_clear_color_image(
				cmd,
				frame.image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0], },
				&[
					vk::ImageSubresourceRange::default()
						.level_count(1)
						.layer_count(2)
						.aspect_mask(vk::ImageAspectFlags::COLOR),
				]
			);

			context.pipeline_barrier(
				cmd,
				None,
				&[],
				&[
					ImageBarrier {
						previous_accesses: &[AccessType::TransferWrite],
						next_accesses: &[AccessType::TransferRead],
						previous_layout: ImageLayout::Optimal,
						next_layout: ImageLayout::Optimal,
						discard_contents: false,
						src_queue_family_index: context.queue_families.graphics,
						dst_queue_family_index: context.queue_families.graphics,
						image: frame.image,
						range: vk::ImageSubresourceRange::default()
							.level_count(1)
							.layer_count(2)
							.aspect_mask(vk::ImageAspectFlags::COLOR),
					},
				]
			);

			//context.cmd_transition_image_layout(cmd, frame.image, session.swapchain.format().into(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, 0..1, 0..2);
		}

		context.device.end_command_buffer(frame.graphics_command_buffer).unwrap();

		let queue = context.device.get_device_queue(context.queue_families.graphics, 0);

		let previous_frame = &session.swapchain.frames[previous_frame_index];
		let submit_info = vk::SubmitInfo::default()
			.command_buffers(slice::from_ref(&previous_frame.graphics_command_buffer))
			.wait_semaphores(&[])
			//.wait_semaphores(slice::from_ref(&previous_frame.image_available_semaphore))
			.wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
			.signal_semaphores(&[]);
			//.signal_semaphores(slice::from_ref(&previous_frame.render_finished_semaphore));

		context.device.queue_submit(queue, slice::from_ref(&submit_info), frame.in_flight_fence).expect("Failed to submit frame.");
	}

	session.swapchain.handle.release_image().expect("Failed to release image.");

	println!("Acquired image: {frame_index}");

	session.swapchain.frame_stream.end(
		predicted_display_time,
		xr::EnvironmentBlendMode::OPAQUE,
		&[],
	).expect("Failed to end frame.");
}*/

fn render_xr(
	context: Res<GpuContextHandle>,
	mut session: ResMut<XrSession>,
	base_pass: Res<BasePass>,
	base_material: Res<BaseMaterial>,
	model: Res<Model>,
	indirect_args: Res<IndirectArgs>,
	settings: Res<RenderSettings>,
) {
	// Android optimization.
	#[cfg(target_os = "android")]
	{
		let ctx = ndk_context::android_context();
		let vm = unsafe { jni::JavaVM::from_raw(ctx.vm().cast()) }.unwrap();
		let env = vm.attach_current_thread_as_daemon();
	}

	let formats = session.handle.enumerate_swapchain_formats()
		.unwrap()
		.into_iter()
		.map(|format| vk::Format::from_raw(format as _))
		.collect::<Vec<_>>();

	println!("Swapchain formats: {formats:?}");

	///
	/// BEGIN XR
	///
	let frame_state = match session.swapchain.frame_waiter.wait() {
		Ok(frame_state) => frame_state,
		Ok(xr::FrameState { should_render: false, .. }) | Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => return,
		Err(e) => panic!("Failed to wait for frame: {e}"),
	};

	session.swapchain.frame_stream.begin().unwrap();
	let image_index = session.swapchain.handle.acquire_image().expect("Failed to acquire image!") as usize;
	session.swapchain.handle.wait_image(xr::Duration::INFINITE).expect("Failed to wait for image!");

	let (view_flags, views) = session.handle.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, frame_state.predicted_display_time, &session.space).expect("Failed to locate views!");
	assert_eq!(views.len(), 2);
	let views = [views[0], views[1]];



	///
	/// BEGIN VK
	///

	let frame = &session.swapchain.frames[image_index];
	let cmd = frame.graphics_command_buffer;
	unsafe {
		context.device.wait_for_fences(slice::from_ref(&frame.in_flight_fence), true, u64::MAX).unwrap();
		context.device.reset_fences(slice::from_ref(&frame.in_flight_fence)).unwrap();

		context.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
	}

	///
	/// BEGIN RENDER
	///

	unsafe {
		record(
			cmd,
			&context,
			&base_pass,
			&base_material,
			frame.image,
			session.swapchain.format().into(),
			session.swapchain.extent(),
			&model,
			&indirect_args,
			image_index,
			image_index,
			2,
			&settings,
		)
	}


	/*unsafe {
		/*context.pipeline_barrier(
			cmd,
			None,
			&[],
			&[ImageBarrier {
				previous_accesses: &[],
				next_accesses: &[AccessType::TransferWrite],
				previous_layout: ImageLayout::Optimal,
				next_layout: ImageLayout::Optimal,
				discard_contents: true,
				src_queue_family_index: context.queue_families.graphics,
				dst_queue_family_index: context.queue_families.graphics,
				image: frame.image,
				range: vk::ImageSubresourceRange::default()
					.level_count(1)
					.layer_count(2)
					.aspect_mask(vk::ImageAspectFlags::COLOR),
			}]
		);*/

		context.device.cmd_pipeline_barrier(
			cmd,
			vk::PipelineStageFlags::empty(),
			vk::PipelineStageFlags::TRANSFER,
			vk::DependencyFlags::empty(),
			&[],
			&[],
			&[vk::ImageMemoryBarrier::default()
				.image(frame.image)
				.src_access_mask(vk::AccessFlags::empty())
				.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
				.old_layout(vk::ImageLayout::UNDEFINED)
				.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.subresource_range(vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.level_count(vk::REMAINING_MIP_LEVELS)
					.layer_count(vk::REMAINING_ARRAY_LAYERS),
				)
			]
		);

		context.device.cmd_clear_color_image(
			cmd,
			frame.image,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			&vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0], },
			&[
				vk::ImageSubresourceRange::default()
					.level_count(vk::REMAINING_MIP_LEVELS)
					.layer_count(vk::REMAINING_ARRAY_LAYERS)
					.aspect_mask(vk::ImageAspectFlags::COLOR),
			]
		);

		context.device.cmd_pipeline_barrier(
			cmd,
			vk::PipelineStageFlags::TRANSFER,
			vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
			vk::DependencyFlags::empty(),
			&[],
			&[],
			&[vk::ImageMemoryBarrier::default()
				.image(frame.image)
				.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
				.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
				.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
				.new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
				.subresource_range(vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.level_count(vk::REMAINING_MIP_LEVELS)
					.layer_count(vk::REMAINING_ARRAY_LAYERS),
				)
			]
		);

		/*context.pipeline_barrier(
			cmd,
			None,
			&[],
			&[ImageBarrier {
				previous_accesses: &[AccessType::TransferWrite],
				next_accesses: &[AccessType::TransferRead],
				previous_layout: ImageLayout::Optimal,
				next_layout: ImageLayout::Optimal,
				discard_contents: false,
				src_queue_family_index: context.queue_families.graphics,
				dst_queue_family_index: context.queue_families.graphics,
				image: frame.image,
				range: vk::ImageSubresourceRange::default()
					.level_count(1)
					.layer_count(2)
					.aspect_mask(vk::ImageAspectFlags::COLOR),
			}]
		);*/
	}*/


	///
	/// END VK
	///

	unsafe {
		context.device.end_command_buffer(cmd).unwrap();

		let queue = context.device.get_device_queue(context.queue_families.graphics, 0);
		let submit_info = vk::SubmitInfo::default()
			.command_buffers(slice::from_ref(&cmd));

		context.device.queue_submit(queue, slice::from_ref(&submit_info), frame.in_flight_fence).expect("Failed to submit frame.");
	}


	///
	/// END XR
	///
	session.swapchain.handle.release_image().expect("Failed to release image.");

	let rect = xr::Rect2Di {
		offset: xr::Offset2Di { x: 0, y: 0 },
		extent: xr::Extent2Di {
			width: session.swapchain.extent().x as _,
			height: session.swapchain.extent().y as _,
		},
	};

	let composition_projection_views: [_; 2] = std::array::from_fn(|i| {
		let xr::View { pose, fov } = views[i];
		xr::CompositionLayerProjectionView::new()
			.pose(pose)
			.fov(fov)
			.sub_image(xr::SwapchainSubImage::new()
				.swapchain(unsafe { &*(&session.swapchain.handle as *const _) })// Ignore the borrow checker.
				.image_array_index(i as _)
				.image_rect(rect)
			)
	});

	let layer_projection = xr::CompositionLayerProjection::new()
		.space(unsafe { &*(&session.space as *const _) })// Ignore the borrow checker.
		.views(&composition_projection_views);

	let result = session.swapchain.frame_stream.end(frame_state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[&*layer_projection]);
	match result {
		Ok(()) => {},
		Err(xr::sys::Result::ERROR_POSE_INVALID) => println!("Pose invalid!: {:?}", views.map(|xr::View { pose, fov }| (pose, fov))),
		Err(e) => panic!("Failed to end frame: {e}"),
	}
}

fn render(
	context: Res<GpuContextHandle>,
	time: Res<Time>,
	mut base_pass: Option<ResMut<BasePass>>,
	base_material: Option<Res<BaseMaterial>>,
	model: Option<Res<Model>>,
	indirect_args: Option<Res<IndirectArgs>>,
	mut query: Query<(&Window, &mut Swapchain)>,
	settings: Res<RenderSettings>,
) {
	let (Some(mut base_pass), Some(base_material), Some(model), Some(indirect_args)) = (base_pass, base_material, model, indirect_args) else {
		return;
	};

	unsafe {
		for (window, mut swapchain) in query.iter_mut() {
			let previous_frame_index = swapchain.current_frame_index;
			
			let (new_frame_index, new_image_index, suboptimal) = swapchain.acquire_next_image();
			
			let previous_frame = &swapchain.frames()[previous_frame_index.0];

			let cmd = previous_frame.graphics_command_buffer;
			context.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();

			// Update uniforms.
			{
				let new_model = Mat4::from_rotation_y(((time.elapsed_secs() * 45.0) % 360.0).to_radians());
				let mut new_proj = Mat4::perspective_rh(120_f32.to_radians(), window.width() / window.height(), 0.1, 100.0);
				new_proj.y_axis.y *= -1.0;

				let pre_rotation = match swapchain.current_transform() {
					vk::SurfaceTransformFlagsKHR::ROTATE_90 => Mat4::from_rotation_z(90_f32.to_radians()),
					vk::SurfaceTransformFlagsKHR::ROTATE_180 => Mat4::from_rotation_z(180_f32.to_radians()),
					vk::SurfaceTransformFlagsKHR::ROTATE_270 => Mat4::from_rotation_z(270_f32.to_radians()),
					_ => Mat4::IDENTITY,
				};

				new_proj = pre_rotation * new_proj;

				let per_frame_view_matrices = unsafe { slice::from_raw_parts_mut(model.view_matrices_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut ViewMatrices, settings.frames_in_flight) };
				let ViewMatrices { model, proj, .. } = &mut per_frame_view_matrices[new_frame_index.0];

				unsafe {
					*model = mem::transmute_copy(&new_model);
					*proj = mem::transmute_copy(&new_proj);
				}
			}
			
			{
				let info = vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

				context.device.begin_command_buffer(cmd, &info).unwrap();
			}
			
			let start = std::time::Instant::now();
			
			/*record(
				cmd,
				&context,
				&base_pass,
				&base_material,
				&swapchain,
				&model,
				&indirect_args,
				new_frame_index.0,
				new_image_index.0,
				&settings
			);*/

			record(
				cmd,
				&context,
				&base_pass,
				&base_material,
				swapchain.images()[new_image_index.0].image,
				swapchain.format().into(),
				swapchain.extent(),
				&model,
				&indirect_args,
				new_frame_index.0,
				new_image_index.0,
				1,
				&settings
			);
			
			// End recording.
			context.device.end_command_buffer(cmd).unwrap();

			let queue = context.device.get_device_queue(context.queue_families.graphics, 0);

			let submit_info = vk::SubmitInfo::default()
				.command_buffers(slice::from_ref(&cmd))
				.wait_semaphores(slice::from_ref(&previous_frame.image_available_semaphore))
				.wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
				.signal_semaphores(slice::from_ref(&previous_frame.render_finished_semaphore));
			
			let start = std::time::Instant::now();

			context.device.queue_submit(queue, slice::from_ref(&submit_info), previous_frame.in_flight_fence).unwrap();
			
			let image_indices = [new_image_index.0 as u32];
			
			let present_info = vk::PresentInfoKHR::default()
				.swapchains(slice::from_ref(&swapchain.handle))
				.image_indices(&image_indices)
				.wait_semaphores(slice::from_ref(&previous_frame.render_finished_semaphore));

			let start = std::time::Instant::now();

			let result = context.extensions.swapchain.queue_present(queue, &present_info);
			
			assert_ne!(result, Err(vk::Result::SUBOPTIMAL_KHR));

			// Conditionally recreate swapchain if required.
			if suboptimal || matches!(result, Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR)) {
				info!("Swapchain is not optimal after present! Status: (suboptimal: {suboptimal}, result: {result:?}). Recreating...");

				// Wait for the GPU to go idle before doing anything.
				// @NOTE: This is very expensive...
				context.wait_idle();
				
				swapchain.recreate(None, None);

				base_pass.resize(swapchain.extent());
			}
		}
	}
}

unsafe fn record(
	cmd: vk::CommandBuffer,
	context: &Arc<GpuContext>,
	base_pass: &BasePass,
	base_material: &BaseMaterial,
	image: vk::Image,
	image_format: vk::Format,
	extent: UVec2,
	model: &Model,
	indirect_args: &IndirectArgs,
	frame_index: usize,
	image_index: usize,
	multiview_count: usize,
	settings: &RenderSettings,
) {
	unsafe {
		// Begin render pass.
		{
			let attachment_count = match &base_pass.msaa_color_attachment {
				Some(_) => 3,
				None => 2,
			};

			let clear_values = (0..attachment_count)
				.map(|_| vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })
				.collect::<Vec<_>>();

			let UVec2 { x: width, y: height } = extent;

			let begin_info = vk::RenderPassBeginInfo::default()
				.render_pass(base_pass.render_pass)
				.framebuffer(base_pass.frame_buffer)
				.render_area(vk::Rect2D {
					extent: vk::Extent2D {
						width,
						height,
					},
					..default()
				})
				.clear_values(&clear_values);

			context.device.cmd_begin_render_pass(cmd, &begin_info, vk::SubpassContents::INLINE);
		}

		// Set viewport
		{
			let UVec2 { x: width, y: height } = extent;

			let viewports = [
				vk::Viewport::default()
					.width(width as f32)
					.height(height as f32)
					.min_depth(1.0) // Reverse-Z for better precision afar.
					.max_depth(0.0),
			];

			context.device.cmd_set_viewport(cmd, 0, &viewports);
		}

		// Set scissor.
		{
			let UVec2 { x: width, y: height } = extent;

			let scissors = [
				vk::Rect2D::default()
					.extent(vk::Extent2D {
						width,
						height,
					}),
			];

			context.device.cmd_set_scissor(cmd, 0, &scissors);
		}

		// Bind descriptor set.
		context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, base_material.pipeline.layout, 0, slice::from_ref(&base_material.descriptor_sets[frame_index]), &[]);

		context.device.cmd_bind_index_buffer(cmd, model.index_buffer.handle, 0, vk::IndexType::UINT16);
		context.device.cmd_bind_vertex_buffers(cmd, 0, &[model.vertex_buffer.handle], &[0]);

		context.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, base_material.pipeline.handle);

		//context.device.cmd_draw_indexed(cmd, CUBE_INDICES.len() as u32, 2, 0, 0, 0);

		context.extensions.draw_indirect_count.cmd_draw_indexed_indirect_count(cmd, indirect_args.args_buffer.handle, 0, indirect_args.count_buffer.handle, 0, 3, size_of::<vk::DrawIndexedIndirectCommand>() as u32);

		// End render pass.

		{
			context.device.cmd_end_render_pass(cmd);
		}

		// Copy image to swapchain.
		{
			//context.cmd_transition_image_layout(cmd, image, image_format, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, 0..1, 0..multiview_count as _);
			context.pipeline_barrier(
				cmd,
				None,
				&[],
				&[ImageBarrier {
					previous_accesses: &[],
					next_accesses: &[AccessType::TransferWrite],
					previous_layout: ImageLayout::Optimal,
					next_layout: ImageLayout::Optimal,
					discard_contents: true,
					src_queue_family_index: context.queue_families.graphics,
					dst_queue_family_index: context.queue_families.graphics,
					image,
					range: vk::ImageSubresourceRange::default()
						.level_count(1)
						.layer_count(multiview_count as _)
						.aspect_mask(vk::ImageAspectFlags::COLOR),
				}]
			);

			let regions = [
				vk::ImageCopy::default()
					.extent(vk::Extent3D::default()
						.width(extent.x)
						.height(extent.y)
						.depth(1)
					)
					.src_subresource(vk::ImageSubresourceLayers::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.layer_count(multiview_count as _)
					)
					.dst_subresource(vk::ImageSubresourceLayers::default()
						.aspect_mask(vk::ImageAspectFlags::COLOR)
						.layer_count(multiview_count as _)
					)
			];

			//context.device.cmd_copy_image(cmd, base_pass.color_attachment.handle, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);

			context.pipeline_barrier(
				cmd,
				None,
				&[],
				&[ImageBarrier {
					previous_accesses: &[AccessType::TransferWrite],
					next_accesses: &[AccessType::Present],
					previous_layout: ImageLayout::Optimal,
					next_layout: ImageLayout::Optimal,
					discard_contents: false,
					src_queue_family_index: context.queue_families.graphics,
					dst_queue_family_index: context.queue_families.graphics,
					image,
					range: vk::ImageSubresourceRange::default()
						.level_count(1)
						.layer_count(multiview_count as _)
						.aspect_mask(vk::ImageAspectFlags::COLOR),
				}]
			);

			//context.cmd_transition_image_layout(cmd, image, image_format, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, 0..1, 0..multiview_count as _);
		}
	}
}

fn create_swapchain_on_window_spawned(
	mut commands: Commands,
	mut events: EventReader<WindowCreated>,
	context: Res<GpuContextHandle>,
	windows: NonSend<WinitWindows>,
	query: Query<(Entity, &Window), Without<Swapchain>>,
	primary_window_query: Query<(), With<PrimaryWindow>>,
	settings: Res<RenderSettings>,
	state: Res<State<DisplayMode>>,
	session: Option<Res<XrSession>>,
) {
	for (e, window) in query.iter() {
		let swapchain = {
			// May not immediately return a both a valid display / window handle! Especially on Android where launching an app focused isn't supported!
			let (display_handle, window_handle) = {
				let window = windows.get_window(e).expect("Failed to get window for entity {e}!");

				match (window.display_handle(), window.window_handle()) {
					(Ok(display_handle), Ok(window_handle)) => (display_handle, window_handle),
					_ => continue,// Failed to create swapchain. One of the handles is not valid yet. Rerun next frame.
				}
			};

			let create_info = SwapchainCreateInfo {
				present_mode: window.present_mode,
				usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
				min_image_count: settings.frames_in_flight as u32,
				format: Some(vk::Format::R8G8B8A8_SRGB),
				color_space: Some(vk::ColorSpaceKHR::SRGB_NONLINEAR),
				..default()
			};

			Swapchain::new("swapchain".into(), Arc::clone(&context), display_handle, window_handle, &create_info)
		};

		if primary_window_query.contains(e) {
			let swapchain = match &session {
				Some(session) => Either::Right(&session.swapchain),
				None => Either::Left(&swapchain),
			};

			create_global_resources(&mut commands, &context, swapchain, &settings);
		}

		// Unfortunate archetype traversal. Should happen very rarely though.
		commands.entity(e).insert(swapchain);
	}
}

fn create_global_resources_once_for_xr(
	mut commands: Commands,
	session: Option<Res<XrSession>>,
	context: Res<GpuContextHandle>,
	settings: Res<RenderSettings>,
	mut created: Local<bool>,
) {
	let (Some(session), false) = (session, *created) else {
		return;
	};

	*created = true;

	println!("Creating global resources for XR!");

	create_global_resources(
		&mut commands,
		&context,
		Either::Right(&session.swapchain),
		&settings,
	);
}

fn create_global_resources(
	commands: &mut Commands,
	context: &Arc<GpuContext>,
	swapchain: Either<&Swapchain, &crate::xr_swapchain::Swapchain>,
	settings: &RenderSettings,
) {
	let (extent, layers) = match swapchain {
		Either::Left(swapchain) => (swapchain.extent(), 1),
		Either::Right(swapchain) => (swapchain.extent(), 2),
	};

	let base_pass = BasePass::new("base_pass".into(), Arc::clone(&context), extent, layers, settings.msaa);

	let pipeline = {
		let create_info = GraphicsPipelineCreateInfo {
			vertex_shader: &shader_path("shader.vert"),
			fragment_shader: &shader_path("shader.frag"),
			vertex_inputs: &[
				VertexInput {
					stride: size_of::<Vertex>() as u32,
					attributes: &[
						VertexAttribute {
							name: "pos",
							offset: offset_of!(Vertex, pos) as u32,
						},
						VertexAttribute {
							name: "uv",
							offset: offset_of!(Vertex, uv) as u32,
						},
					]
				},
			],
			msaa: settings.msaa,
			fill_mode: TriangleFillMode::Fill,
		};

		unsafe { GraphicsPipeline::new("base_material".into(), Arc::clone(&context), base_pass.render_pass, &create_info) }
	};

	let texture = {
		let path = Path::new("assets/models/chom/chom_festive.jpg");

		let image_data = {
			let buffer = io::read(path).unwrap();
			image::load_from_memory(&buffer).unwrap()
		};

		let create_info = ImageCreateInfo {
			resolution: UVec2::new(image_data.width(), image_data.height()),
			format: Format::RgbaU8Norm,
			usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
			msaa: MsaaCount::None,
			mip_count: MipCount::Max,
			..default()
		};

		let image = Image::new("texture".into(), Arc::clone(&context), &create_info);

		context.upload_image(&image, None, |mips| {
			// Create first mip.
			image_data
				.pixels()
				.zip(mips[0].chunks_exact_mut(4))
				.for_each(|((_, _, image::Rgba(rgba)), data)| data.copy_from_slice(&rgba));

			// Generate subsequent mips.
			for mip in 1..mips.len() {
				let ([.., src], [dst, ..]) = mips.split_at_mut(mip) else {
					unreachable!();
				};

				let prev_mip = mip - 1;

				let width = (image.resolution.x as usize >> mip).max(1);
				let prev_width = (image.resolution.x as usize >> prev_mip).max(1);
				dst.par_chunk_map_mut(ComputeTaskPool::get(), 4, |i, data| {
					let (x, y) = (i % width, i / width);
					let mut aggregate = [0; 4];
					let mut add = |offset_x, offset_y| {
						let i = ((x * 2 + offset_x) + (y * 2 + offset_y) * prev_width) * 4;
						aggregate
							.iter_mut()
							.zip(src[i..i + 4].iter())
							.for_each(|(dst, src)| *dst += *src as u32);
					};

					add(0, 0);
					add(1, 0);
					add(0, 1);
					add(1, 1);

					data
						.iter_mut()
						.zip(aggregate.iter())
						.for_each(|(dst, &src)| *dst = (src / 4) as u8);
				});
			}
		});

		image
	};

	let index_buffer = {
		let create_info = BufferCreateInfo {
			size: size_of_val(&CUBE_INDICES),
			usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			memory: MemoryLocation::GpuOnly,
		};

		let buffer = Buffer::new("index_buffer".into(), Arc::clone(&context), &create_info);

		context.upload_buffer(&buffer, |data| {
			unsafe { std::ptr::copy_nonoverlapping(CUBE_INDICES.as_ptr(), data.as_mut_ptr() as *mut _, CUBE_INDICES.len()); }
		});

		buffer
	};

	let vertex_buffer = {
		let create_info = BufferCreateInfo {
			size: size_of_val(&CUBE_VERTICES),
			usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			memory: MemoryLocation::GpuOnly,
		};

		let buffer = Buffer::new("vertex_buffer".into(), Arc::clone(&context), &create_info);

		context.upload_buffer(&buffer, |data| {
			unsafe { std::ptr::copy_nonoverlapping(CUBE_VERTICES.as_ptr(), data.as_mut_ptr() as *mut _, CUBE_VERTICES.len()); }
		});

		buffer
	};

	let view_matrices_buffer = {
		let view_matrices = unsafe {
			let model = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0));
			let view = Mat4::look_at_rh(Vec3::new(5.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
			let proj = Mat4::perspective_rh(45.0f32.to_radians(), extent.x as f32 / extent.y as f32, 0.1, 100.0);

			let mut proj = proj.to_cols_array_2d();
			proj[1][1] *= -1.0;

			let proj = Mat4::from_cols_array_2d(&proj);

			ViewMatrices {
				model: mem::transmute_copy(&model),
				view: mem::transmute_copy(&view),
				proj: mem::transmute_copy(&proj),
			}
		};

		let create_info = BufferCreateInfo {
			size: size_of::<ViewMatrices>() * settings.frames_in_flight,
			usage: vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			memory: MemoryLocation::CpuToGpu,
		};

		let buffer = Buffer::new("view_matrices_buffer".into(), Arc::clone(&context), &create_info);

		let mapped_slice = unsafe { slice::from_raw_parts_mut(buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut _, settings.frames_in_flight) };
		for value in mapped_slice.iter_mut() {
			*value = view_matrices;
		}

		buffer
	};

	let descriptor_pool = {
		let pool_sizes = [
			vk::DescriptorPoolSize::default()
				.ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.descriptor_count(/*settings.frames_in_flight as u32*/1000),
		];

		let create_info = vk::DescriptorPoolCreateInfo::default()
			.max_sets(settings.frames_in_flight as u32)
			.pool_sizes(&pool_sizes);

		unsafe { context.device.create_descriptor_pool(&create_info, None) }.unwrap()
	};

	let descriptor_sets = {
		assert_eq!(pipeline.descriptor_set_layouts.len(), 1);

		let set_layouts = (0..settings.frames_in_flight)
			.flat_map(|_| pipeline.descriptor_set_layouts.iter().copied())
			.collect::<Vec<_>>();

		println!("SET_LAYOUT_COUNT: {}", set_layouts.len());

		let allocate_info = vk::DescriptorSetAllocateInfo::default()
			.descriptor_pool(descriptor_pool)
			.set_layouts(&set_layouts);

		let descriptor_sets = unsafe { context.device.allocate_descriptor_sets(&allocate_info) }
			.unwrap_or_else(|e| panic!("Failed to allocate descriptor sets! {} frames in flight!: {e}", settings.frames_in_flight));

		let sampler = {
			let create_info = vk::SamplerCreateInfo::default()
				.mag_filter(vk::Filter::LINEAR) // Smooth magnification
				.min_filter(vk::Filter::LINEAR) // Smooth minification
				.mipmap_mode(vk::SamplerMipmapMode::LINEAR) // Trilinear filtering
				.address_mode_u(vk::SamplerAddressMode::REPEAT) // Wrap in U direction
				.address_mode_v(vk::SamplerAddressMode::REPEAT) // Wrap in V direction
				.address_mode_w(vk::SamplerAddressMode::REPEAT) // Wrap in W direction
				.mip_lod_bias(0.0) // No LOD bias
				.anisotropy_enable(false) // Enable anisotropic filtering
				.max_anisotropy(16.0) // Use max anisotropy (ensure device supports it)
				.compare_enable(false) // No depth comparison
				.min_lod(0.0) // Minimum mip level
				.max_lod(vk::LOD_CLAMP_NONE)
				.border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK) // Border color (only for CLAMP modes)
				.unnormalized_coordinates(false); // Use normalized texture coordinates

			unsafe { context.device.create_sampler(&create_info, None) }.unwrap()
		};

		let image_info = vk::DescriptorImageInfo::default()
			.sampler(sampler)
			.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
			.image_view(texture.view);

		/*let view_matrices_writes = [
			vk::DescriptorBufferInfo::default()
				.buffer(view_matrices_buffer.handle)
				.offset(0)
				.range(view_matrices_buffer.size() as vk::DeviceSize),
		];

		let descriptor_writes = [
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_sets)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
				.buffer_info(&view_matrices_writes),
			vk::WriteDescriptorSet::default()
				.dst_set(descriptor_sets)
				.dst_binding(1)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.image_info(slice::from_ref(&image_info))
		];*/

		let view_matrices_buffer_info = (0..settings.frames_in_flight)
			.map(|i| vk::DescriptorBufferInfo::default()
				.buffer(view_matrices_buffer.handle)
				.offset((i * size_of::<ViewMatrices>()) as _)
				.range(size_of::<ViewMatrices>() as _)
			)
			.collect::<Vec<_>>();

		let descriptor_writes = descriptor_sets
			.iter()
			.zip(view_matrices_buffer_info.iter())
			.flat_map(|(&descriptor_set, view_matrices_buffer_info)| [
				vk::WriteDescriptorSet::default()
					.dst_set(descriptor_set)
					.dst_binding(0)
					.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
					.buffer_info(slice::from_ref(&view_matrices_buffer_info)),
				vk::WriteDescriptorSet::default()
					.dst_set(descriptor_set)
					.dst_binding(1)
					.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
					.image_info(slice::from_ref(&image_info))
			]
			)
			.collect::<Vec<_>>();

		unsafe { context.device.update_descriptor_sets(&descriptor_writes, &[]); }

		descriptor_sets
	};

	#[cfg(debug_assertions)]
	unsafe {
		context.set_debug_name("base_material_descriptor_pool", descriptor_pool);
		for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
			context.set_debug_name(format!("{}_descriptor_set[{i}]", base_pass.name).as_str(), descriptor_set);
		}
	}

	let base_material = BaseMaterial {
		pipeline,
		descriptor_pool,
		descriptor_sets,
	};

	let indirect_args = {
		let args_buffer = Buffer::new("args_buffer".into(), Arc::clone(&context), &BufferCreateInfo {
			size: size_of::<vk::DrawIndexedIndirectCommand>() * 3,
			usage: vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			memory: MemoryLocation::GpuOnly,
		});

		context.upload_buffer(&args_buffer, |data| {
			unsafe {
				let cmds = slice::from_raw_parts_mut(data.as_mut_ptr() as *mut _, 3);
				for cmd in cmds {
					*cmd = vk::DrawIndexedIndirectCommand {
						index_count: CUBE_INDICES.len() as _,
						instance_count: 2,
						first_index: 0,
						vertex_offset: 0,
						first_instance: 0,
					};
				}
			}
		});

		let count_buffer = Buffer::new("count_buffer".into(), Arc::clone(&context), &BufferCreateInfo {
			size: size_of::<u32>(),
			usage: vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			memory: MemoryLocation::GpuOnly,
		});

		context.upload_buffer(&count_buffer, |data| {
			unsafe {
				*(data.as_mut_ptr() as *mut u32) = 3;
			}
		});

		IndirectArgs {
			args_buffer,
			count_buffer,
		}
	};

	commands.insert_resource(base_pass);
	commands.insert_resource(base_material);
	commands.insert_resource(Texture(texture));
	commands.insert_resource(Model {
		name: "cube".into(),
		index_buffer,
		vertex_buffer,
		view_matrices_buffer,
	});
	commands.insert_resource(indirect_args);
}
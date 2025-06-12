use std::mem::offset_of;
use std::{mem, slice};
use std::ffi::CStr;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use bevy::prelude::*;
use crate::{rhi_hush, rhi_label, GpuContext, GpuContextHandle, Image, ImageCreateInfo, MipCount, RhiHush};
use crate::render_settings::{DisplayMode, RenderSettings};
use crate::render::xr::*;
use ash::vk;
use ash::vk::Handle;
use azart_asset::{AssetCache, Asset, DefaultAssetHandler, AssetResource};
use azart_gfx_utils::{shader_path, Format, Msaa, TriangleFillMode};
use azart_utils::debug_string::DebugString;
use azart_utils::io;
use bevy::tasks::{block_on, ComputeTaskPool, ParallelSliceMut};
use bevy::window::{PresentMode, PrimaryWindow};
use bevy::winit::{WakeUp, WinitWindows};
use either::{for_both, Either};
use gpu_allocator::MemoryLocation;
use image::GenericImageView;
use openxr as xr;
use std140::repr_std140;
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasRawWindowHandle, HasWindowHandle};
use vk_sync::{AccessType, ImageBarrier, ImageLayout};
use crate::buffer::{Buffer, BufferCreateInfo};
use crate::glb::{Indices, Scene};
use crate::graphics_pipeline::{GraphicsPipeline, GraphicsPipelineCreateInfo, VertexAttribute, VertexInput};
use crate::pbr::mesh::{Material, Mesh, MeshAssetHandler};
use crate::render::base_pass::{base_pass, begin_base_pass, begin_base_pass_xr, end_base_pass, end_base_pass_xr, record_base_pass, update_global_ubo, update_global_ubo_xr, update_xr_camera_transform, XrFrameState};
use crate::render::camera::{Camera, XrCamera};
use crate::render::gpu_scene::{GlobalUbo, GpuScene};
use crate::render::swapchain::{Swapchain, SwapchainDesc};
use crate::render_pass::RenderPass;

const Z_NEAR: f32 = 0.025;
const Z_FAR: f32 = 1000.0;

pub struct RenderPlugin {
	pub settings: RenderSettings,
	pub display_mode: DisplayMode,
}

impl Default for RenderPlugin {
	fn default() -> Self {
		Self {
			settings: default(),
			#[cfg(not(target_os = "android"))]
			display_mode: std::env::args()
				.find_map(|str| matches!(str.as_str(), "vr" | "xr").then(|| DisplayMode::Xr))
				.unwrap_or(DisplayMode::Std),
			/// Force DisplayMode::Xr on Android.
			/// @TODO: Detect whether VR is enabled or not and automatically decide.
			#[cfg(target_os = "android")]
			display_mode: DisplayMode::Xr,
		}
	}
}

impl Plugin for RenderPlugin {
	fn build(&self, app: &mut App) {
	  use azart_asset::RegisterAssetPlugin;
		use crate::ImageAssetHandler;

		let mut settings = self.settings.clone();

		let event_loop = app.world().non_send_resource::<EventLoop<WakeUp>>();
		let swapchain_exts = ash_window::enumerate_required_extensions(event_loop.display_handle().unwrap().as_raw())
			.expect("Failed to get display handle")
			.into_iter()
			.map(|&ext| unsafe { CStr::from_ptr(ext) })
			.collect::<Vec<_>>();

		let cx = match self.display_mode {
			DisplayMode::Std => Arc::new(GpuContext::new(&swapchain_exts, None)),
			DisplayMode::Xr => {
				let instance = XrInstance::new(&swapchain_exts);
				let context = Arc::clone(&instance.context());
				let (session, frame_waiter, frame_stream, space) = instance.create_session(xr::ReferenceSpaceType::STAGE).expect("Failed to create session!");

				app
					.insert_resource(instance)
					.insert_resource(session)
					.insert_resource(frame_waiter)
					.insert_resource(frame_stream)
					.insert_resource(space);

				context
			},
		};

		app
			.insert_resource(GpuContextHandle(Arc::clone(&cx)))
			.insert_resource(settings)
			// Register Image asset handling plugin.
			.add_plugins((
				RegisterAssetPlugin::with_handler(MeshAssetHandler::new(Arc::clone(&cx))),
				RegisterAssetPlugin::with_handler(ImageAssetHandler::new(Arc::clone(&cx))),
				RegisterAssetPlugin::with_handler(DefaultAssetHandler::<Material>::default()),
			))
			.insert_state(self.display_mode)
			.insert_state(XrState::Idle)
			.add_systems(
				Startup,
				(
					create_base_pass,
					create_global_resources,
					create_xr_swapchain.run_if(in_state(DisplayMode::Xr))
				)
					.chain()
			)
			.add_systems(
				Update,
				(
					create_swapchain_on_window_spawned,
					//render,
					begin_base_pass,
					update_global_ubo,
					base_pass,
					end_base_pass,
				)
					.chain()
			)
			.add_systems(
				Update,
				(
					poll_xr_events,
					update_xr_camera_transform,
					begin_base_pass_xr,
					update_global_ubo_xr,
					base_pass,
					end_base_pass_xr,
				)
					.chain()
					.run_if(in_state(DisplayMode::Xr))
			);

		match renderdoc::RenderDoc::new() {
			Ok(renderdoc) => _ = app.insert_non_send_resource(RenderDoc(renderdoc)),
			Err(e) => info!("Failed to initialize renderdoc: {e}"),
		}
	}
}

const CUBE_INDICES: [u16; 36] = [
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

const CUBE_VERTICES: [Vertex; 24] = [
	Vertex { pos: [-0.5,  0.5,  0.5], uv: [0.0, 0.0] },
	Vertex { pos: [ 0.5,  0.5,  0.5], uv: [1.0, 0.0] },
	Vertex { pos: [ 0.5, -0.5,  0.5], uv: [1.0, 1.0] },
	Vertex { pos: [-0.5, -0.5,  0.5], uv: [0.0, 1.0] },

	Vertex { pos: [ 0.5,  0.5,  0.5], uv: [0.0, 0.0] },
	Vertex { pos: [ 0.5,  0.5, -0.5], uv: [1.0, 0.0] },
	Vertex { pos: [ 0.5, -0.5, -0.5], uv: [1.0, 1.0] },
	Vertex { pos: [ 0.5, -0.5,  0.5], uv: [0.0, 1.0] },

	Vertex { pos: [ 0.5,  0.5, -0.5], uv: [0.0, 0.0] },
	Vertex { pos: [-0.5,  0.5, -0.5], uv: [1.0, 0.0] },
	Vertex { pos: [-0.5, -0.5, -0.5], uv: [1.0, 1.0] },
	Vertex { pos: [ 0.5, -0.5, -0.5], uv: [0.0, 1.0] },

	Vertex { pos: [-0.5,  0.5, -0.5], uv: [0.0, 0.0] },
	Vertex { pos: [-0.5,  0.5,  0.5], uv: [1.0, 0.0] },
	Vertex { pos: [-0.5, -0.5,  0.5], uv: [1.0, 1.0] },
	Vertex { pos: [-0.5, -0.5, -0.5], uv: [0.0, 1.0] },

	Vertex { pos: [-0.5,  0.5,  0.5], uv: [0.0, 0.0] },
	Vertex { pos: [ 0.5,  0.5,  0.5], uv: [1.0, 0.0] },
	Vertex { pos: [ 0.5,  0.5, -0.5], uv: [1.0, 1.0] },
	Vertex { pos: [-0.5,  0.5, -0.5], uv: [0.0, 1.0] },

	Vertex { pos: [-0.5, -0.5, -0.5], uv: [0.0, 0.0] },
	Vertex { pos: [ 0.5, -0.5, -0.5], uv: [1.0, 0.0] },
	Vertex { pos: [ 0.5, -0.5,  0.5], uv: [1.0, 1.0] },
	Vertex { pos: [-0.5, -0.5,  0.5], uv: [0.0, 1.0] },
];


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
	pub pos: [f32; 3],
	pub uv: [f32; 2],
}

/*#[repr_std140]
#[derive(Copy, Clone, Debug)]
pub struct ViewMatrices {
	pub model: std140::mat4x4,
	pub view: std140::mat4x4,
	pub proj: std140::mat4x4,
}*/

// Scalar layout types.
pub mod scalar {
	pub type Mat4 = [[f32; 4]; 4];
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ViewMatrices {
	pub model: Mat4,
	pub views: [Mat4; 2],
	pub projs: [Mat4; 2],
}

#[derive(Resource)]
pub struct CubeMesh {
	pub index_buffer: Buffer,
	pub vertex_buffer: Buffer,
}

#[derive(Resource, Deref, DerefMut)]
pub struct PbrPipeline(pub GraphicsPipeline);

#[derive(Resource, Deref, DerefMut)]
pub struct ViewMatricesBuffer(pub Buffer);

#[derive(Resource, Deref, DerefMut)]
pub struct Texture(pub Image);

#[derive(Resource, Deref, DerefMut)]
pub struct BasePass(pub Arc<RenderPass>);

#[derive(Resource)]
pub struct Sampler {
	name: DebugString,
	context: Arc<GpuContext>,
	pub handle: vk::Sampler,
}

impl Sampler {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		create_info: &vk::SamplerCreateInfo,
	) -> Self {
		let sampler = unsafe {
			let sampler = context.device.create_sampler(create_info, None).expect("Failed to create sampler!");

			#[cfg(debug_assertions)]
			context.set_debug_name(name.as_str(), sampler);

			sampler
		};

		Self {
			name,
			context,
			handle: sampler,
		}
	}
}

impl Drop for Sampler {
	fn drop(&mut self) {
		unsafe {
			self.context.device.destroy_sampler(self.handle, None);
		}
	}
}

#[derive(Deref, DerefMut)]
pub struct RenderDoc(pub renderdoc::RenderDoc<renderdoc::V110>);

#[must_use]
#[inline(always)]
pub const fn align_to(value: usize, align: usize) -> usize {
	(value + align - 1) & !(align - 1)
}

/// Computes the minimum stride for accessing a buffered ubo.
#[must_use]
#[inline(always)]
pub const fn ubo_stride<T: Sized>(ubo_min_align: usize) -> usize {
	align_to(size_of::<T>(), ubo_min_align)
}

/// Computes the total size required to create a frame-buffered ubo while respecting ubo min alignment.
#[must_use]
#[inline(always)]
pub const fn ubo_size<T: Sized>(frames_in_flight: usize, ubo_min_align: usize) -> usize {
	ubo_stride::<T>(ubo_min_align) * (frames_in_flight - 1) + size_of::<T>()
}

fn create_base_pass(
	mut commands: Commands,
	context: Res<GpuContextHandle>,
	display_mode: Res<State<DisplayMode>>,
	settings: Res<RenderSettings>,
) {
	let base_pass = {
		const COLOR_FORMAT: Format = Format::RgbaU8Srgb;
		const DEPTH_FORMAT: Format = Format::DF32;

		let (multiview_count, multiview_mask) = match display_mode.get() {
			DisplayMode::Std => (None, 0b1),
			DisplayMode::Xr => (Some(NonZeroUsize::new(2).unwrap()), 0b11),
			_ => unreachable!(),
		};

		const PRESENT_ATTACHMENT: u32 = 0;// Always 0. If MSAA is enabled. Write to index 2 and resolve to 0.
		const DEPTH_ATTACHMENT: u32 = 1;// Always 1.

		let color_attachment_ref = vk::AttachmentReference2::default()
			.attachment(match settings.msaa {
				Msaa::None => PRESENT_ATTACHMENT,
				_ => 2,
			})
			.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
			.aspect_mask(vk::ImageAspectFlags::COLOR);

		let depth_attachment_ref = vk::AttachmentReference2::default()
			.attachment(1)
			.layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
			.aspect_mask(vk::ImageAspectFlags::DEPTH);

		let color_resolve_ref = (settings.msaa != Msaa::None).then(|| vk::AttachmentReference2::default()
			.attachment(PRESENT_ATTACHMENT)// Resolve for color (always 0).
			.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
			.aspect_mask(vk::ImageAspectFlags::COLOR)
		);

		let subpasses = match &color_resolve_ref {
			Some(color_resolve_ref) => [
				vk::SubpassDescription2::default()
					.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
					.color_attachments(slice::from_ref(&color_attachment_ref))
					.depth_stencil_attachment(&depth_attachment_ref)
					.resolve_attachments(slice::from_ref(color_resolve_ref))
					.view_mask(multiview_mask),
			],
			None => [
				vk::SubpassDescription2::default()
					.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
					.color_attachments(slice::from_ref(&color_attachment_ref))
					.depth_stencil_attachment(&depth_attachment_ref)
					.view_mask(multiview_mask),
			],
		};

		let dependencies = [
			vk::SubpassDependency2::default()
				.src_subpass(vk::SUBPASS_EXTERNAL)
				.dst_subpass(0)
				.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
				.src_access_mask(vk::AccessFlags::empty())
				.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
				.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
		];

		let attachments = [
			vk::AttachmentDescription2::default()
				.format(COLOR_FORMAT.into())
				.initial_layout(vk::ImageLayout::UNDEFINED)
				.final_layout(match settings.msaa {
					Msaa::None => vk::ImageLayout::PRESENT_SRC_KHR,// Will be presented.
					_ => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,// Will be resolved.
				})
				.load_op(vk::AttachmentLoadOp::CLEAR)
				.samples(settings.msaa.into()),
			vk::AttachmentDescription2::default()
				.format(DEPTH_FORMAT.into())
				.initial_layout(vk::ImageLayout::UNDEFINED)
				.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
				.load_op(vk::AttachmentLoadOp::CLEAR)
				.samples(settings.msaa.into()),
		];

		let attachments = match settings.msaa {
			Msaa::None => Either::Left([
				vk::AttachmentDescription2::default()
					.format(COLOR_FORMAT.into())
					.initial_layout(vk::ImageLayout::UNDEFINED)
					.final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.store_op(vk::AttachmentStoreOp::STORE)
					.samples(vk::SampleCountFlags::TYPE_1),
				vk::AttachmentDescription2::default()
					.format(DEPTH_FORMAT.into())
					.initial_layout(vk::ImageLayout::UNDEFINED)
					.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.store_op(vk::AttachmentStoreOp::DONT_CARE)
					.samples(vk::SampleCountFlags::TYPE_1),
			]),
			msaa => Either::Right([
				vk::AttachmentDescription2::default()
					.format(COLOR_FORMAT.into())
					.initial_layout(vk::ImageLayout::UNDEFINED)
					.final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
					.load_op(vk::AttachmentLoadOp::DONT_CARE)
					.store_op(vk::AttachmentStoreOp::STORE)
					.samples(vk::SampleCountFlags::TYPE_1),
				vk::AttachmentDescription2::default()
					.format(DEPTH_FORMAT.into())
					.initial_layout(vk::ImageLayout::UNDEFINED)
					.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.store_op(vk::AttachmentStoreOp::DONT_CARE)
					.samples(msaa.into()),
				vk::AttachmentDescription2::default()
					.format(COLOR_FORMAT.into())
					.initial_layout(vk::ImageLayout::UNDEFINED)
					.final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
					.load_op(vk::AttachmentLoadOp::CLEAR)
					.store_op(vk::AttachmentStoreOp::DONT_CARE)
					.samples(msaa.into()),
			]),
		};

		let create_info = vk::RenderPassCreateInfo2::default()
			.attachments(for_both!(&attachments, a => a.as_slice()))
			.subpasses(&subpasses)
			.dependencies(&dependencies)
			.correlated_view_masks(slice::from_ref(&multiview_mask));

		let render_pass = unsafe { context.exts.create_render_pass2.create_render_pass2(&create_info, None) }.expect("Failed to create render pass!");

		BasePass(Arc::new(RenderPass {
			name: "base_pass".into(),
			context: Arc::clone(&context),
			handle: render_pass,
			msaa: settings.msaa,
			multiview_count,
		}))
	};

	commands.insert_resource(base_pass);
}

fn create_xr_swapchain(
	mut commands: Commands,
	context: Res<GpuContextHandle>,
	render_pass: Res<BasePass>,
	session: Res<XrSession>,
	display_mode: Res<State<DisplayMode>>,
	mut settings: ResMut<RenderSettings>,
) {
	commands.insert_resource(XrSwapchain::new(
		"xr_swapchain".into(),
		Arc::clone(&context),
		Arc::clone(&render_pass),
		&session,
		&XrSwapchainDesc {
			frames_in_flight: settings.frames_in_flight as _,
			msaa: settings.msaa,
			fmt: Some(Format::RgbaU8Srgb),
			usage: xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::TRANSFER_SRC | xr::SwapchainUsageFlags::TRANSFER_DST,
		},
	));

	commands.insert_resource(XrFrameState(None));
}

fn create_swapchain_on_window_spawned(
	window_query: Query<(Entity, &Window), Without<Swapchain>>,
	windows: NonSend<WinitWindows>,
	mut commands: Commands,
	context: Res<GpuContextHandle>,
	base_pass: Res<BasePass>,
	display_mode: Res<State<DisplayMode>>,
	settings: Res<RenderSettings>,
) {
	// TODO: Make this work with xr runtime (render pass permutation handling)
	if *display_mode.get() == DisplayMode::Xr {
		return;
	}

	for (e, window) in window_query.iter() {
		let winit_window = windows.get_window(e).unwrap();

		// Either handle may not immediately be ready.
		let (display_handle, window_handle) = match (winit_window.display_handle(), winit_window.window_handle()) {
			(Ok(display_handle), Ok(window_handle)) => (display_handle, window_handle),
			_ => continue,
		};

		commands.entity(e).insert(Swapchain::new(
			"swapchain".into(),
			Arc::clone(&context),
			Arc::clone(&base_pass),
			display_handle,
			window_handle,
			&SwapchainDesc {
				present_mode: window.present_mode,
				format: Some(Format::RgbaU8Srgb),
				color_space: Some(vk::ColorSpaceKHR::SRGB_NONLINEAR),
				msaa: settings.msaa,
				usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
				frames_in_flight: settings.frames_in_flight,
				..default()
			},
		));
	}
}

fn create_global_resources(
	mut commands: Commands,
	base_pass: Res<BasePass>,
	cx: Res<GpuContextHandle>,
	display_mode: Res<State<DisplayMode>>,
	settings: Res<RenderSettings>,
	cache: Res<AssetCache>,
) {
	let asset: Asset<Mesh> = block_on(cache.load("assets/corset/pCube49.mesh")).expect("Failed to load pCube49!");
	println!("ASSET: {asset:?}");
	println!("base_color: {:?}", &*asset.material.base_color);

	match display_mode.get() {
		DisplayMode::Xr => _ = commands.spawn((
			XrCamera::default(),
			Transform::default(),
		)),
		DisplayMode::Std => _ = commands.spawn((
			Camera { fov: 70.0 },
			Transform::from_matrix(Mat4::look_at_rh(Vec3::new(0.0, -0.5, -1.5), Vec3::new(0.0, -1.5, 0.0), Vec3::Y)),
		)),
	}

	commands.insert_resource(GpuScene {
		global_ubo: Buffer::new(
			"global_ubo".into(),
			Arc::clone(&cx),
			&BufferCreateInfo {
				size: align_to(size_of::<GlobalUbo>(), cx.limits.ubo_min_align) * settings.frames_in_flight,
				usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
				memory: MemoryLocation::CpuToGpu,// Updated every frame so just mapped to the CPU.
			},
		),
	});

	let pbr_pipeline = PbrPipeline(GraphicsPipeline::new(
		"pbr_pipeline".into(),
		Arc::clone(&cx),
		base_pass.handle,
		&GraphicsPipelineCreateInfo {
			vertex_shader: &shader_path("shader.vert"),
			fragment_shader: &shader_path("shader.frag"),
			vertex_inputs: &[
				VertexInput {
					stride: size_of::<Vertex>() as _,
					attributes: &[
						VertexAttribute {
							name: "pos",
							offset: offset_of!(Vertex, pos) as _,
						},
						VertexAttribute {
							name: "uv",
							offset: offset_of!(Vertex, uv) as _,
						},
					],
				},
			],
			msaa: settings.msaa,
			fill_mode: TriangleFillMode::Fill,
		},
	));

	const ASSET_PATH: &str = "assets/models/Corset/glTF/Corset.gltf";

	let path = io::data_path().join(ASSET_PATH);

	for entry in io::read_dir("assets/models/Corset/glTF").expect("Failed to read dir!").iter() {
		println!("Entry: {entry:?}");
		let data = io::read(entry).unwrap_or_else(|e| panic!("Failed to read {entry:?}!: {e}"));
		io::write(entry, &data).unwrap_or_else(|e| panic!("Failed to write {entry:?}!: {e}"));
	}

	let mut scene = Scene::load(&path).expect("Failed to load gltf!");

	let mesh = {
		let mesh = &scene.meshes[0];

		let index_buffer = Buffer::new(
			"cube_index_buffer".into(),
			Arc::clone(&cx),
			&BufferCreateInfo {
				size: match &mesh.indices {
					Indices::U8(indices) => indices.len() * size_of::<u8>(),
					Indices::U16(indices) => indices.len() * size_of::<u16>(),
					Indices::U32(indices) => indices.len() * size_of::<u32>(),
				},
				usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				memory: MemoryLocation::GpuOnly,
			},
		);

		let vertex_buffer = Buffer::new(
			"cube_vertex_buffer".into(),
			Arc::clone(&cx),
			&BufferCreateInfo {
				size: mesh.positions.len() * size_of::<Vertex>(),
				usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
				memory: MemoryLocation::GpuOnly,
			},
		);

		cx.upload_buffer(&index_buffer, |dst| unsafe {
			let dst = dst.as_mut_ptr();
			let (src, size) = match &mesh.indices {
				Indices::U8(indices) => (indices.as_ptr(), indices.len()),
				Indices::U16(indices) => (indices.as_ptr() as *const u8, indices.len() * size_of::<u16>()),
				Indices::U32(indices) => (indices.as_ptr() as *const u8, indices.len() * size_of::<u32>()),
			};

			std::ptr::copy_nonoverlapping(src, dst, size);
		});

		cx.upload_buffer(&vertex_buffer, |dst| unsafe {
			/*let dst = dst.as_mut_ptr();
			let src = mesh.positions.as_ptr() as *const u8;
			let size = mesh.positions.len() * size_of::<[f32; 3]>();

			std::ptr::copy_nonoverlapping(src, dst, size);*/

			let positions = &mesh.positions;
			let dst = slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Vertex, mesh.positions.len());

			let transform_pos = |pos: [f32; 3]| {
			  let mut pos = pos.map(|x| x * 20.0);
				pos[1] -= 2.5;
				pos
			};

			if let Some(uvs) = &mesh.uvs {
				assert_eq!(positions.len(), uvs.len());

				dst
					.iter_mut()
					.zip(positions.iter().zip(uvs.iter()))
					.for_each(|(dst, (&pos, &uv))| *dst = Vertex { pos: transform_pos(pos), uv });
			} else {
				dst
					.iter_mut()
					.zip(positions.iter())
					.for_each(|(dst, &pos)| *dst = Vertex { pos: transform_pos(pos), uv: [0.0, 0.0] });
			}
		});

		CubeMesh {
			index_buffer,
			vertex_buffer,
		}

		/*let index_buffer = Buffer::new("cube_index_buffer".into(), Arc::clone(&context), &BufferCreateInfo {
			size: size_of_val(&CUBE_INDICES),
			usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			memory: MemoryLocation::GpuOnly,
		});

		let vertex_buffer = Buffer::new("cube_vertex_buffer".into(), Arc::clone(&context), &BufferCreateInfo {
			size: size_of_val(&CUBE_VERTICES),
			usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
			memory: MemoryLocation::GpuOnly,
		});

		context.upload_buffer(&index_buffer, |data| unsafe {
			std::ptr::copy_nonoverlapping(CUBE_INDICES.as_ptr() as *const _, data.as_mut_ptr(), size_of_val(&CUBE_INDICES));
		});

		context.upload_buffer(&vertex_buffer, |data| unsafe {
			std::ptr::copy_nonoverlapping(CUBE_VERTICES.as_ptr() as *const _, data.as_mut_ptr(), size_of_val(&CUBE_VERTICES));
		});

		CubeMesh {
			index_buffer,
			vertex_buffer,
		}*/
	};

	let view_matrices_buffer = ViewMatricesBuffer(Buffer::new("view_matrices".into(), Arc::clone(&cx), &BufferCreateInfo {
		size: size_of::<ViewMatrices>() * settings.frames_in_flight + (align_to(size_of::<ViewMatrices>(), cx.limits.ubo_min_align) * (settings.frames_in_flight - 1)),
		usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
		memory: MemoryLocation::CpuToGpu,
	}));

	let texture = {
		/*let path = Path::new("assets/models/chom/chom_festive.jpg");

		let image_data = {
			let buffer = io::read(path).unwrap();
			image::load_from_memory(&buffer).unwrap()
		};*/

		let image_data = scene.textures.drain(..).next().unwrap();

		let create_info = ImageCreateInfo {
			resolution: UVec2::new(image_data.width(), image_data.height()),
			format: Format::RgbaU8Norm,
			usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
			msaa: Msaa::None,
			mip_count: MipCount::Max,
			..default()
		};

		let image = Image::new("texture".into(), Arc::clone(&cx), &create_info);

		cx.upload_image(&image, None, |mips| {
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

		Texture(image)
	};

	let sampler = Sampler::new("sampler".into(), Arc::clone(&cx), &vk::SamplerCreateInfo::default()
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
		.unnormalized_coordinates(false)
	);

	commands.insert_resource(pbr_pipeline);
	commands.insert_resource(mesh);
	commands.insert_resource(view_matrices_buffer);
	commands.insert_resource(texture);
	commands.insert_resource(sampler);
}

fn poll_xr_events(
	mut commands: Commands,
	session: Res<XrSession>,
	mut xr_state: ResMut<NextState<XrState>>,
	mut previous_state: Local<Option<xr::SessionState>>,
) {
	let previous_state = match &mut *previous_state {
		Some(previous_state) => previous_state,
		None => {
			*previous_state = Some(xr::SessionState::IDLE);
			&mut previous_state.unwrap()
		},
	};

	let mut buffer = default();
	while let Some(event) = session.instance().poll_event(&mut buffer).unwrap() {
		match event {
			xr::Event::SessionStateChanged(state) => {
				match state.state() {
					xr::SessionState::READY => {
						info!("Beginning OpenXR session!");
						session.begin(xr::ViewConfigurationType::PRIMARY_STEREO).expect("Failed to begin OpenXR session!");
						xr_state.set(XrState::Focused);
					},
					xr::SessionState::STOPPING => {
						info!("Ending OpenXR session!");
						session.end().expect("Failed to end OpenXR session!");
						xr_state.set(XrState::Idle);
					},
					/*xr::SessionState::IDLE => {
						info!("IDLING...");
						xr_state.set(XrState::Idle);
					},
					xr::SessionState::FOCUSED => {
						info!("FOCUSED!");
						xr_state.set(XrState::Focused);
					},*/
					_ => {}
				}

				println!("STATE CHANGED FROM {:?} TO {:?}", *previous_state, state.state());
				*previous_state = state.state();
			},
			_ => info!("UNKNOWN EVENT!"),
		}
	}
}

fn begin_record_base_pass(
	mut commands: Commands,
	cx: Res<GpuContextHandle>,
	display_mode: DisplayMode,
	settings: Res<RenderSettings>,
) {

}


fn render(
	context: Res<GpuContextHandle>,
	mut swapchains: Query<&mut Swapchain>,
	base_pass: Res<BasePass>,
	pipeline: Res<PbrPipeline>,
	mesh: Res<CubeMesh>,
	view_matrices: Res<ViewMatricesBuffer>,
	texture: Res<Texture>,
	sampler: Res<Sampler>,
	mut renderdoc: Option<NonSendMut<RenderDoc>>,
	settings: Res<RenderSettings>,
	time: Res<Time>,
) {
	for mut swapchain in swapchains.iter_mut() {
		///
		/// WAIT FOR FRAME TO FINISH
		///
		let frame_index = swapchain.current_frame_index;
		let frame = &swapchain.frames[frame_index];
		let (image_index, suboptimal) = unsafe {
			context.device.wait_for_fences(slice::from_ref(&frame.in_flight_fence), true, u64::MAX).expect("Failed to wait for frame fences!");

			let result = context.exts.swapchain.acquire_next_image(swapchain.handle, u64::MAX, frame.image_available_semaphore, vk::Fence::null());
			assert_ne!(result, Err(vk::Result::SUBOPTIMAL_KHR));

			let (image_index, suboptimal) = match result {
				Ok(result) => result,
				Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => todo!("Recreate swapchain."),
				Err(e) => panic!("Failed to acquire swapchain image!: {e}"),
			};

			context.device.reset_fences(slice::from_ref(&frame.in_flight_fence)).unwrap();

			swapchain.current_frame_index = frame_index;
			swapchain.current_frame_buffer_index = image_index as _;

			(image_index as usize, suboptimal)
		};

		// Appease the borrow-checker.
		let frame_index = swapchain.current_frame_index;
		let frame = &swapchain.frames[frame_index];

		let swapchain_frame_buffer = &swapchain.swapchain_frame_buffers[image_index];

		// Update view matrices.
		unsafe {
			let data = view_matrices.allocation.mapped_ptr().unwrap().as_ptr().offset((align_to(size_of::<ViewMatrices>(), context.limits.ubo_min_align) * frame_index) as _) as *mut ViewMatrices;

			let model = Mat4::from_rotation_y(time.elapsed_secs() * std::f32::consts::PI * 0.5);
			let view = Mat4::look_at_rh(Vec3::new(5.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
			let mut proj = Mat4::perspective_rh(45_f32.to_radians(), swapchain.resolution.x as f32 / swapchain.resolution.y as f32, Z_NEAR, Z_FAR);
			proj.y_axis.y *= -1.0;

			data.write(ViewMatrices {
				model: mem::transmute_copy(&model),
				views: [mem::transmute_copy(&view), default()],
				projs: [mem::transmute_copy(&proj), default()],
			});
		}

		let cmd = unsafe {
			/*let cmd = {
				context.device.reset_command_pool(frame.graphics_cmd_pool, vk::CommandPoolResetFlags::empty()).unwrap();
				context.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default()
					.command_pool(frame.graphics_cmd_pool)
					.command_buffer_count(1)
					.level(vk::CommandBufferLevel::PRIMARY)
				).expect("Failed to allocate command buffer!")[0]
			};*/

			context.device.reset_command_pool(frame.graphics_cmd_pool, vk::CommandPoolResetFlags::empty()).unwrap();
			let cmd = frame.graphics_cmd;

			context.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

			cmd
		};

		///
		/// BEGIN RECORD
		///

		record_base_pass(
			&context,
			cmd,
			swapchain_frame_buffer.image,
			base_pass.handle,
			swapchain_frame_buffer.frame_buffer,
			swapchain.resolution,
			&pipeline,
			&mesh,
			&view_matrices,
			&texture,
			&sampler,
			frame_index,
			settings.msaa,
		);

		let queue = unsafe { context.device.get_device_queue(context.queue_families.graphics, 0) };

		// Submit queue.
		unsafe {
			context.device.end_command_buffer(cmd).unwrap();

			let submit_info = vk::SubmitInfo::default()
				.command_buffers(slice::from_ref(&cmd))
				.wait_semaphores(slice::from_ref(&frame.image_available_semaphore))// @NOTE: Might need to be on image buffer instead?
				.wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
				.signal_semaphores(slice::from_ref(&frame.render_finished_semaphore));

			context.device.queue_submit(queue, slice::from_ref(&submit_info), frame.in_flight_fence).expect("Failed to submit command buffer!");
		}

		// Queue present.
		unsafe {
			let image_indices = [image_index as _];
			let present_info = vk::PresentInfoKHR::default()
				.swapchains(slice::from_ref(&swapchain.handle))
				.image_indices(&image_indices)
				.wait_semaphores(slice::from_ref(&frame.render_finished_semaphore));

			let result = context.exts.swapchain.queue_present(queue, &present_info);
			assert_ne!(result, Err(vk::Result::SUBOPTIMAL_KHR));

			if suboptimal || matches!(result, Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR)) {
				todo!("Handle");
			}
		}

		swapchain.current_frame_index = (frame_index + 1) % swapchain.frames.len();
		swapchain.current_frame_buffer_index = image_index;
	}
}

fn render_xr(
	context: Res<GpuContextHandle>,
	session: Res<XrSession>,
	mut frame_waiter: ResMut<XrFrameWaiter>,
	mut frame_stream: ResMut<XrFrameStream>,
	mut swapchain: ResMut<XrSwapchain>,
	space: Res<XrSpace>,
	pipeline: Res<PbrPipeline>,
	mesh: Res<CubeMesh>,
	view_matrices: Res<ViewMatricesBuffer>,
	texture: Res<Texture>,
	sampler: Res<Sampler>,
	mut renderdoc: Option<NonSendMut<RenderDoc>>,
	settings: Res<RenderSettings>,
	time: Res<Time>,
) {
	#[cfg(target_os = "android")]
	{
		let ctx = ndk_context::android_context();
		let vm = unsafe { jni::JavaVM::from_raw(ctx.vm().cast()) }.unwrap();
		let env = vm.attach_current_thread().unwrap();
	}

	if let Some(renderdoc) = renderdoc.as_mut() {
		if renderdoc.is_frame_capturing() {
			renderdoc.start_frame_capture(context.device.handle().as_raw() as *const _, std::ptr::null());
		}
	}

	///
	/// BEGIN XR
	///

	let frame_state = match frame_waiter.wait() {
		Ok(frame_state) => frame_state,
		Ok(xr::FrameState { should_render: false, .. }) | Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => {
			warn!("Session not running!");
			return;
		},
		Err(e) => panic!("Failed to wait frame: {e}"),
	};

	frame_stream.begin().expect("Failed to begin frame stream!");
	let image_index = swapchain.handle.acquire_image().expect("Failed to acquire next image in swapchain!") as usize;
	let frame_index = (swapchain.current_frame_index + 1) % swapchain.frames.len();
	swapchain.handle.wait_image(xr::Duration::INFINITE).expect("Failed to wait image!");

	swapchain.current_frame_buffer_index = image_index;
	swapchain.current_frame_index = frame_index;

	let (view_flags, views) = session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, frame_state.predicted_display_time, &space).unwrap();
	assert_eq!(views.len(), 2, "There must be 2 views!");
	let views = [views[0], views[1]];

	///
	/// BEGIN VK
	///

	let frame = &swapchain.frames[frame_index];
	let swapchain_frame_buffer = &swapchain.swapchain_frame_buffers[image_index];

	unsafe {
		context.device.wait_for_fences(slice::from_ref(&frame.in_flight_fence), true, u64::MAX).expect("Failed to wait for frame fences!");
		context.device.reset_fences(slice::from_ref(&frame.in_flight_fence)).unwrap();
	}

	// Update view matrices.
	unsafe {
		let data = view_matrices.allocation.mapped_ptr().unwrap().as_ptr().offset((align_to(size_of::<ViewMatrices>(), context.limits.ubo_min_align) * frame_index) as _) as *mut ViewMatrices;

		let model_matrix = Mat4::from_rotation_y(time.elapsed_secs() * std::f32::consts::PI * 0.5);
		let view_matrices = views.map(|xr::View { pose: xr::Posef { position, orientation }, .. }| {
			let translation = Vec3::new(position.x, position.y, position.z);
			let rotation = Quat::from_xyzw(orientation.x, orientation.y, orientation.z, orientation.w);

			Mat4::from_rotation_translation(rotation, translation)
		});

		let proj_matrices = views.map(|xr::View { pose: xr::Posef { position, orientation }, fov: xr::Fovf { angle_right: r, angle_left: l, angle_up: u, angle_down: d } }| {
			let translation = Vec3::new(position.x, position.y, position.z);
			let rotation = Quat::from_xyzw(orientation.x, orientation.y, orientation.z, orientation.w);

			let view = Mat4::from_rotation_translation(rotation, translation).inverse();// world -> eye

			let (r, l, u, d) = (r.tan(), l.tan(), u.tan(), d.tan());
			let (w, h) = (r - l, d - u);

			let proj = Mat4::from_cols_array_2d(&[
				[2.0 / w, 0.0, 0.0, 0.0],
				[0.0, 2.0 / h, 0.0, 0.0],
				[(r + l) / w, (u + d) / h, -(Z_FAR + Z_NEAR) / (Z_FAR - Z_NEAR), -1.0],
				[0.0, 0.0, -(Z_FAR * (Z_NEAR + Z_NEAR)) / (Z_FAR - Z_NEAR), 0.0],
			]);

			proj * view
		});

		data.write(ViewMatrices {
			model: mem::transmute_copy(&model_matrix),
			views: [(); 2].map(|_| Mat4::IDENTITY),//mem::transmute_copy(&view_matrices),
			projs: mem::transmute_copy(&proj_matrices),
		});
	}

	let cmd = unsafe {
		let cmd = {
			context.device.reset_command_pool(frame.graphics_cmd_pool, vk::CommandPoolResetFlags::empty()).unwrap();
			context.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default()
				.command_pool(frame.graphics_cmd_pool)
				.command_buffer_count(1)
				.level(vk::CommandBufferLevel::PRIMARY)
			).expect("Failed to allocate command buffer!")[0]
		};

		context.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

		cmd
	};

	///
	/// BEGIN RECORD
	///

	record_base_pass(
		&context,
		cmd,
		swapchain_frame_buffer.image,
		swapchain.render_pass.handle,
		swapchain_frame_buffer.frame_buffer,
		swapchain.resolution,
		&pipeline,
		&mesh,
		&view_matrices,
		&texture,
		&sampler,
		frame_index,
		settings.msaa,
	);

	///
	///	END VK
	///

	unsafe {
		context.device.end_command_buffer(cmd).unwrap();

		let queue = context.device.get_device_queue(context.queue_families.graphics, 0);
		let submit_info = vk::SubmitInfo::default()
			.command_buffers(slice::from_ref(&cmd));

		context.device.queue_submit(queue, slice::from_ref(&submit_info), frame.in_flight_fence).expect("Failed to submit command buffer!");
	}

	///
	/// END XR
	///

	swapchain.handle.release_image().expect("Failed to release image!");

	let rect = xr::Rect2Di {
		offset: xr::Offset2Di { x: 0, y: 0 },
		extent: xr::Extent2Di {
			width: swapchain.resolution.x as _,
			height: swapchain.resolution.y as _,
		},
	};

	let composition_projection_views: [_; 2] = std::array::from_fn(|i| {
		let xr::View { pose, fov } = views[i];
		xr::CompositionLayerProjectionView::new()
			.pose(pose)
			.fov(fov)
			.sub_image(xr::SwapchainSubImage::new()
				.swapchain(&swapchain.handle)// Ignore the borrow checker.
				.image_array_index(i as _)
				.image_rect(rect)
			)
	});

	let layer_projection = xr::CompositionLayerProjection::new()
		.space(&space)// Ignore the borrow checker.
		.views(&composition_projection_views);

	let result = frame_stream.end(frame_state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[&*layer_projection]);
	match result {
		Ok(()) => {},
		Err(xr::sys::Result::ERROR_POSE_INVALID) => println!("Pose invalid!: {:?}", views.map(|xr::View { pose, fov }| (pose, fov))),
		Err(e) => panic!("Failed to end frame: {e}"),
	}

	if let Some(renderdoc) = renderdoc.as_mut() {
		if renderdoc.is_frame_capturing() {
			renderdoc.end_frame_capture(context.device.handle().as_raw() as *const _, std::ptr::null());
		}
	}
}

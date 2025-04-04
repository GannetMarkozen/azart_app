use std::ffi::CStr;
use std::mem::offset_of;
use std::num::NonZero;
use std::ops::Deref;
use std::{mem, slice};
use std::slice::Windows;
use bevy::prelude::*;
use std::sync::Arc;
use bevy::window::{PresentMode, PrimaryWindow, RequestRedraw, WindowCreated, WindowResized};
use bevy::winit::{WakeUp, WinitWindows};
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use super::{GpuContext, GpuContextHandle, SwapchainCreateInfo};
use super::Swapchain;
use ash::vk;
use bevy::reflect::{DynamicTypePath, TypeRegistry};
use image::GenericImageView;
use crate::graphics_pipeline::{GraphicsPipeline, GraphicsPipelineCreateInfo, VertexAttribute, VertexInput};
use crate::image::{Image, ImageCreateInfo};
use azart_gfx_utils::{Format, MsaaCount, ShaderPath, TriangleFillMode};
use azart_utils::debug_string::*;
use azart_gfx_utils::{asset_path, shader_path};
use gpu_allocator::MemoryLocation;
use std140::repr_std140;
use crate::buffer::{Buffer, BufferCreateInfo};
use crate::render_settings::RenderSettings;


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
	Vertex::new(Vec3::new(-1.0, -1.0,  1.0), Vec2::new(0.0, 0.0)), // 0
	Vertex::new(Vec3::new( 1.0, -1.0,  1.0), Vec2::new(1.0, 0.0)), // 1
	Vertex::new(Vec3::new( 1.0,  1.0,  1.0), Vec2::new(1.0, 1.0)), // 2
	Vertex::new(Vec3::new(-1.0,  1.0,  1.0), Vec2::new(0.0, 1.0)), // 3

	// -------------------------
	// RIGHT FACE (+X)
	Vertex::new(Vec3::new( 1.0, -1.0,  1.0), Vec2::new(0.0, 0.0)), // 4
	Vertex::new(Vec3::new( 1.0, -1.0, -1.0), Vec2::new(1.0, 0.0)), // 5
	Vertex::new(Vec3::new( 1.0,  1.0, -1.0), Vec2::new(1.0, 1.0)), // 6
	Vertex::new(Vec3::new( 1.0,  1.0,  1.0), Vec2::new(0.0, 1.0)), // 7

	// -------------------------
	// BACK FACE (-Z)
	Vertex::new(Vec3::new( 1.0, -1.0, -1.0), Vec2::new(0.0, 0.0)), // 8
	Vertex::new(Vec3::new(-1.0, -1.0, -1.0), Vec2::new(1.0, 0.0)), // 9
	Vertex::new(Vec3::new(-1.0,  1.0, -1.0), Vec2::new(1.0, 1.0)), // 10
	Vertex::new(Vec3::new( 1.0,  1.0, -1.0), Vec2::new(0.0, 1.0)), // 11

	// -------------------------
	// LEFT FACE (-X)
	Vertex::new(Vec3::new(-1.0, -1.0, -1.0), Vec2::new(0.0, 0.0)), // 12
	Vertex::new(Vec3::new(-1.0, -1.0,  1.0), Vec2::new(1.0, 0.0)), // 13
	Vertex::new(Vec3::new(-1.0,  1.0,  1.0), Vec2::new(1.0, 1.0)), // 14
	Vertex::new(Vec3::new(-1.0,  1.0, -1.0), Vec2::new(0.0, 1.0)), // 15

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


#[derive(Default)]
pub struct RenderPlugin {
	pub settings: RenderSettings,
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
		
		let context = Arc::new(GpuContext::new(&extensions));

		app
			.insert_resource(self.settings.clone())
			.insert_resource(GpuContextHandle::new(context))
			.add_systems(PreUpdate, create_swapchain_on_window_spawned)
			.add_systems(PreUpdate, create_render_pass_on_primary_window_spawned);

		// TMP
		app
			.add_systems(Startup, init)
			.add_systems(Update, render);
	}
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
		msaa: MsaaCount,
	) -> Self {
		let msaa_color_attachment = match msaa.enabled() {
			true => Some({
				let create_info = ImageCreateInfo {
					resolution,
					format: Format::RgbaU8Norm,
					usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
					msaa,
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
				..default()
			};
			
			Image::new(dbgfmt!("{name}_depth_stencil"), Arc::clone(&context), &create_info)
		};

		let render_pass = {
			let attachments = match &msaa_color_attachment {
				Some(msaa_color_attachment) => vec![
					vk::AttachmentDescription::default()
						.format(msaa_color_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(msaa.as_vk_sample_count()),
					vk::AttachmentDescription::default()
						.format(depth_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(msaa.as_vk_sample_count()),
					vk::AttachmentDescription::default()
						.format(color_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::DONT_CARE)
						.samples(vk::SampleCountFlags::TYPE_1),
				],
				None => vec![
					vk::AttachmentDescription::default()
						.format(color_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)// TODO: Conditionally use DONT_CARE on certain platforms perchance.
						.samples(vk::SampleCountFlags::TYPE_1),
					vk::AttachmentDescription::default()
						.format(depth_attachment.format.into())
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(vk::SampleCountFlags::TYPE_1),
				]
			};
			
			let color_attachment_ref = vk::AttachmentReference::default()
				.attachment(0)
				.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
			
			let depth_attachment_ref = vk::AttachmentReference::default()
				.attachment(1)
				.layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

			// Only needed for Msaa.
			let resolve_color_attachment_ref = vk::AttachmentReference::default()
				.attachment(2)
				.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

			let color_attachments = match msaa.enabled() {
				true => vec![
					color_attachment_ref,
					resolve_color_attachment_ref,
				],
				false => vec![
					color_attachment_ref
				],
			};

			let subpass = match msaa.enabled() {
				true => vk::SubpassDescription::default()
					.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
					.color_attachments(&color_attachments)
					.resolve_attachments(slice::from_ref(&resolve_color_attachment_ref))
					.depth_stencil_attachment(&depth_attachment_ref),
				false => vk::SubpassDescription::default()
					.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
					.color_attachments(&color_attachments)
					.depth_stencil_attachment(&depth_attachment_ref),
			};
			
			let dependencies = vk::SubpassDependency::default()
				.src_subpass(vk::SUBPASS_EXTERNAL)
				.dst_subpass(0)
				.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
				.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
				.src_access_mask(vk::AccessFlags::empty())
				.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

			let create_info = vk::RenderPassCreateInfo::default()
				.attachments(&attachments)
				.subpasses(slice::from_ref(&subpass))
				.dependencies(slice::from_ref(&dependencies));
			
			unsafe { context.device.create_render_pass(&create_info, None) }.unwrap()
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
				.layers(1);
			
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

fn render(
	context: Res<GpuContextHandle>,
	time: Res<Time>,
	mut base_pass: ResMut<BasePass>,
	base_material: ResMut<BaseMaterial>,
	model: Res<Model>,
	mut query: Query<&mut Swapchain>,
	settings: Res<RenderSettings>,
) {
	unsafe {
		for mut swapchain in query.iter_mut() {
			let previous_frame_index = swapchain.current_frame_index;
			
			let (new_frame_index, suboptimal) = swapchain.acquire_next_image();
			
			let previous_frame = &swapchain.frames()[previous_frame_index];

			let cmd = previous_frame.graphics_command_buffer;
			context.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();

			// Update uniforms.
			{
				let model_matrix = Mat4::from_rotation_y(((time.elapsed_secs() * 45.0) % 360.0).to_radians());
				let per_frame_view_matrices = unsafe { slice::from_raw_parts_mut(model.view_matrices_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut ViewMatrices, settings.frames_in_flight) };
				per_frame_view_matrices[new_frame_index].model = unsafe { mem::transmute_copy(&model_matrix) };
			}
			
			{
				let info = vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

				context.device.begin_command_buffer(cmd, &info).unwrap();
			}
			
			record(cmd, &context, &base_pass, &base_material, &swapchain, &model, new_frame_index, &settings);

			// End recording.
			context.device.end_command_buffer(cmd).unwrap();

			let queue = context.device.get_device_queue(context.queue_families.graphics, 0);

			let submit_info = vk::SubmitInfo::default()
				.command_buffers(slice::from_ref(&cmd))
				.wait_semaphores(slice::from_ref(&previous_frame.image_available_semaphore))
				.wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
				.signal_semaphores(slice::from_ref(&previous_frame.render_finished_semaphore));

			context.device.queue_submit(queue, slice::from_ref(&submit_info), previous_frame.in_flight_fence).unwrap();

			let image_indices = [new_frame_index as u32];

			let present_info = vk::PresentInfoKHR::default()
				.swapchains(slice::from_ref(&swapchain.handle))
				.image_indices(&image_indices)
				.wait_semaphores(slice::from_ref(&previous_frame.render_finished_semaphore));

			let result = context.extensions.swapchain.queue_present(queue, &present_info);
			
			assert_ne!(result, Err(vk::Result::SUBOPTIMAL_KHR));

			// Conditionally recreate swapchain if required.
			if suboptimal || matches!(result, Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR)) {
				info!("Swapchain is not optimal after present! Recreating...");
				
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
	swapchain: &Swapchain,
	model: &Model,
	frame_index: usize,
	settings: &RenderSettings,
) {
	// Begin render pass.
	{
		let attachment_count = match &base_pass.msaa_color_attachment {
			Some(_) => 3,
			None => 2,
		};
		
		let clear_values = (0..attachment_count)
			.map(|_| vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })
			.collect::<Vec<_>>();
		
		let UVec2 { x: width, y: height } = swapchain.extent();
		
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
		let UVec2 { x: width, y: height } = swapchain.extent();
		
		let viewports = [
			vk::Viewport::default()
				.width(width as f32)
				.height(height as f32)
				.min_depth(1.0)// Reverse-Z for better precision afar.
				.max_depth(0.0),
		];
		
		context.device.cmd_set_viewport(cmd, 0, &viewports);
	}

	// Set scissor.
	{
		let UVec2 { x: width, y: height } = swapchain.extent();

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

	context.device.cmd_draw_indexed(cmd, CUBE_INDICES.len() as u32, 2, 0, 0, 0);

	// End render pass.
	{
		context.device.cmd_end_render_pass(cmd);
	}

	// Copy image to swapchain.
	{
		let frame = &swapchain.frames()[frame_index];
		let image = frame.image;

		context.cmd_transition_image_layout(cmd, image, swapchain.format(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, 1);

		let regions = [
			vk::ImageCopy::default()
				.extent(vk::Extent3D::default()
					.width(swapchain.extent().x)
					.height(swapchain.extent().y)
					.depth(1)
				)
				.src_subresource(vk::ImageSubresourceLayers::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.layer_count(1)
				)
				.dst_subresource(vk::ImageSubresourceLayers::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.layer_count(1)
				)
		];

		context.device.cmd_copy_image(cmd, base_pass.color_attachment.handle, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);

		context.cmd_transition_image_layout(cmd, image, swapchain.format(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, 1);
	}
}

fn create_render_pass_on_primary_window_spawned(
	mut commands: Commands,
	mut events: EventReader<WindowCreated>,
	context: Res<GpuContextHandle>,
	query: Query<&Window, With<PrimaryWindow>>,
	settings: Res<RenderSettings>,
) {
	for &WindowCreated { window: e } in events.read() {
		let Ok(window) = query.get(e) else {
			continue;
		};

		let base_pass = BasePass::new("base_pass".into(), Arc::clone(&context), window.resolution.physical_size(), settings.msaa);

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

		let descriptor_pool = {
			let pool_sizes = [
				vk::DescriptorPoolSize::default()
					.ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
					.descriptor_count(1),
			];

			let create_info = vk::DescriptorPoolCreateInfo::default()
				.max_sets(settings.frames_in_flight as u32)
				.pool_sizes(&pool_sizes);

			unsafe { context.device.create_descriptor_pool(&create_info, None) }.unwrap()
		};

		let texture = {
			//let path = asset_path("models/FlightHelmet/FlightHelmet_Materials_MetalPartsMat_OcclusionRoughMetal.png");

			let path = asset_path("models/chom/chom.jpg");
			let image_data = image::ImageReader::open(&*path)
				.unwrap_or_else(|e| panic!("Failed to read data path {path:?}. Error: {e}"))
				.decode()
				.expect("Failed to decode image {path:?}!");

			let create_info = ImageCreateInfo {
				resolution: UVec2::new(image_data.width(), image_data.height()),
				format: Format::RgbaU8Norm,
				usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
				msaa: MsaaCount::None,
				..default()
			};

			use image::DynamicImage::*;
			let format = match &image_data {
				ImageRgba8(_) => "rgba8",
				ImageRgb8(_) => "rgb8",
				ImageRgba16(_) => "rgba16",
				ImageRgb16(_) => "rgb16",
				_ => "unknown",
			};

			println!("Image format: {format:?}");

			let image = Image::new("texture".into(), Arc::clone(&context), &create_info);

			println!("Allocation size: {}", image.allocation.size());
			context.upload_image_buffer(&image, |data| {
				image_data
					.pixels()
					.zip(data.chunks_exact_mut(4))
					.for_each(|((_, _, image::Rgba(rgba)), data)| data.copy_from_slice(&rgba));
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
				let view = Mat4::look_at_rh(Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO, Vec3::Y);
				let proj = Mat4::perspective_rh(45.0f32.to_radians(), window.resolution.physical_size().x as f32 / window.resolution.physical_size().y as f32, 0.1, 100.0);

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

		let descriptor_sets = {
			assert_eq!(pipeline.descriptor_set_layouts.len(), 1);

			let set_layouts = (0..settings.frames_in_flight)
				.flat_map(|_| pipeline.descriptor_set_layouts.iter().copied())
				.collect::<Vec<_>>();

			let allocate_info = vk::DescriptorSetAllocateInfo::default()
				.descriptor_pool(descriptor_pool)
				.set_layouts(&set_layouts);

			let descriptor_sets = unsafe { context.device.allocate_descriptor_sets(&allocate_info) }.unwrap();

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

		commands.insert_resource(base_pass);
		commands.insert_resource(base_material);
		commands.insert_resource(Texture(texture));
		commands.insert_resource(Model {
			name: "cube".into(),
			index_buffer,
			vertex_buffer,
			view_matrices_buffer,
		});
	}
}

fn create_swapchain_on_window_spawned(
	mut commands: Commands,
	mut events: EventReader<WindowCreated>,
	context: Res<GpuContextHandle>,
	windows: NonSend<WinitWindows>,
	query: Query<&Window, Without<Swapchain>>,
	settings: Res<RenderSettings>,
) {
	for &WindowCreated { window: e } in events.read() {
		// If this fails then there's already a swapchain associated with the entity.
		let Ok(window) = query.get(e) else {
			continue;
		};

		let swapchain = {
			let (display_handle, window_handle) = {
				let window = windows.get_window(e).unwrap_or_else(|| panic!("Failed to get window for entity {e}!"));
				(window.display_handle().unwrap(), window.window_handle().unwrap())
			};
			
			let create_info = SwapchainCreateInfo {
				present_mode: window.present_mode,
				usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
				image_count: settings.frames_in_flight as u32,
				..default()
			};
			
			Swapchain::new("swapchain".into(), Arc::clone(&context), display_handle, window_handle, &create_info)
		};

		// Unfortunate archetype traversal. Should happen very rarely though.
		commands.entity(e).insert(swapchain);
	}
}
use std::ffi::CStr;
use std::num::NonZero;
use std::ops::Deref;
use std::slice;
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
use azart_gfx_utils::{MsaaCount, ShaderPath};
use azart_utils::debug_string::*;
use azart_gfx_utils::{asset_path, shader_path};

pub struct RenderPlugin;

#[derive(Reflect)]
pub struct Something {
	pub value: u32,
	pub nothing: String,
}

impl Plugin for RenderPlugin {
	fn build(&self, app: &mut App) {
		let event_loop = app.world().get_non_send_resource::<EventLoop<WakeUp>>().expect("event loop not found!");
		let extensions = ash_window::enumerate_required_extensions(event_loop.display_handle().unwrap().as_raw())
			.expect("Failed to enumerate required extensions!")
			.iter()
			.map(|&x| unsafe { CStr::from_ptr(x) })
			.collect::<Vec<_>>();
		
		let context = Arc::new(GpuContext::new(&extensions));

		app
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
					format: vk::Format::R8G8B8A8_UNORM,
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
				format: vk::Format::R8G8B8A8_UNORM,
				usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
				msaa: MsaaCount::Sample1,// Always 1. If Msaa is enabled this is the resolve image.
				..default()
			};

			Image::new(dbgfmt!("{name}_color_attachment"), Arc::clone(&context), &create_info)
		};

		let depth_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: vk::Format::D32_SFLOAT,
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
						.format(msaa_color_attachment.format)
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(msaa.as_vk_sample_count()),
					vk::AttachmentDescription::default()
						.format(depth_attachment.format)
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)
						.samples(msaa.as_vk_sample_count()),
					vk::AttachmentDescription::default()
						.format(color_attachment.format)
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::DONT_CARE)
						.samples(vk::SampleCountFlags::TYPE_1),
				],
				None => vec![
					vk::AttachmentDescription::default()
						.format(color_attachment.format)
						.initial_layout(vk::ImageLayout::UNDEFINED)
						.final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
						.load_op(vk::AttachmentLoadOp::CLEAR)// TODO: Conditionally use DONT_CARE on certain platforms perchance.
						.samples(vk::SampleCountFlags::TYPE_1),
					vk::AttachmentDescription::default()
						.format(depth_attachment.format)
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
				msaa: MsaaCount::Sample1,
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
	descriptor_set: vk::DescriptorSet,
}

fn init(
	mut commands: Commands,
	mut query: Query<&mut Swapchain>,
) {

}

fn render(
	context: Res<GpuContextHandle>,
	mut base_pass: ResMut<BasePass>,
	base_material: ResMut<BaseMaterial>,
	mut query: Query<&mut Swapchain>,
) {
	unsafe {
		for mut swapchain in query.iter_mut() {
			let previous_frame_index = swapchain.current_frame_index;
			
			let (new_frame_index, suboptimal) = swapchain.acquire_next_image();
			
			let previous_frame = &swapchain.frames()[previous_frame_index];

			let cmd = previous_frame.graphics_command_buffer;
			context.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
			
			{
				let info = vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

				context.device.begin_command_buffer(cmd, &info).unwrap();
			}
			
			record(cmd, &context, &base_pass, &base_material, &swapchain, new_frame_index);

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
	frame_index: usize,
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
	context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, base_material.pipeline.layout, 0, slice::from_ref(&base_material.descriptor_set), &[]);
	
	context.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, base_material.pipeline.handle);
	context.device.cmd_draw(cmd, 3, 1, 0, 0);

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
) {
	for &WindowCreated { window: e } in events.read() {
		let Ok(window) = query.get(e) else {
			continue;
		};

		const MSAA: MsaaCount = MsaaCount::Sample8;
		let base_pass = BasePass::new("base_pass".into(), Arc::clone(&context), window.resolution.physical_size(), MSAA);

		let pipeline = {
			let create_info = GraphicsPipelineCreateInfo {
				vertex_shader: &shader_path("shader.vert"),
				fragment_shader: &shader_path("shader.frag"),
				vertex_inputs: &[],
				msaa: MSAA,
			};

			unsafe { GraphicsPipeline::new("base_material".into(), Arc::clone(&context), base_pass.render_pass, &create_info) }
		};

		let descriptor_pool = {
			let pool_sizes = [
				vk::DescriptorPoolSize::default()
					.ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
					.descriptor_count(1)
			];

			let create_info = vk::DescriptorPoolCreateInfo::default()
				.max_sets(1)
				.pool_sizes(&pool_sizes);

			unsafe { context.device.create_descriptor_pool(&create_info, None) }.unwrap()
		};

		let texture = {
			let path = asset_path("models/FlightHelmet/FlightHelmet_Materials_MetalPartsMat_OcclusionRoughMetal.png");
			let png_data = std::fs::read(&*path)
				.unwrap_or_else(|e| panic!("Failed to read data path {path:?}. Error: {e}"));

			let png = image::load_from_memory(&png_data).unwrap();
			let create_info = ImageCreateInfo {
				resolution: UVec2::new(png.width(), png.height()),
				format: vk::Format::R8G8B8A8_UNORM,
				usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
				msaa: MsaaCount::Sample1,
				..default()
			};

			let image = Image::new("texture".into(), Arc::clone(&context), &create_info);

			context.upload_image(&image, |data| {
				png
					.pixels()
					.zip(data.chunks_exact_mut(4))
					.for_each(|((_, _, image::Rgba(rgba)), data)| data.copy_from_slice(&rgba));
			});

			image
		};

		let descriptor_set = {
			assert_eq!(pipeline.descriptor_set_layouts.len(), 1);

			let allocate_info = vk::DescriptorSetAllocateInfo::default()
				.descriptor_pool(descriptor_pool)
				.set_layouts(&pipeline.descriptor_set_layouts);

			let descriptor_set = unsafe { context.device.allocate_descriptor_sets(&allocate_info) }.unwrap()[0];

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

			let descriptor_writes = [
				vk::WriteDescriptorSet::default()
					.dst_set(descriptor_set)
					.dst_binding(0)
					.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
					.image_info(slice::from_ref(&image_info))
			];

			unsafe { context.device.update_descriptor_sets(&descriptor_writes, &[]); }

			descriptor_set
		};

		let base_material = BaseMaterial {
			pipeline,
			descriptor_pool,
			descriptor_set,
		};

		commands.insert_resource(base_pass);
		commands.insert_resource(base_material);
		commands.insert_resource(Texture(texture));
	}
}

fn create_swapchain_on_window_spawned(
	mut commands: Commands,
	mut events: EventReader<WindowCreated>,
	context: Res<GpuContextHandle>,
	windows: NonSend<WinitWindows>,
	query: Query<&Window, Without<Swapchain>>,
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
				image_count: 2,
				..default()
			};
			
			Swapchain::new("swapchain".into(), Arc::clone(&context), display_handle, window_handle, &create_info)
		};

		// Unfortunate archetype traversal. Should happen very rarely though.
		commands.entity(e).insert(swapchain);
	}
}
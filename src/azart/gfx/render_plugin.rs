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
use ash::vk::Pipeline;
use crate::azart::gfx::graphics_pipeline::{GraphicsPipeline, GraphicsPipelineCreateInfo};
use crate::azart::gfx::image::{Image, ImageCreateInfo};
use crate::azart::gfx::misc::ShaderPath;
use crate::azart::utils::debug_string::*;
use crate::shader_path;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
	fn build(&self, app: &mut App) {
		let event_loop = app.world().get_non_send_resource::<EventLoop<WakeUp>>().expect("event loop not found!");
		let extensions = ash_window::enumerate_required_extensions(event_loop.display_handle()
			.unwrap()
			.as_raw())
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
pub struct BasePass {
	name: DebugString,
	render_pass: vk::RenderPass,
	frame_buffer: vk::Framebuffer,
	color_attachment: Image,
	depth_attachment: Image,
	context: Arc<GpuContext>,
}

impl BasePass {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		resolution: UVec2,
	) -> Self {
		let color_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: vk::Format::R8G8B8A8_UNORM,
				usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
				..default()
			};

			Image::new(dbgfmt!("{name}_color_attachment"), Arc::clone(&context), &create_info)
		};

		let depth_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: vk::Format::D32_SFLOAT,
				usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
				..default()
			};
			
			Image::new(dbgfmt!("{name}_depth_stencil"), Arc::clone(&context), &create_info)
		};

		let render_pass = {
			let attachments = [
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
			];
			
			let color_attachment_ref = vk::AttachmentReference::default()
				.attachment(0)
				.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
			
			let depth_attachment_ref = vk::AttachmentReference::default()
				.attachment(1)
				.layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
			
			let subpass = vk::SubpassDescription::default()
				.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
				.color_attachments(slice::from_ref(&color_attachment_ref))
				.depth_stencil_attachment(&depth_attachment_ref);
			
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
			let attachments = [
				color_attachment.view,
				depth_attachment.view,
			];
			
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
			color_attachment,
			depth_attachment,
			context,
		}
	}

	pub unsafe fn resize(&mut self, resolution: UVec2) {
		self.context.device.destroy_framebuffer(self.frame_buffer, None);

		let (color_attachment, depth_attachment) = Self::create_attachments(&self.name, Arc::clone(&self.context), resolution);
		self.color_attachment = color_attachment;
		self.depth_attachment = depth_attachment;

		let attachments = [
			self.color_attachment.view,
			self.depth_attachment.view,
		];

		let create_info = vk::FramebufferCreateInfo::default()
			.render_pass(self.render_pass)
			.width(resolution.x)
			.height(resolution.y)
			.attachments(&attachments)
			.layers(1);

		self.frame_buffer = self.context.device.create_framebuffer(&create_info, None).unwrap();
	}

	fn create_attachments(
		name: &DebugString,
		context: Arc<GpuContext>,
		resolution: UVec2
	) -> (Image, Image) {
		let color_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: vk::Format::R8G8B8A8_UNORM,
				usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
				..default()
			};

			Image::new(dbgfmt!("{name}_color_attachment"), Arc::clone(&context), &create_info)
		};

		let depth_attachment = {
			let create_info = ImageCreateInfo {
				resolution,
				format: vk::Format::D32_SFLOAT,
				usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
				..default()
			};

			Image::new(dbgfmt!("{name}_depth_stencil"), context, &create_info)
		};

		(color_attachment, depth_attachment)
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
			let new_frame = &swapchain.frames()[new_frame_index];

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
		let begin_info = vk::RenderPassBeginInfo::default()
			.render_pass(base_pass.render_pass)
			.framebuffer(base_pass.frame_buffer)
			.render_area(vk::Rect2D {
				extent: vk::Extent2D {
					width: swapchain.extent().x,
					height: swapchain.extent().y,
				},
				..default()
			})
			.clear_values(&[
				vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
				vk::ClearValue { color: vk::ClearColorValue { float32: [1.0, 1.0, 1.0, 1.0] } },
			]);

		context.device.cmd_begin_render_pass(cmd, &begin_info, vk::SubpassContents::INLINE);
	}
	
	// Set viewport
	{
		let viewports = [
			vk::Viewport::default()
				.width(swapchain.extent().x as f32)
				.height(swapchain.extent().y as f32)
				.min_depth(0.0)
				.max_depth(1.0),
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
		
		let base_pass = BasePass::new("base_pass".into(), Arc::clone(&context), window.resolution.physical_size());
		
		let base_material = BaseMaterial {
			pipeline: {
				let create_info = GraphicsPipelineCreateInfo {
					vertex_shader: shader_path!("shader.vert"),
					fragment_shader: shader_path!("shader.frag"),
				};
				
				unsafe { GraphicsPipeline::new("base_material".into(), Arc::clone(&context), base_pass.render_pass, &create_info) }
			},
		};

		commands.insert_resource(base_pass);
		commands.insert_resource(base_material);
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
				..default()
			};
			
			Swapchain::new("swapchain".into(), Arc::clone(&context), display_handle, window_handle, &create_info)
		};

		// Unfortunate archetype traversal. Should happen very rarely though.
		commands.entity(e).insert(swapchain);
	}
}
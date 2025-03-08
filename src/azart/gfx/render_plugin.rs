use std::ffi::CStr;
use std::num::NonZero;
use std::ops::Deref;
use std::slice;
use std::slice::Windows;
use bevy::prelude::*;
use std::sync::Arc;
use bevy::window::{PresentMode, RequestRedraw, WindowCreated, WindowResized};
use bevy::winit::{WakeUp, WinitWindows};
use winit::event_loop::EventLoop;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use crate::azart::gfx::swapchain::{Swapchain, SwapchainCreateInfo};
use super::context::GpuContext;
use ash::vk;
use bevy::ecs::world::DeferredWorld;
use log::warn;
use crate::azart::gfx::render_loop::RenderModule;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
	fn build(&self, app: &mut App) {
		let event_loop = app.world().get_non_send_resource::<EventLoop<WakeUp>>().expect("event loop not found!");
		let extensions = ash_window::enumerate_required_extensions(event_loop.display_handle()
			.unwrap()
			.as_raw())
			.expect("Failed to enumerate required extensions!")
			.into_iter()
			.map(|&x| unsafe { CStr::from_ptr(x) })
			.collect::<Vec<_>>();

		println!("window extensions: {:?}", extensions);

		app
			.insert_resource(GpuContextResource::new(Arc::new(GpuContext::new(&extensions))))
			.add_systems(PreUpdate, create_swapchain_on_window_spawned);

		// TMP
		app
			.add_systems(Startup, init)
			.add_systems(Update, other_render);
	}
}

fn init(
	mut commands: Commands,
	mut query: Query<&mut RenderModule>,
) {

}

static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn other_render(
	context: Res<GpuContextResource>,
	mut query: Query<&mut RenderModule>,
) {
	unsafe {
		for mut module in query.iter_mut() {
			let frame_index = module.current_frame_index;
			let frame = &module.frames()[frame_index];

			context.device.wait_for_fences(slice::from_ref(&frame.in_flight_fence), true, u64::MAX).unwrap();

			let count = COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
			
			let (next_frame_index, suboptimal) = match context.extensions.swapchain.acquire_next_image(module.swapchain.handle, u64::MAX, frame.image_available_semaphore, vk::Fence::null()) {
				Ok((index, suboptimal)) => (index as usize, suboptimal),// If suboptimal, defer swapchain recreation until after rendering.
				Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {// Immediately recreate swapchain.
					info!("Swapchain is out of date after acquire next image! Recreating...");

					context.wait_idle();
					module.swapchain.recreate();
					continue;
				},
				Err(vk::Result::SUBOPTIMAL_KHR) => unreachable!(),
				Err(e) => panic!("Failed to acquire next image after {} frames! Error {e}", count),
			};
			
			module.current_frame_index = frame_index;
			let frame = &module.frames()[frame_index];

			println!("Acquired image {next_frame_index}. Previous: {}", frame_index);

			// Only reset fences if work actually occured.
			context.device.reset_fences(&[frame.in_flight_fence]).unwrap();

			let cmd = frame.graphics_command_buffer;
			context.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();

			{
				let info = vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

				context.device.begin_command_buffer(cmd, &info);
			}

			// Record.
			{
				let image = module.swapchain.image(next_frame_index);

				context.cmd_transition_image_layout(cmd, image, module.swapchain.format(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, 1);

				let image_subresource_range = vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_mip_level(0)
					.level_count(1)
					.base_array_layer(0)
					.layer_count(1);

				let clear_color_value = vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 1.0] };

				context.device.cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &clear_color_value, slice::from_ref(&image_subresource_range));

				context.cmd_transition_image_layout(cmd, image, module.swapchain.format(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, 1);
			}

			// End recording.
			context.device.end_command_buffer(cmd);

			let queue = context.device.get_device_queue(context.queue_families.graphics, 0);

			let submit_info = vk::SubmitInfo::default()
				.command_buffers(slice::from_ref(&cmd))
				.wait_semaphores(slice::from_ref(&frame.image_available_semaphore))
				.wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
				.signal_semaphores(slice::from_ref(&frame.render_finished_semaphore));

			context.device.queue_submit(queue, slice::from_ref(&submit_info), frame.in_flight_fence).unwrap();

			let image_indices = [next_frame_index as u32];

			let present_info = vk::PresentInfoKHR::default()
				.swapchains(slice::from_ref(&module.swapchain.handle))
				.image_indices(&image_indices)
				.wait_semaphores(slice::from_ref(&frame.render_finished_semaphore));

			let result = context.extensions.swapchain.queue_present(queue, &present_info);
			
			// Conditionally recreate swapchain if required.
			if suboptimal || matches!(result, Ok(true) | Err(vk::Result::SUBOPTIMAL_KHR) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR)) {
				info!("Swapchain is not optimal after present! Recreating...");
				
				context.wait_idle();
				module.swapchain.recreate();
			}
		}
	}
}

fn render(
	time: Res<Time>,
	mut commands: Commands,
	context: Res<GpuContextResource>,
	mut query: Query<&mut RenderModule>,
) {
	for mut module in query.iter_mut() {
		let (index, optimal) = match module.acquire_next_image() {
			Ok(x) => x,
			Err(e) => {
				error!("Failed to acquire next image! Error {e}");
				continue;
			},
		};

		let frame = &module.frames()[index];
		let cmd = frame.graphics_command_buffer;

		unsafe { context.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap(); }

		// Begin recording.
		{
			let info = vk::CommandBufferBeginInfo::default()
				.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

			unsafe { context.device.begin_command_buffer(cmd, &info) }.unwrap();
		}

		// Record.
		unsafe {
			let image = module.swapchain.image(index);


		}

		// End recording.
		unsafe { context.device.end_command_buffer(cmd) }.unwrap();

		unsafe {
			let queue = context.device.get_device_queue(context.queue_families.graphics, 0);

			//let wait_semaphores = [frame.image_available_semaphore, frame.render_finished_semaphore];

			let submit_info = vk::SubmitInfo::default()
				.command_buffers(slice::from_ref(&cmd))
				.wait_semaphores(slice::from_ref(&frame.image_available_semaphore))
				.wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
				.signal_semaphores(slice::from_ref(&frame.render_finished_semaphore));

			context.device.queue_submit(queue, &[submit_info], frame.in_flight_fence).expect("Failed to submit command buffer!");

			let swapchains = [module.swapchain.handle];
			let image_indices = [index as u32];

			let present_info = vk::PresentInfoKHR::default()
				.swapchains(&swapchains)
				.image_indices(&image_indices)
				.wait_semaphores(slice::from_ref(&frame.render_finished_semaphore));

			context.extensions.swapchain.queue_present(queue, &present_info).expect("Failed to present swapchain image!");
		}
	}
}

fn create_swapchain_on_window_spawned(
	mut commands: Commands,
	mut events: EventReader<WindowCreated>,
	context: Res<GpuContextResource>,
	windows: NonSend<WinitWindows>,
	query: Query<&Window, Without<RenderModule>>,
) {
	for &WindowCreated { window: e } in events.read() {
		// If this fails then there's already a swapchain associated with the entity.
		let Ok(window) = query.get(e) else {
			continue;
		};

		let surface = {
			let (display_handle, window_handle) = {
				let window = windows.get_window(e).unwrap_or_else(|| panic!("Failed to get window for entity {e}!"));
				(window.display_handle().unwrap(), window.window_handle().unwrap())
			};
			RenderModule::new(Arc::clone(&*context), display_handle, window_handle, window.present_mode, Some(window.resolution.physical_size()))
		};

		// Unfortunate archetype traversal. Should happen very rarely though.
		commands.entity(e).insert(surface);
	}
}

fn resize_swapchain(
	mut events: EventReader<WindowResized>,
	mut query: Query<&mut RenderModule>,
) {
	for &WindowResized { window: e, width, height } in events.read() {
		let Ok(mut surface) = query.get_mut(e) else {
			continue;
		};

		surface.swapchain.resize(Some(UVec2 { x: width as u32, y: height as u32 }));
	}
}

#[derive(Resource, Clone)]
pub struct GpuContextResource(pub Arc<GpuContext>);

impl GpuContextResource {
	pub const fn new(context: Arc<GpuContext>) -> Self {
		Self(context)
	}
}

impl Deref for GpuContextResource {
	type Target = Arc<GpuContext>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}
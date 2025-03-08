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
use super::{GpuContext, GpuContextHandle, SwapchainCreateInfo};
use super::Swapchain;
use ash::vk;
use bevy::ecs::world::DeferredWorld;
use log::warn;
use crate::azart::gfx::image::{Image, ImageCreateInfo};
use crate::azart::utils::debug_string::*;

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

		println!("window extensions: {:?}", extensions);

		app
			.insert_resource(GpuContextHandle::new(Arc::new(GpuContext::new(&extensions))))
			.add_systems(PreUpdate, create_swapchain_on_window_spawned);

		// TMP
		app
			.add_systems(Startup, init)
			.add_systems(Update, render);
	}
}

fn init(
	mut commands: Commands,
	mut query: Query<&mut Swapchain>,
) {

}

fn render(
	context: Res<GpuContextHandle>,
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
			
			let some_image = Image::new(dbgstr!("image[{new_frame_index}]"), Arc::clone(&*context), &ImageCreateInfo::default());

			{
				let info = vk::CommandBufferBeginInfo::default()
					.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

				context.device.begin_command_buffer(cmd, &info).unwrap();
			}

			// Record.
			{
				let image = new_frame.image;

				context.cmd_transition_image_layout(cmd, image, swapchain.format(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, 1);

				let image_subresource_range = vk::ImageSubresourceRange::default()
					.aspect_mask(vk::ImageAspectFlags::COLOR)
					.base_mip_level(0)
					.level_count(1)
					.base_array_layer(0)
					.layer_count(1);

				let clear_color_value = vk::ClearColorValue { float32: [0.0, 0.0, 1.0, 1.0] };

				context.device.cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &clear_color_value, slice::from_ref(&image_subresource_range));

				context.cmd_transition_image_layout(cmd, image, swapchain.format(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, 1);
			}

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
			}
		}
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
			
			Swapchain::new(dbgstr!("swapchain"), Arc::clone(&*context), display_handle, window_handle, &create_info)
		};

		// Unfortunate archetype traversal. Should happen very rarely though.
		commands.entity(e).insert(swapchain);
	}
}

fn resize_swapchain(
	mut events: EventReader<WindowResized>,
	mut query: Query<&mut Swapchain>,
) {
	for &WindowResized { window: e, width, height } in events.read() {
		let Ok(mut swapchain) = query.get_mut(e) else {
			continue;
		};

		swapchain.recreate(Some(UVec2 { x: width as u32, y: height as u32 }), None);
	}
}
use crate::azart::utils::debug_string::DebugString;
use std::sync::Arc;
use bevy::prelude::*;
use crate::azart::gfx::swapchain::{Swapchain, SwapchainCreateInfo};
use ash::vk;
use bevy::window::PresentMode;
use winit::raw_window_handle::{DisplayHandle, WindowHandle};
use crate::azart::{gfx::context::GpuContext, utils::debug_string::dbgfmt};

// Represents everything required to submit commands and render to a surface.
#[derive(Component)]
pub struct RenderModule {
	frames: Vec<FrameState>,
	pub current_frame_index: usize,
	pub swapchain: Swapchain,
}

impl RenderModule {
	pub fn new(context: Arc<GpuContext>, display_handle: DisplayHandle, window_handle: WindowHandle, present_mode: PresentMode, extent: Option<UVec2>) -> Self {
		const IMAGE_COUNT: u32 = 2;// TODO: Should be parameterized.

		let frames = (0..IMAGE_COUNT)
			.map(|i| {
				let graphics_command_pool = {
					let create_info = vk::CommandPoolCreateInfo::default()
						.queue_family_index(context.queue_families.graphics)
						.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

					unsafe { context.device.create_command_pool(&create_info, None) }.expect("Failed to create command pool!")
				};

				let graphics_command_buffer = {
					let create_info = vk::CommandBufferAllocateInfo::default()
						.command_pool(graphics_command_pool)
						.level(vk::CommandBufferLevel::PRIMARY)
						.command_buffer_count(1);

					unsafe { context.device.allocate_command_buffers(&create_info) }.expect("Failed to allocate command buffers!")[0]
				};

				let in_flight_fence = {
					let create_info = vk::FenceCreateInfo::default()
						.flags(vk::FenceCreateFlags::SIGNALED);

					unsafe { context.device.create_fence(&create_info, None) }.expect("Failed to create fence!")
				};

				let (image_available_semaphore, render_finished_semaphore) = {
					let create_info = vk::SemaphoreCreateInfo::default();

					unsafe {
						(
							context.device.create_semaphore(&create_info, None).expect("Failed to create semaphore!"),
							context.device.create_semaphore(&create_info, None).expect("Failed to create semaphore!"),
						)
					}
				};
				
				#[cfg(debug_assertions)]
				unsafe {
					context.set_debug_name_for_gpu_resource(format!("graphics_command_pool[{i}]").as_str(), graphics_command_pool);
					context.set_debug_name_for_gpu_resource(format!("graphics_command_buffer[{i}]").as_str(), graphics_command_buffer);
					context.set_debug_name_for_gpu_resource(format!("in_flight_fence[{i}]").as_str(), in_flight_fence);
					context.set_debug_name_for_gpu_resource(format!("image_available_semaphore[{i}]").as_str(), image_available_semaphore);
					context.set_debug_name_for_gpu_resource(format!("render_finished_semaphore[{i}]").as_str(), render_finished_semaphore);
				}

				FrameState {
					graphics_command_pool,
					graphics_command_buffer,
					in_flight_fence,
					image_available_semaphore,
					render_finished_semaphore,
				}
			})
			.collect();

		let swapchain = {
			let create_info = SwapchainCreateInfo {
				extent,
				image_count: IMAGE_COUNT,
				present_mode: match present_mode {
					PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
					PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
					PresentMode::Fifo => vk::PresentModeKHR::FIFO,
					PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
					_ => vk::PresentModeKHR::FIFO,
				},
				..default()
			};
			
			Swapchain::new(dbgfmt!("swapchain"), context, display_handle, window_handle, &create_info)
		};

		Self {
			frames,
			current_frame_index: 0,
			swapchain,
		}
	}

	// @NOTE: Flushes and waits for the work on the frame attempting to be acquired before acquisition.
	// Returns index on success. Else returns Err(Ok(())) if a resize is required. Else a vulkan error.
	pub fn acquire_next_image(&mut self) -> Result<(usize, bool), vk::Result> {
		unsafe {
			trace!("Waiting for fence...");
			self.context().device.wait_for_fences(&[self.frames[self.current_frame_index].in_flight_fence], true, u64::MAX).expect("Failed to wait for fence!");
			self.context().device.reset_fences(&[self.frames[self.current_frame_index].in_flight_fence]).expect("Failed to reset fence!");
		}

		let current_frame = &self.frames[self.current_frame_index];
		let (index, optimal) = self.swapchain.acquire_next_image(Some(current_frame.image_available_semaphore), None, None)?;

		assert!(index < self.frames.len());

		self.current_frame_index = index;
		
		Ok((index, optimal))
	}

	pub fn frames(&self) -> &[FrameState] {
		&self.frames
	}

	pub fn context(&self) -> &GpuContext {
		&*self.swapchain.context
	}
}

impl Drop for RenderModule {
	fn drop(&mut self) {
		unsafe {
			self.context().wait_idle();
			
			for frame in self.frames.iter() {
				self.context().device.free_command_buffers(frame.graphics_command_pool, &[frame.graphics_command_buffer]);
				self.context().device.destroy_command_pool(frame.graphics_command_pool, None);

				self.context().device.destroy_fence(frame.in_flight_fence, None);
				self.context().device.destroy_semaphore(frame.image_available_semaphore, None);
				self.context().device.destroy_semaphore(frame.render_finished_semaphore, None);
			}
		}
	}
}

pub struct FrameState {
	pub graphics_command_pool: vk::CommandPool,
	pub graphics_command_buffer: vk::CommandBuffer,
	pub in_flight_fence: vk::Fence,
	pub image_available_semaphore: vk::Semaphore,
	pub render_finished_semaphore: vk::Semaphore,
}
use std::num::NonZeroUsize;
use std::sync::Arc;
use azart_utils::debug_string::DebugString;
use ash::vk;
use azart_gfx_utils::Msaa;
use crate::GpuContext;

// @TODO: Make safe ctor.
pub struct RenderPass {
	pub name: DebugString,
	pub context: Arc<GpuContext>,
	pub handle: vk::RenderPass,
	pub msaa: Msaa,
	pub multiview_count: Option<NonZeroUsize>,
}

impl Drop for RenderPass {
	fn drop(&mut self) {
		unsafe {
			self.context.device.destroy_render_pass(self.handle, None);
		}
	}
}
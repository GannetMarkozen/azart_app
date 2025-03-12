use std::sync::Arc;
use ash::vk;
use crate::azart::gfx::GpuContext;
use crate::azart::gfx::misc::MsaaCount;
use crate::azart::utils::debug_string::DebugString;

pub struct RenderPass {
	name: DebugString,
	pub(crate) handle: vk::RenderPass,
	pub(crate) context: Arc<GpuContext>,
}

// TODO
impl RenderPass {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		info: &RenderPassInfo,
	) -> Self {
		todo!();
		/*let render_pass = {
			let attachments = [
				vk::AttachmentDescription::default()
					.format(vk::Format::R8G8B8A8_SRGB)
			];

			let create_info = vk::RenderPassCreateInfo::default()
				.attachments(&attachments);

		};

		Self {
			name,
			context,
		}*/
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPassInfo {
	pub msaa: MsaaCount,
}
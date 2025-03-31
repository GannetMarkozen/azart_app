use std::any::Any;
use std::sync::Arc;
use ash::vk;
use bevy::utils::HashSet;
use crate::GpuContext;

pub struct CommandBuffer {
	pub(crate) handle: vk::CommandBuffer,
	pub(crate) fence: vk::Fence,
	pub(crate) resources_in_use: HashSet<Arc<dyn Any>>,
	pub(crate) context: Arc<GpuContext>,
}
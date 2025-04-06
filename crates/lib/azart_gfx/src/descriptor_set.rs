use std::sync::Arc;
use ash::vk;
use azart_utils::debug_string::DebugString;
use bevy::prelude::*;
use crate::GpuContext;

pub struct DescriptorSet {
	name: DebugString,
	pub(crate) handle: vk::DescriptorSet,
	pub(crate) context: Arc<GpuContext>,
}

impl DescriptorSet {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		bindings: &[u32],
	) -> Self {
		todo!();
	}
}

#[derive(Resource)]
pub struct DescriptorPool {
	name: DebugString,
	pub(crate) handle: vk::DescriptorPool,
	pub(crate) context: Arc<GpuContext>,
}

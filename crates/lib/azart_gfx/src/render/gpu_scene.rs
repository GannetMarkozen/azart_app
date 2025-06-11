use std::sync::Arc;
use bevy::math::Mat4;
use bevy::prelude::{Deref, DerefMut, Resource};
use crate::buffer::Buffer;

#[derive(Resource)]
pub(crate) struct GpuScene {
	pub global_ubo: Buffer,
}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct RenderFrameIndex(pub u64);

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GlobalUbo {
	pub(crate) views: [Mat4; 2],// 2 if in VR.
}
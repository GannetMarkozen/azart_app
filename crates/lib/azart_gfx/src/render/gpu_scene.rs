use std::sync::Arc;
use bevy::math::Mat4;
use bevy::prelude::{Deref, DerefMut, Resource};
use crate::buffer::Buffer;

#[derive(Resource)]
pub(crate) struct GpuScene {
	/// The ubo responsible for global data such as view matrices.
	pub global_ubo: Buffer,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct GlobalUbo {
	pub(crate) views: [[[f32; 4]; 4]; 2],// 2 if in VR. Only 0th index is valid outside of VR.
}
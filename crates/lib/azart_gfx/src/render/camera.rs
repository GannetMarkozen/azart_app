use bevy::prelude::*;
use ash::vk;
use crate::buffer::Buffer;
use openxr as xr;

#[derive(Debug, Component)]
pub struct Camera {
	pub fov: f32,
}

#[derive(Debug, Component)]
pub struct XrCamera {
	pub views: [XrView; 2],
	pub ipd: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct XrView {
	pub pos: Vec3,
	pub rot: Quat,
	pub fov: xr::Fovf,
}
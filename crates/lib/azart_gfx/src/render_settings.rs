use azart_gfx_utils::Msaa;
use bevy::prelude::*;

#[derive(Clone, Debug, Reflect, Resource)]
pub struct RenderSettings {
	pub msaa: Msaa,
	pub frames_in_flight: usize,
}

impl Default for RenderSettings {
	fn default() -> Self {
		Self {
			msaa: Msaa::x4,
			frames_in_flight: 3,
		}
	}
}

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash, Reflect, States)]
pub enum DisplayMode {
	#[default]
	Std,
	Xr,
}
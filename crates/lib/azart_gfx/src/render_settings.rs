use azart_gfx_utils::MsaaCount;
use bevy::prelude::*;

#[derive(Clone, Debug, Reflect, Resource)]
pub struct RenderSettings {
	pub msaa: MsaaCount,
	pub frames_in_flight: usize,
}

impl Default for RenderSettings {
	fn default() -> Self {
		Self {
			msaa: MsaaCount::Sample4,
			frames_in_flight: 2,
		}
	}
}

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash, Reflect, States)]
pub enum DisplayMode {
	#[default]
	Standard,
	Xr,
}
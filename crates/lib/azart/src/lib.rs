use bevy::{prelude::*, a11y::AccessibilityPlugin};
use bevy::diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::input::InputPlugin;
use bevy::log::LogPlugin;
use bevy::MinimalPlugins;
use bevy::state::app::StatesPlugin;
use bevy::window::{PresentMode, WindowMode};
use bevy::winit::{WakeUp, WinitPlugin};

pub struct AzartPlugin;

impl Plugin for AzartPlugin {
	fn build(&self, app: &mut App) {
		app
			.add_plugins(MinimalPlugins)
			.add_plugins(StatesPlugin::default())
			.add_plugins(AssetPlugin::default())
			.add_plugins(InputPlugin::default())
			.add_plugins(WindowPlugin {
				primary_window: Some(Window {
					title: "azart".to_owned(),
					present_mode: PresentMode::Fifo,
					focused: true,
					..default()
				}),
				..default()
			})
			.add_plugins(WinitPlugin::<WakeUp>::default())
			.add_plugins(AccessibilityPlugin)
			.add_plugins(LogPlugin::default());

		#[cfg(feature = "gfx")]
		app.add_plugins(azart_gfx::render_plugin::RenderPlugin::default());

		//#[cfg(debug_assertions)]
		app.add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default(), DiagnosticsPlugin));
	}
}
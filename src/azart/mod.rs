use bevy::{prelude::*, a11y::AccessibilityPlugin};
use bevy::diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::input::InputPlugin;
use bevy::log::LogPlugin;
use bevy::MinimalPlugins;
use bevy::window::PresentMode;
use bevy::winit::{WakeUp, WinitPlugin};
use crate::azart::gfx::render_plugin::RenderPlugin;

pub mod gfx;
pub mod utils;
pub mod mesh;
pub mod prelude;
pub mod assets;

pub struct AzartPlugin;

impl Plugin for AzartPlugin {
	fn build(&self, app: &mut App) {
		app
			.add_plugins(MinimalPlugins)
			.add_plugins(AssetPlugin::default())
			.add_plugins(InputPlugin::default())
			.add_plugins(WindowPlugin {
				primary_window: Some(Window {
					title: "azart".to_owned(),
					present_mode: PresentMode::Mailbox,
					..default()
				}),
				..default()
			})
			.add_plugins(WinitPlugin::<WakeUp>::default())
			.add_plugins(AccessibilityPlugin)
			.add_plugins(LogPlugin::default())
			.add_plugins(RenderPlugin);
		
		//#[cfg(debug_assertions)]
		app.add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default(), DiagnosticsPlugin));
	}
}
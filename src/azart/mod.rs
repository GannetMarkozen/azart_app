use bevy::{prelude::*, a11y::AccessibilityPlugin};
use bevy::gltf::GltfPlugin;
use bevy::input::InputPlugin;
use bevy::input::keyboard::KeyboardInput;
use bevy::log::LogPlugin;
use bevy::MinimalPlugins;
use bevy::winit::{WakeUp, WinitPlugin};
use crate::azart::gfx::render_plugin::RenderPlugin;

pub mod gfx;
mod utils;
//mod window;

pub struct AzartPlugin;

impl Plugin for AzartPlugin {
	fn build(&self, app: &mut App) {
		app
			.add_plugins(MinimalPlugins)
			.add_plugins(AssetPlugin::default())
			.add_plugins(GltfPlugin::default())
			.add_plugins(InputPlugin::default())
			.add_plugins(WindowPlugin {
				primary_window: Some(Window {
					title: "Azart".to_owned(),
					..default()
				}),
				..default()
			})
			.add_plugins(WinitPlugin::<WakeUp>::default())
			.add_plugins(AccessibilityPlugin)
			.add_plugins(LogPlugin::default())
			.add_plugins(RenderPlugin);
	}
}
pub mod prelude;

use bevy::{prelude::*, a11y::AccessibilityPlugin};
use bevy::diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::input::InputPlugin;
use bevy::log::LogPlugin;
use bevy::MinimalPlugins;
use bevy::state::app::StatesPlugin;
use bevy::window::{ExitCondition, PresentMode, WindowMode};
use bevy::winit::{WakeUp, WinitPlugin};

pub struct AzartPlugin;

impl Plugin for AzartPlugin {
	fn build(&self, app: &mut App) {
		app
			.add_plugins(MinimalPlugins)
			.add_plugins(StatesPlugin::default())
			.add_plugins(AssetPlugin::default())
			.add_plugins(InputPlugin::default())
			.add_plugins(azart_asset::AssetPlugin)
			.add_plugins(WindowPlugin {
				primary_window: Some(Window {
					title: "azart".to_owned(),
					present_mode: PresentMode::Immediate,
					focused: true,
					..default()
				}),
				..default()
			})
			.add_plugins(WinitPlugin::<WakeUp>::default())
			.add_plugins(AccessibilityPlugin)
			.add_plugins(LogPlugin::default());

		#[cfg(feature = "gfx")]
		app.add_plugins(azart_gfx::render::RenderPlugin::default());

		#[cfg(not(target_os = "android"))]
		app.add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default(), DiagnosticsPlugin));
	}
}

fn runner(mut app: App) -> AppExit {
	let mut count = 0_u64;
	loop {
		//println!("Running: {count}");

		app.update();

		count += 1;
	}

	AppExit::Success
}

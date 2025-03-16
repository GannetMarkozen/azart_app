mod azart;

use azart::*;

use bevy::prelude::*;

fn main() {
	#[cfg(debug_assertions)]
	{
		unsafe { std::env::set_var("RUST_BACKTRACE", "full"); }
		color_eyre::install().unwrap();
	}

	App::new()
		.add_plugins(AzartPlugin)
		.add_systems(Startup, load_flight_helmet)
		.run();
}

fn load_flight_helmet(
	mut commands: Commands,
	asset_server: Res<AssetServer>,
) {
}
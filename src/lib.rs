use azart::*;
use bevy::prelude::*;

#[bevy_main]
fn main() {
	#[cfg(debug_assertions)]
	{
		unsafe { std::env::set_var("RUST_BACKTRACE", "full"); }
		color_eyre::install().unwrap();
	}

	App::new()
		.add_plugins(AzartPlugin)
		.run();
}

use std::path::Path;
use std::env;

fn main() {
	return;
	println!("cargo:rerun-if-changed=build.rs");
	println!("cargo:rerun-if-changed=assets");

	color_eyre::install().unwrap();
	unsafe { env::set_var("RUST_BACKTRACE", "full"); }

	let assets_path = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("assets");

	let assets_dst = Path::new(&env::var("OUT_DIR").unwrap())
		.parent().unwrap()
		.parent().unwrap()
		.parent().unwrap()
		.join("assets");

	for entry in walkdir::WalkDir::new(&assets_path).into_iter().filter_map(|e| e.ok()) {
		if !entry.file_type().is_file() {
			continue;
		}

		let new_path = assets_dst
			.join(entry
				.path()
				.strip_prefix(&assets_path)
				.unwrap()
			);

		if let Some(dir) = new_path.parent() {
			std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("Failed to create dir {dir:?}: {e}"));
		}
		
		std::fs::copy(entry.path(), &new_path).unwrap_or_else(|e| panic!("Failed to copy data to dir {new_path:?}: {e}"));
		println!("Copied contents of {entry:?} to {new_path:?}!");
	}
	panic!("yeah, that just happened.");

	// Package assets with build.

	// Nothing special required if not packaging for android.
	/*if !matches!(std::env::var("TARGET"), Ok(target) if target.contains("android")) {
		return;
	};

	let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
	let assets_dst = Path::new(&manifest_dir);*/

	//panic!("asset_dst: {assets_dst:?}");
}
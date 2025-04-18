use std::path::Path;
use std::{env, fs};
use std::io::Read;

fn main() {
	println!("cargo:rerun-if-changed=build.rs");
	println!("cargo:rerun-if-changed=assets");

	color_eyre::install().unwrap();
	unsafe { env::set_var("RUST_BACKTRACE", "full"); }

	// This is not necessary for Android. Asset packaging is handled by XBuild.
	if matches!(env::var("TARGET"), Ok(target) if target.contains("android")) {
		return;
	}

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
			fs::create_dir_all(dir).unwrap_or_else(|e| panic!("Failed to create dir {dir:?}: {e}"));
		}
		
		fs::copy(entry.path(), &new_path).unwrap_or_else(|e| panic!("Failed to copy data to dir {new_path:?}: {e}"));
		println!("Copied contents of {entry:?} to {new_path:?}!");
	}
}

fn download_openxr_loader() {
	println!("cargo::rerun-if-changed=build.rs");

	const XR_VER: &str = "1.1.38";

	let platform = env::var("CARGO_CFG_TARGET_OS").unwrap();
	let (arch, ext) = match platform.as_str() {
		"windows" => ("x86_64", "zip"),
		"android" => ("arm64-v8a", "aar"),
		_ => panic!("Unhandled platform {platform}"),
	};

	println!("platform: {platform}");

	let mut resp = reqwest::blocking::get(
		format!("https://github.com/KhronosGroup/OpenXR-SDK-Source/releases/download/release-{XR_VER}/openxr_loader_for_{platform}-{XR_VER}.{ext}")
	).unwrap();

	let mut file = vec![];
	resp.read_to_end(&mut file).unwrap();

	//let dst_path = Path::new("runtime_libs/openxr_loader").join("openxr_loader_android.arr");
	let dst_path = Path::new("runtime_libs/openxr_loader").join(format!("openxr_loader_{platform}.{ext}"));
	fs::write(&dst_path, file).unwrap();
	let file = fs::File::open(&dst_path).unwrap();

	let mut zip_file = zip::ZipArchive::new(file).unwrap();
	eprintln!("{:#?}", zip_file.file_names().collect::<Vec<_>>());
	let mut loader_file = zip_file
		//.by_name("prefab/modules/openxr_loader/libs/android.arm64-v8a/libopenxr_loader.so")
		.by_name(format!("prefab/modules/openxr_loader/libs/{platform}.{arch}/libopenxr_loader.so").as_str())
		.unwrap();

	let mut file = Vec::new();
	loader_file.read_to_end(&mut file).unwrap();
	let out_dir = env::var_os("CARGO_MANIFEST_DIR").unwrap();
	let dest_path = Path::new(&out_dir).join(format!("./runtime_libs/{arch}/libopenxr_loader.so"));
	let _ = fs::create_dir_all(dest_path.parent().unwrap());
	fs::write(&dest_path, file).unwrap();
}
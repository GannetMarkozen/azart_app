use std::ffi::CString;
use std::path::Path;

// Platform agnostic io operations.

#[cfg(not(target_os = "android"))]
#[inline]
pub fn read(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
	std::fs::read(path.as_ref())
}

#[cfg(target_os = "android")]
pub fn read(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
	let asset_manager = bevy::window::ANDROID_APP
		.get()
		.expect("Failed to get android app! Must be called after #[bevy_main]!")
		.asset_manager();

	let Some(mut opened_asset) = asset_manager.open(&CString::new(path.as_ref().to_str().unwrap()).unwrap()) else {
		return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "File not found!"));
	};

	Ok(opened_asset.buffer()?.to_vec())
}
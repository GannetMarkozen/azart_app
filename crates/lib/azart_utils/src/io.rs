use std::path::{Path, PathBuf};

/// Platform agnostic file reading. On Android this will attempt to read from the internal data file
/// path first and if that fails then it will attempt to read that file path from the asset manager.
#[cfg(not(target_os = "android"))]
#[inline]
pub fn read(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
	std::fs::read(path)
}

#[cfg_attr(doc, doc(cfg(target_os = "android")))]
#[cfg(target_os = "android")]
pub fn read(path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
  use std::io::{Error, ErrorKind};
  use std::ffi::CString;

use bevy::log::warn;

  let android_app = bevy::window::ANDROID_APP
    .get()
    .expect("Failed to get android app! Must be called after #[bevy_main]!");

  // First attempt to read from the data file path. Then try reading from the asset manager.
  let file_path = android_app
    .external_data_path()
    .expect("Failed get android internal data path!")
    .join(path.as_ref());

  let file_err = match std::fs::read(&file_path) {
    Ok(data) => return Ok(data),
    Err(e) => e,
  };

  // Next attempt to read from the asset manager.
	let mut opened_asset = android_app
    .asset_manager()
    .open(&CString::new(path.as_ref().to_str().unwrap()).unwrap())
    .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("File not found in external data path or asset manager!: {file_err:?}")))?;

	Ok(opened_asset.buffer()?.to_vec())
}

/// Writes contents to a file. Overwriting existing content. Creates the required directories if applicable.
#[cfg(not(target_os = "android"))]
#[inline]
pub fn write(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> std::io::Result<()> {
  let path = path.as_ref();
  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent)?;
  }

  std::fs::write(path, data)
}

#[cfg_attr(doc, doc(cfg(target_os = "android")))]
#[cfg(target_os = "android")]
pub fn write(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> std::io::Result<()> {
  use std::io::Error;

  let path = bevy::window::ANDROID_APP
    .get()
    .expect("Failed to get android app! Must be called after #[bevy_main]!")
    .external_data_path()
    .expect("Failed to get android external data path!")
    .join(path.as_ref());

  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent)?;
  }

  std::fs::write(&path, data)
}

#[cfg(not(target_os = "android"))]
pub fn data_path() -> &'static Path {
  Path::new("")
}

#[cfg_attr(doc, doc(cfg(target_os = "android")))]
#[cfg(target_os = "android")]
pub fn data_path() -> PathBuf {
  bevy::window::ANDROID_APP
    .get()
    .expect("Failed to get android app! Must be called after #[bevy_main]!")
    .external_data_path()
    .expect("Failed to get android external data path!")
}

#[cfg(not(target_os = "android"))]
pub fn read_dir(path: impl AsRef<Path>) -> std::io::Result<Vec<PathBuf>> {
  let mut entries = Vec::new();
  for entry in std::fs::read_dir(path)? {
    entries.push(entry?.path());
  }

  Ok(entries)
}

#[cfg_attr(doc, doc(cfg(target_os = "android")))]
#[cfg(target_os = "android")]
pub fn read_dir(path: impl AsRef<Path>) -> std::io::Result<Vec<PathBuf>> {
  use std::{io::{Error, ErrorKind}, ffi::{CStr, CString, OsStr}};

  let path = path.as_ref();
  let file_err = match std::fs::read_dir(data_path().join(path)) {
    Ok(dir) => {
      let mut entries = Vec::new();
      for entry in dir {
        entries.push(entry?.path());
      }
      return Ok(entries);
    },
    Err(e) => e,
  };

  let entries = bevy::window::ANDROID_APP
    .get()
    .expect("Failed to get android app! Must be called after #[bevy_main]!")
    .asset_manager()
    .open_dir(&CString::new(path.to_str().unwrap()).unwrap())
    .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Directory not found in external data path or asset manager!: {file_err:?}")))?
    .map(|inner| path.join(Path::new(inner.to_str().unwrap())))
    .collect::<Vec<_>>();

  Ok(entries)
}

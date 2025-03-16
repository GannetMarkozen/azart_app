use std::fs;
use std::process::Command;
use std::path::Path;

const SHADER_DIR: &str = "shaders";
const SPV_DIR: &str = "spv";

// TODO: Only recompile changed shader files.
fn main() {
	color_eyre::install().unwrap();

	println!("cargo:rerun-if-changed={}", SHADER_DIR);

	if !Path::new(SHADER_DIR).exists() {
		println!("cargo:warning=Shader directory \"{SHADER_DIR}\" does not exist!");
		return;
	}

	let compiled_count = fs::read_dir(SHADER_DIR)
		.unwrap()
		.filter_map(|x| {
			let Ok(entry) = x else {
				return None;
			};

			let path = entry.path();
			let Some(ext) = path.extension() else {
				println!("cargo:warning=No extension for file: \"{}\"", path.display());
				return None;
			};

			if !matches!(ext.to_str().unwrap(), "vert" | "frag" | "comp" | "geom" | "tese" | "tesc" | "mesh") {
				println!("cargo:warning=Unsupported shader extension: \"{ext:?}\" for {path:?}!");
				return None;
			}

			Some(path)
		})
		.map(|path| compile_shader(&path) as usize)
		.sum::<usize>();

	println!("Successfully compiled {compiled_count} shaders!");
}

fn compile_shader(path: &Path) -> bool {
	let shader_name = path.file_name().unwrap().to_str().unwrap();

	let shader_path = path.to_str().unwrap();
	let spv_path = shader_path.replace(SHADER_DIR, SPV_DIR) + ".spv";
	
	println!("Compiling {shader_name}...");

	let output = Command::new("glslangValidator")
		.arg("-V")
		.arg(shader_path)
		.arg("-o")
		.arg(spv_path)
		.output()
		.expect("Failed to execute glslangValidator");

	let success = output.status.success();
	if success {
		println!("Compiled {shader_name}! Log: {}", String::from_utf8_lossy(&output.stdout));
	} else {
		panic!("Failed to compile {shader_name}!\nError: {}", String::from_utf8_lossy(&output.stdout));
	}
	
	success
}
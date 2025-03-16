use ash::vk;

pub fn format_size(format: vk::Format) -> usize {
	match format {
		vk::Format::R8_UNORM | vk::Format::R8_SNORM | vk::Format::R8_UINT | vk::Format::R8_SINT => 1,
		vk::Format::R8G8_UNORM | vk::Format::R8G8_SNORM | vk::Format::R8G8_UINT | vk::Format::R8G8_SINT => 2,
		vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SNORM | vk::Format::R8G8B8_UINT | vk::Format::R8G8B8_SINT => 3,
		vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SNORM | vk::Format::R8G8B8A8_UINT | vk::Format::R8G8B8A8_SINT => 4,
		vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => 4,
		vk::Format::R16_UNORM | vk::Format::R16_SNORM | vk::Format::R16_UINT | vk::Format::R16_SINT | vk::Format::R16_SFLOAT => 2,
		vk::Format::R16G16_UNORM | vk::Format::R16G16_SNORM | vk::Format::R16G16_UINT | vk::Format::R16G16_SINT | vk::Format::R16G16_SFLOAT => 4,
		vk::Format::R16G16B16A16_UNORM | vk::Format::R16G16B16A16_SNORM | vk::Format::R16G16B16A16_UINT | vk::Format::R16G16B16A16_SINT | vk::Format::R16G16B16A16_SFLOAT => 8,
		vk::Format::R32_UINT | vk::Format::R32_SINT | vk::Format::R32_SFLOAT => 4,
		vk::Format::R32G32_UINT | vk::Format::R32G32_SINT | vk::Format::R32G32_SFLOAT => 8,
		vk::Format::R32G32B32_UINT | vk::Format::R32G32B32_SINT | vk::Format::R32G32B32_SFLOAT => 12,
		vk::Format::R32G32B32A32_UINT | vk::Format::R32G32B32A32_SINT | vk::Format::R32G32B32A32_SFLOAT => 16,
		_ => panic!("Unsupported format {format:?}!"),
	}
}
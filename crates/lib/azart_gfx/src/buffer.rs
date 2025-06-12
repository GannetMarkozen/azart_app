use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use ash::vk;
use azart_asset::{bincode, AssetHandler};
use gpu_allocator::vulkan::{Allocation, AllocationScheme};
use crate::GpuContext;
use azart_utils::debug_string::DebugString;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use azart_gfx_utils::GpuResource;
use derivative::Derivative;
use serde::{Deserialize, Deserializer, Serialize};
use crate::render::plugin::align_to;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Buffer {
	pub(crate) name: DebugString,
	pub(crate) handle: vk::Buffer,
	#[derivative(Debug = "ignore")]
	pub(crate) allocation: ManuallyDrop<Allocation>,
	size: usize,
	#[derivative(Debug = "ignore")]
	pub(crate) cx: Arc<GpuContext>,
}

impl Buffer {
	pub fn new(
		name: DebugString,
		context: Arc<GpuContext>,
		create_info: &BufferCreateInfo,
	) -> Self {
		assert_ne!(create_info.size, 0);
		assert!(!create_info.usage.is_empty());
		
		let buffer = {
			let create_info = vk::BufferCreateInfo::default()
				.usage(create_info.usage)
				.sharing_mode(vk::SharingMode::EXCLUSIVE)
				.size(create_info.size as u64);
				
			unsafe { context.device.create_buffer(&create_info, None) }.unwrap()
		};

		let allocation = {
			let (requirements, dedicated_allocation) = {
				let info = vk::BufferMemoryRequirementsInfo2::default()
					.buffer(buffer);

				let mut requirements = vk::MemoryRequirements2::default();
				let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
				requirements.push_next(&mut dedicated_requirements);
				
				unsafe { context.device.get_buffer_memory_requirements2(&info, &mut requirements); }
				(requirements.memory_requirements, dedicated_requirements.prefers_dedicated_allocation == vk::TRUE)
			};
			
			let allocation_scheme = match dedicated_allocation {
				true => AllocationScheme::DedicatedBuffer(buffer),
				false => AllocationScheme::GpuAllocatorManaged,
			};

			let create_info = AllocationCreateDesc {
				name: name.as_str(),
				requirements,
				location: create_info.memory,
				linear: true,
				allocation_scheme,
			};

			context.alloc(&create_info)
		};

		unsafe { context.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.expect("Failed to bind buffer memory!");

		#[cfg(debug_assertions)]
		unsafe {
			context.set_debug_name(name.as_str(), buffer);
		}

		Self {
			name,
			handle: buffer,
			allocation: ManuallyDrop::new(allocation),
			size: create_info.size,
			cx: context,
		}
	}

	#[inline(always)]
	pub fn size(&self) -> usize {
		self.size
	}
	
	#[inline(always)]
	pub fn name(&self) -> &DebugString {
		&self.name
	}

	/// Returns None if this Buffer was not created with the memory location CpuToGpu.
	#[inline(always)]
	pub fn mapping(&self) -> Option<&[u8]> {
		self
			.allocation
			.mapped_slice()
			.map(|data| &data[..self.size])
	}

	#[inline(always)]
	pub fn mapping_mut(&mut self) -> Option<&mut [u8]> {
		self
			.allocation
			.mapped_slice_mut()
			.map(|data| &mut data[..self.size])
	}
}

impl Drop for Buffer {
	fn drop(&mut self) {
		unsafe {
			self.cx.dealloc(ManuallyDrop::take(&mut self.allocation));
			self.cx.device.destroy_buffer(self.handle, None);
		}
	}
}

impl GpuResource for Buffer {}

pub struct BufferCreateInfo {
	pub size: usize,
	pub usage: vk::BufferUsageFlags,
	pub memory: MemoryLocation,
}

impl BufferCreateInfo {
	#[inline(always)]
	pub const fn storage_buffer<T>(len: usize) -> Self {
		Self {
			size: len * size_of::<T>(),
			usage: vk::BufferUsageFlags::STORAGE_BUFFER,
			memory: MemoryLocation::GpuOnly,
		}
	}
}

impl Default for BufferCreateInfo {
	fn default() -> Self {
		Self {
			size: 0,
			usage: vk::BufferUsageFlags::empty(),
			memory: MemoryLocation::GpuOnly,
		}
	}
}

#[derive(Serialize, Deserialize)]
pub struct SerdeBuffer<'a> {
	name: &'a str,
	data: &'a [u8],
}

pub struct BufferAssetHandler {
	cx: Arc<GpuContext>,
}

impl BufferAssetHandler {
	#[inline]
	pub const fn new(cx: Arc<GpuContext>) -> Self {
		Self { cx }
	}
}

impl AssetHandler for BufferAssetHandler {
	type Target = Buffer;

	fn load(&self, data: &[u8]) -> std::io::Result<Self::Target> {
		use std::io::Error;

		let SerdeBuffer { name, data: src } = bincode::serde::borrow_decode_from_slice(data, bincode::config::standard()).map_err(Error::other)?.0;

		let buffer = Buffer::new(
			name.to_owned().into(),
			Arc::clone(&self.cx),
			&BufferCreateInfo {
				size: data.len(),
				..Default::default()
			}
		);

		self
			.cx
			.upload_buffer(
				&buffer,
				|dst| dst.copy_from_slice(src),
			);

		Ok(buffer)
	}

	fn store(&self, value: &Self::Target) -> std::io::Result<Vec<u8>> {
		unimplemented!();
	}
}
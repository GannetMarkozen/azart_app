use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use ash::vk;

use crate::{buffer::{Buffer, BufferCreateInfo}, GpuContext, image::Image};
use azart_asset::{bincode, AnyAsset, AssetHandler, DefaultAssetHandler, SerdeAsset};
use bevy::{math::Vec3, reflect::Reflect};
use serde::{Serialize, Deserialize};
use azart_asset::Asset;

pub struct Mesh {
  pub material: Asset<Material>,
  pub indices: Buffer,
  pub positions: Buffer,
  pub attributes: Buffer,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Material {
  pub base_color: Asset<Image>,
  pub normals: Option<Asset<Image>>,
  pub metallic_roughness: Option<Asset<Image>>,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct VertexAttributes {
  pub uv: [f32; 2],
  pub normal: [f32; 3],
}

pub(crate) struct MeshAssetHandler {
  cx: Arc<GpuContext>,
}

#[derive(Serialize, Deserialize)]
pub struct SerdeMesh {
  pub material: Asset<Material>,
  pub indices: Vec<u32>,
  pub positions: Vec<Vec3>,
  pub attributes: Vec<VertexAttributes>,
}

impl MeshAssetHandler {
  #[inline]
  pub(crate) fn new(cx: Arc<GpuContext>) -> Self {
    Self { cx }
  }
}

impl AssetHandler for MeshAssetHandler {
  type Target = Mesh;

  fn load(&self, data: &[u8]) -> std::io::Result<Self::Target> {
    use std::io::Error;

    let SerdeMesh { material, indices,  positions, attributes } = bincode::serde::borrow_decode_from_slice(data, bincode::config::standard()).map_err(Error::other)?.0;

    #[inline(always)]
    const fn alloc_size_for_slice<T: Sized>(slice: &[T]) -> usize {
      slice.len() * size_of::<T>()
    }

    let indices_buffer = Buffer::new(
      "indices".into(),
      Arc::clone(&self.cx),
      &BufferCreateInfo {
        size: alloc_size_for_slice(indices.as_slice()),
        usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ..Default::default()
      },
    );

    let positions_buffer = Buffer::new(
      "positions".into(),
      Arc::clone(&self.cx),
      &BufferCreateInfo {
        size: alloc_size_for_slice(positions.as_slice()),
        usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ..Default::default()
      },
    );

    let attributes_buffer = Buffer::new(
      "attributes".into(),
      Arc::clone(&self.cx),
      &BufferCreateInfo {
        size: alloc_size_for_slice(attributes.as_slice()),
        usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ..Default::default()
      }
    );

    self
      .cx
      .upload_buffer(
        &indices_buffer,
        |dst| unsafe {
          copy_nonoverlapping(indices.as_ptr() as *const u8, dst.as_mut_ptr(), dst.len());
        }
      );

    self
      .cx
      .upload_buffer(
        &positions_buffer,
        |dst| unsafe {
          copy_nonoverlapping(positions.as_ptr() as *const u8, dst.as_mut_ptr(), dst.len());
        }
      );

    self
      .cx
      .upload_buffer(
        &attributes_buffer,
        |dst| unsafe {
          copy_nonoverlapping(positions.as_ptr() as *const u8, dst.as_mut_ptr(), attributes.len());
        }
      );

    Ok(Mesh {
      material,
      indices: indices_buffer,
      positions: positions_buffer,
      attributes: attributes_buffer,
    })
  }

  fn store(&self, value: &Self::Target) -> std::io::Result<Vec<u8>> {
    unimplemented!();
  }
} 
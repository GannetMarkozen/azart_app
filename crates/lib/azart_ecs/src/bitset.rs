use smallvec::SmallVec;

pub struct BitSet {
	level0: u64,
	level1: SmallVec<[u64; 2]>,
	level2: SmallVec<[u64; 2]>,
}

impl BitSet {
	pub const LAYERS: usize = 3;
	pub const MAX: usize = 64 * 64 * 64;

	#[inline]
	pub const fn new() -> Self {
		Self {
			level0: 0,
			level1: SmallVec::new_const(),
			level2: SmallVec::new_const(),
		}
	}

	pub fn add(&mut self, index: usize) -> bool {
		assert!(index < Self::MAX, "Index ({index}) exceeds maximum range: {}", Self::MAX);

		let mut added = false;
		
		

		added
	}

	#[inline]
	const fn interval(layer: usize) -> usize {
		assert!(layer < Self::LAYERS);
		64_usize.pow((Self::LAYERS - layer) as u32)
	}
}

impl Default for BitSet {
	#[inline]
	fn default() -> Self {
		Self::new()
	}
}
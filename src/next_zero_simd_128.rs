use std::{
    arch::x86_64::{__m128i, _mm_cmpestri, _mm_setzero_si128, _SIDD_CMP_EQUAL_ORDERED},
    simd::u8x16,
};

use crate::block_iter::NextZeroIndex;

#[derive(Default)]
pub struct SimdBlocks16 {}

impl NextZeroIndex for SimdBlocks16 {
    fn next_zero_index(data: &[u8]) -> Option<usize> {
        let mut nonzero_bytes = 0;

        //for block in AlignedIter::new(data, 16) { // worse performance :(
        for block in data.chunks(16) {
            if block.len() != 16 {
                for b in block {
                    if *b == 0 {
                        return Some(nonzero_bytes);
                    } else {
                        nonzero_bytes += 1;
                    }
                }
                continue;
            }

            let v = u8x16::from_slice(block);
            let res = unsafe {
                _mm_cmpestri(
                    _mm_setzero_si128(),
                    1,
                    __m128i::from(v),
                    16,
                    _SIDD_CMP_EQUAL_ORDERED,
                )
            };
            nonzero_bytes += res as usize;
            if res < 16 {
                return Some(nonzero_bytes);
            }
        }

        None
    }
}

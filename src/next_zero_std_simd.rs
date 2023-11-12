use crate::block_iter::NextZeroIndex;

use num::PrimInt;
use std::simd::prelude::*;
use std::simd::LaneCount;
use std::simd::SupportedLaneCount;
use std::simd::ToBitMask;

#[derive(Default)]
pub struct SimdBlocksGeneric<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
    Mask<i8, N>: ToBitMask,
    <Mask<i8, N> as ToBitMask>::BitMask: PrimInt, {}

impl<const N: usize> NextZeroIndex for SimdBlocksGeneric<N>
where
    LaneCount<N>: SupportedLaneCount,
    Mask<i8, N>: ToBitMask,
    <Mask<i8, N> as ToBitMask>::BitMask: PrimInt,
{
    fn next_zero_index(data: &[u8]) -> Option<usize> {
        let mut nonzero_bytes = 0;
        let mut chunks_iter = data.array_chunks::<N>();

        for block in chunks_iter.by_ref() {
            let index = first_zero_in_vector::<N>(Simd::from(*block));
            nonzero_bytes += index;
            if (index as usize) < N {
                return Some(nonzero_bytes as usize);
            }
        }

        for b in chunks_iter.remainder() {
            if *b == 0 {
                return Some(nonzero_bytes as usize);
            } else {
                nonzero_bytes += 1;
            }
        }
        None
    }
}

// 256b vector: 32xu8
pub fn first_zero_in_vector<const N: usize>(block: Simd<u8, N>) -> u32
where
    LaneCount<N>: SupportedLaneCount,
    Mask<i8, N>: ToBitMask,
    <Mask<i8, N> as ToBitMask>::BitMask: PrimInt,
{
    // Equality check creates mask, mask to int ?u256???, clz instruction
    let mask = block.simd_eq(Simd::<u8, N>::splat(0u8));
    let bitmask = mask.to_bitmask();
    bitmask.trailing_zeros()
}

#[cfg(test)]
mod tests {
    use std::simd::{Simd, SimdPartialEq, ToBitMask};

    #[test]
    fn bitmask_assumptions() {
        let input_vec: Simd<u8, 32> = Simd::from([
            50, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 50,
        ]);
        let search = Simd::splat(50);
        let result_mask = input_vec.simd_eq(search);
        println!("{:?}", result_mask);
        // Bitmask: index starting at lowest bit
        let result_bitmask: u32 = result_mask.to_bitmask();
        println!("{:#032b}", result_bitmask);
        assert!(result_bitmask & (1 << 7) != 0);
        assert!(result_bitmask & (1 << 0) != 0);
        assert!(result_bitmask & (1 << 31) != 0);
    }

    #[test]
    fn trailing_zeros_assumptions() {
        let input: u32 = 0b10000;
        assert_eq!(input.trailing_zeros(), 4);
        let zero: u32 = 0;
        assert_eq!(zero.trailing_zeros(), 32);
    }
}

use std::arch::x86_64::*;
use std::cmp::min;
use std::simd::u8x16;

#[allow(unused)]
use crate::aligned_iter::AlignedIter;

pub struct BlockIter<'a> {
    input_data: &'a [u8], // BlockIter can't outlive input data
    processed: usize,     // Number of bytes already processed
    max_block_size: usize,
}

impl<'a> BlockIter<'a> {
    pub fn new(input_data: &[u8], max_block_size: usize) -> BlockIter {
        BlockIter {
            input_data,
            processed: 0,
            max_block_size,
        }
    }
}

#[allow(unused)]
fn next_zero_index(data: &[u8]) -> Option<usize> {
    data.iter().position(|x| *x == 0)
}

fn next_zero_index_simd(data: &[u8]) -> Option<usize> {
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

        // TODO: 256bit vectors with _mm256_cmpeq_epi8 and _mm256_movemask_epi8 and _lzcnt_u32

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

impl<'a> Iterator for BlockIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.processed == self.input_data.len() + 1 {
            // Process all bytes plus appended zero
            return None;
        }

        if self.processed == self.input_data.len() {
            // Only the appended zero remaining, without data
            self.processed += 1;
            return Some(&[]);
        }

        let start_index = self.processed;

        let upper_bound = min(start_index + self.max_block_size, self.input_data.len());

        //let zero_index = next_zero_index(&self.input_data[start_index..upper_bound]);
        let zero_index = next_zero_index_simd(&self.input_data[start_index..upper_bound]);
        match zero_index {
            Some(i) => {
                // Process data inclusive zero at i
                self.processed += i + 1;

                // Return data without zero
                Some(&self.input_data[start_index..start_index + i])
            }
            None => {
                if upper_bound == self.input_data.len() {
                    // No zeros until end, pretend there is one appended
                    self.processed = self.input_data.len() + 1;
                    Some(&self.input_data[start_index..self.input_data.len()])
                } else {
                    // Block without zeros of length self.max_block_size, but not at the end
                    self.processed += self.max_block_size;
                    Some(&self.input_data[start_index..start_index + self.max_block_size])
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::{arch::x86_64::*, simd::*};

    #[test]
    fn iter() {
        let data = [1, 2, 3, 4, 0, 1, 2, 3];
        let blocks: Vec<_> = BlockIter::new(&data, 254).collect();

        assert_eq!(blocks[0], &[1, 2, 3, 4]);
        assert_eq!(blocks[1], &[1, 2, 3]);
    }

    use super::BlockIter;
    #[test]
    fn simd() {
        let mut data = vec![27_u8; 1000];
        data[15] = 0;
        let v = u8x16::from_slice(&data[0..16]);
        let res = unsafe {
            _mm_cmpestri(
                _mm_setzero_si128(),
                1,
                __m128i::from(v),
                16,
                _SIDD_CMP_EQUAL_ORDERED,
            )
        };
        dbg!(res);
    }

    #[quickcheck]
    fn max_block_size(input_data: Vec<u8>) -> bool {
        for b in BlockIter::new(&input_data, 254) {
            if b.len() > 254 {
                return false;
            }
        }
        true
    }

    #[quickcheck]
    fn blocks_dont_contain_zero_qc(input_data: Vec<u8>) -> bool {
        blocks_dont_contain_zero(input_data)
    }

    fn blocks_dont_contain_zero(input_data: Vec<u8>) -> bool {
        for block in BlockIter::new(&input_data, 254) {
            if block.len() > 1 {
                // Blocks of length 1 only contain a zero
                for byte in block.iter().take(block.len() - 1) {
                    // All bytes but the last must be non-zero
                    if *byte == 0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn blocks_dont_contain_zero_special() {
        let mut special_input = vec![3_u8; 254];
        special_input[253] = 0;
        assert!(blocks_dont_contain_zero(special_input));
    }
}

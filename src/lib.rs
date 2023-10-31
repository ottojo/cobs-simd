#![feature(portable_simd)]

#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use std::iter::once;

use crate::block_iter::BlockIter;
mod aligned_iter;
mod block_iter;

pub fn encoded_size_upper_bound(input_size: usize) -> usize {
    // max ⌈n/254⌉ overhead
    input_size + (input_size + 254 - 1) / 254
}

// Returns number of bytes written
pub fn cobs_encode_to(input: &[u8], output: &mut [u8]) -> usize {
    let mut written = 0;
    let mut current_block_length: u8 = 0;

    for b in input.iter().cloned().chain(once(0)) {
        if current_block_length == 0 {
            written += 1; // overhead for next group
        }

        if b == 0 {
            // End of group
            let overhead_byte_index = written - 1 - current_block_length as usize;
            output[overhead_byte_index] = current_block_length + 1;
            current_block_length = 0;
            continue;
        }

        output[written] = b;
        written += 1;
        current_block_length += 1;

        if current_block_length == 254 {
            // End of group of 254 non-zero bytes
            let overhead_byte_index = written - 1 - current_block_length as usize;
            output[overhead_byte_index] = 255;
            current_block_length = 0;
        }
    }

    written
}

pub fn cobs_encode_to_opt(input: &[u8], output: &mut [u8]) -> usize {
    let mut out_idx = 0;
    for block in BlockIter::new(input, 254) {
        output[out_idx] = block.len() as u8 + 1;
        // Copy all
        output[out_idx + 1..out_idx + 1 + block.len()].copy_from_slice(block);
        out_idx += block.len() + 1;
    }

    out_idx
}

pub fn cobs_encode_to_chained_iter(input: &[u8], output: &mut [u8]) -> usize {
    let mut out_idx = 0;
    // This finds large non-zero blocks first, and then divides them, instead of directly finding non-zero blocks with maximum size
    for large_block in BlockIter::new(input, input.len()) {
        // Manual flat_map, since chunking empty slice does not yield an empty slice, but we want to preserve it...
        if !large_block.is_empty() {
            for block in large_block.chunks(254) {
                output[out_idx] = block.len() as u8 + 1;
                // Copy all
                output[out_idx + 1..out_idx + 1 + block.len()].copy_from_slice(block);
                out_idx += block.len() + 1;
            }
        } else {
            output[out_idx] = large_block.len() as u8 + 1;
            // Copy all
            output[out_idx + 1..out_idx + 1 + large_block.len()].copy_from_slice(large_block);
            out_idx += large_block.len() + 1;
        }
    }

    out_idx
}

#[allow(unused)]
pub fn cobs_encode(input: &[u8]) -> Vec<u8> {
    let mut res = vec![];

    let mut current_block_length: u8 = 0;

    for b in input.iter().cloned().chain(once(0)) {
        if current_block_length == 0 {
            res.push(0) // overhead for next group
        }

        if b == 0 {
            // End of group
            let overhead_byte_index = res.len() - 1 - current_block_length as usize;
            res[overhead_byte_index] = current_block_length + 1;
            current_block_length = 0;
            continue;
        }

        res.push(b);
        current_block_length += 1;

        if current_block_length == 254 {
            // End of group of 254 non-zero bytes
            let overhead_byte_index = res.len() - 1 - current_block_length as usize;
            res[overhead_byte_index] = 255;
            current_block_length = 0;
        }
    }

    res
}

pub fn encode_to_wrapper(function: fn(&[u8], &mut [u8]) -> usize) -> Box<dyn Fn(&[u8]) -> Vec<u8>> {
    Box::new(move |input: &[u8]| {
        let mut output_data = vec![0; encoded_size_upper_bound(input.len())];
        let s = function(input, &mut output_data);
        output_data.truncate(s);
        output_data
    })
}

#[allow(unused)]
pub fn cobs_decode(input: &[u8]) -> Vec<u8> {
    let mut res = vec![];

    let mut current_group_length = 0;

    let mut it = input.iter().cloned();
    while let Some(overhead_byte) = it.next() {
        for i in 0..overhead_byte - 1 {
            // TODO: extend(take) without loop?
            res.push(it.next().unwrap())
        }
        if overhead_byte != 255 && it.len() > 0 {
            res.push(0)
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use crate::{
        cobs_decode, cobs_encode, cobs_encode_to, cobs_encode_to_chained_iter, cobs_encode_to_opt,
        encode_to_wrapper,
    };

    use concat_idents::concat_idents;

    macro_rules! encode_tests {
        ($name:ident, $func:expr) => {

            concat_idents!(test_name = $name, _, encoding_1_one_zero {
                #[test]
                fn test_name() {
                    assert_eq!( $func (&[0]), vec![1, 1])
                }
            });

            concat_idents!(test_name = $name, _, encoding_2_only_zeros_short {
                #[test]
                fn test_name() {
                    assert_eq!($func(&[0, 0]), vec![1, 1, 1])
                }
            });

            concat_idents!(test_name = $name, _, encoding_3 {
                #[test]
                fn test_name() {
                    assert_eq!($func(&[0, 0x11, 0]), vec![0x01, 0x02, 0x11, 0x01])
                }
            });

            concat_idents!(test_name = $name, _, encoding_4_zeros_short {
                #[test]
                fn test_name() {
                    assert_eq!(
                        $func(&[0x11, 0x22, 0x00, 0x33]),
                        vec![0x03, 0x11, 0x22, 0x02, 0x33]
                    )
                }
            });

            concat_idents!(test_name = $name, _, encoding_5_no_zeros_short {
                #[test]
                fn test_name() {
                    assert_eq!(
                        $func(&[0x11, 0x22, 0x33, 0x44]),
                        vec![0x05, 0x11, 0x22, 0x33, 0x44]
                    )
                }
            });

            concat_idents!(test_name = $name, _, encoding_9_no_zeros_long {
                #[test]
                fn test_name() {
                    let input: Vec<_> = (0x01..=0xFF_u8).collect();

                    let mut expected_output = vec![0xFF];
                    expected_output.extend(0x01..=0xFE);
                    expected_output.extend([0x02, 0xFF]);

                    assert_eq!($func(&input), expected_output)
                }
            });
        };
    }

    encode_tests!(default, cobs_encode);

    encode_tests!(to_buffer, encode_to_wrapper(cobs_encode_to));

    encode_tests!(to_buffer_opt, encode_to_wrapper(cobs_encode_to_opt));

    encode_tests!(chained_iter, encode_to_wrapper(cobs_encode_to_chained_iter));

    #[test]
    fn decoding_no_zeros_short() {
        assert_eq!(
            cobs_decode(&[0x05, 0x11, 0x22, 0x33, 0x44]),
            vec![0x11, 0x22, 0x33, 0x44]
        )
    }
}

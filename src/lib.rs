#![feature(portable_simd)]
#![feature(array_chunks)]

#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use std::iter::once;

use block_iter::NextZeroIndex;
use next_zero_simd_128::SimdBlocks16;
use next_zero_std_simd::SimdBlocksGeneric;

use strum_macros::{Display, EnumIter};

use crate::block_iter::BlockIter;
mod aligned_iter;
mod block_iter;
mod next_zero_simd_128;
mod next_zero_std_simd;

/// Determines the upper bound of the encoded message size depending on the input length
///
/// COBS induces a maximum of ⌈n/254⌉ bytes overhead for n data bytes.
pub fn encoded_size_upper_bound(input_size: usize) -> usize {
    input_size + (input_size + 254 - 1) / 254
}

/// Encoding method
///
/// These are different methods for COBS encoding.
/// They all produce the same output, but have different runtime characteristics.
#[derive(Clone, Display, EnumIter)]
pub enum Method {
    /// Simple loop, sequentially processing every byte without (explicitly) using SIMD instructions.
    Trivial,
    /// Direct translation of unhinged C implementation from wikipedia
    Crazy,
    /// Optimized version which uses an iterator producing blocks that internally uses SIMD intrinsics for finding zeros in the data.
    Simd16,
    /// Versions that use std::Simd operations to be generic over vector length
    StdSimd8,
    StdSimd16,
    StdSimd32,
    /// Versions that use std::Simd operations to be generic over vector length and separate the zero-finding and splitting of large blocks, which may yield a small performance benefit
    StdSimd8TwoStage,
    StdSimd16TwoStage,
    StdSimd32TwoStage,
}

/// COBS-encode data to a buffer.
///
/// User must ensure that the buffer is big enough. Actual buffer usage depends on input, but always fits within encoded_size_upper_bound(input.len()).
///
/// # Example
///
/// ```
/// use cobs_simd::{cobs_encode_to, encoded_size_upper_bound, Method};
///
/// let input_data = [1, 3, 0, 7, 0, 8];
/// let mut encoded_output = vec![0; encoded_size_upper_bound(input_data.len())];
/// let output_length = cobs_encode_to(&input_data, &mut encoded_output, Method::StdSimd32TwoStage);
/// encoded_output.truncate(output_length);
/// ```
///
pub fn cobs_encode_to(input: &[u8], output: &mut [u8], method: Method) -> usize {
    match method {
        Method::Trivial => cobs_encode_to_trivial(input, output),
        Method::Simd16 => cobs_encode_to_opt(input, output),
        Method::Crazy => cobs_encode_to_c(input, output),
        Method::StdSimd8 => cobs_encode_to_std::<8>(input, output),
        Method::StdSimd16 => cobs_encode_to_std::<16>(input, output),
        Method::StdSimd32 => cobs_encode_to_std::<32>(input, output),
        Method::StdSimd8TwoStage => {
            cobs_encode_to_chained_iter::<SimdBlocksGeneric<8>>(input, output)
        }
        Method::StdSimd16TwoStage => {
            cobs_encode_to_chained_iter::<SimdBlocksGeneric<16>>(input, output)
        }
        Method::StdSimd32TwoStage => {
            cobs_encode_to_chained_iter::<SimdBlocksGeneric<32>>(input, output)
        }
    }
}

fn cobs_encode_to_std<const N: usize>(input: &[u8], output: &mut [u8]) -> usize {
    let mut out_idx = 0;
    for block in BlockIter::<SimdBlocksGeneric<32>>::new(input, 254) {
        output[out_idx] = block.len() as u8 + 1;
        // Copy all
        output[out_idx + 1..out_idx + 1 + block.len()].copy_from_slice(block);
        out_idx += block.len() + 1;
    }

    out_idx
}

fn cobs_encode_to_trivial(input: &[u8], output: &mut [u8]) -> usize {
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

fn cobs_encode_to_c(input: &[u8], output: &mut [u8]) -> usize {
    assert!(output.len() >= encoded_size_upper_bound(input.len()));
    assert!(!input.is_empty());
    assert!(!output.is_empty());
    let mut encode = &mut output[0] as *mut u8; // Encoded byte pointer
    let mut codep = encode; // Output code pointer
    encode = unsafe { encode.add(1) };
    let mut code = 1; // Code value

    let mut byte = &input[0] as *const u8;
    let mut length = input.len();

    while length > 0 {
        length -= 1;

        // SAFETY: byte points to input and is only incremented once per loop. loop only iterates for the length of input, guarded by length variable.
        if unsafe { *byte } != 0 {
            // Byte not zero, write it
            unsafe { *encode = *byte };

            code += 1;
            encode = unsafe { encode.add(1) };
        }

        if (unsafe { *byte } == 0) || code == 0xff {
            // Input is zero or block completed, restart

            unsafe { *codep = code };
            code = 1;
            codep = encode;

            if unsafe { *byte } == 0 || (length != 0) {
                encode = unsafe { encode.add(1) };
            }
        }

        byte = unsafe { byte.add(1) };
    }

    unsafe { *codep = code };

    unsafe { encode.offset_from(&output[0] as *const u8) as usize }
}

fn cobs_encode_to_opt(input: &[u8], output: &mut [u8]) -> usize {
    let mut out_idx = 0;
    for block in BlockIter::<SimdBlocks16>::new(input, 254) {
        output[out_idx] = block.len() as u8 + 1;
        // Copy all
        output[out_idx + 1..out_idx + 1 + block.len()].copy_from_slice(block);
        out_idx += block.len() + 1;
    }

    out_idx
}

fn cobs_encode_to_chained_iter<ZeroMethod: NextZeroIndex>(
    input: &[u8],
    output: &mut [u8],
) -> usize {
    let mut out_idx = 0;
    // This finds large non-zero blocks first, and then divides them, instead of directly finding non-zero blocks with maximum size
    for large_block in BlockIter::<ZeroMethod>::new(input, input.len()) {
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
pub fn cobs_encode_to_vec(input: &[u8]) -> Vec<u8> {
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
        cobs_decode, cobs_encode_to_c, cobs_encode_to_chained_iter, cobs_encode_to_opt,
        cobs_encode_to_trivial, cobs_encode_to_vec, encoded_size_upper_bound,
        next_zero_simd_128::SimdBlocks16, next_zero_std_simd::SimdBlocksGeneric,
    };
    use concat_idents::concat_idents;

    type EncodingFunction = dyn Fn(&[u8]) -> Vec<u8>;

    fn encode_to_wrapper(function: fn(&[u8], &mut [u8]) -> usize) -> Box<EncodingFunction> {
        Box::new(move |input: &[u8]| {
            let mut output_data = vec![0; encoded_size_upper_bound(input.len())];
            let s = function(input, &mut output_data);
            output_data.truncate(s);
            output_data
        })
    }

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

    encode_tests!(default, cobs_encode_to_vec);

    encode_tests!(to_buffer, encode_to_wrapper(cobs_encode_to_trivial));

    encode_tests!(to_buffer_opt, encode_to_wrapper(cobs_encode_to_opt));

    encode_tests!(
        chained_iter,
        encode_to_wrapper(cobs_encode_to_chained_iter::<SimdBlocks16>)
    );

    encode_tests!(
        chained_iter_32,
        encode_to_wrapper(cobs_encode_to_chained_iter::<SimdBlocksGeneric<32>>)
    );

    encode_tests!(c, encode_to_wrapper(cobs_encode_to_c));

    #[test]
    fn decoding_no_zeros_short() {
        assert_eq!(
            cobs_decode(&[0x05, 0x11, 0x22, 0x33, 0x44]),
            vec![0x11, 0x22, 0x33, 0x44]
        )
    }
}

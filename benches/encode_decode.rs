use std::{cmp::max, time::Duration};

use cobs_simd::{
    cobs_encode, cobs_encode_to, cobs_encode_to_chained_iter, cobs_encode_to_opt,
    encoded_size_upper_bound,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding");
    let seed: <Pcg64Mcg as SeedableRng>::Seed = Default::default();
    let mut rng = Pcg64Mcg::from_seed(seed);

    for size in (1000..=1000000).step_by(1000) {
        group.throughput(criterion::Throughput::Bytes(size as u64));
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(1));

        let mut data = vec![0_u8; size];
        rng.fill_bytes(&mut data);
        let slice: &[u8] = &data;

        let mut output_data = vec![
            0;
            max(
                encoded_size_upper_bound(size),
                corncobs::max_encoded_len(size)
            )
        ];
        let output_slice: &mut [u8] = &mut output_data;

        group.bench_with_input(BenchmarkId::new("Vec", size), slice, |b, input_data| {
            b.iter(|| cobs_encode(input_data));
        });
        group.bench_with_input(
            BenchmarkId::new("ToBuffer", size),
            slice,
            |b, input_data| {
                b.iter(|| cobs_encode_to(input_data, output_slice));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("BlockIter", size),
            slice,
            |b, input_data| {
                b.iter(|| cobs_encode_to_opt(input_data, output_slice));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("corncobs", size),
            slice,
            |b, input_data| {
                b.iter(|| corncobs::encode_buf(input_data, output_slice));
            },
        );

        group.bench_with_input(BenchmarkId::new("chained", size), slice, |b, input_data| {
            b.iter(|| cobs_encode_to_chained_iter(input_data, output_slice));
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

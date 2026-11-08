//! Bandwidth benchmark for vector_add across tile sizes.
//!
//! cargo run --release -p tilelab-kernels --bin bench_vector_add

use cuda_core::Context;
use cutile::bench::{do_bench, BenchOptions};
use cutile::prelude::*;
use tilelab_kernels::{elementwise::vector_add, with_big_stack};

fn main() {
    with_big_stack(|| {
        let device = Context::new(0).expect("no CUDA device 0");
        let stream = device.new_stream().expect("stream");
        let len: usize = 1 << 26; // 64M f32 = 256 MiB per tensor
        let bytes = 3 * len * std::mem::size_of::<f32>();

        let x = api::ones::<f32>(&[len]).sync_on(&stream).unwrap();
        let y = api::ones::<f32>(&[len]).sync_on(&stream).unwrap();
        let mut z = api::zeros::<f32>(&[len]).sync_on(&stream).unwrap();

        println!(
            "vector_add, n = {len}, {} MiB moved per launch",
            bytes >> 20
        );
        println!("{:>6}  {:>10}  {:>10}", "tile", "median ms", "GB/s");
        for tile in [256usize, 512, 1024, 2048, 4096] {
            let m = do_bench(&stream, &BenchOptions::default(), |s| {
                vector_add((&mut z).partition([tile]), &x, &y)
                    .sync_on(s)
                    .map(|_| ())
                    .map_err(Into::into)
            })
            .expect("bench");
            let ms = m.median_ms() as f64;
            println!(
                "{tile:>6}  {ms:>10.4}  {:>10.1}",
                bytes as f64 / (ms * 1e-3) / 1e9
            );
        }
    });
}

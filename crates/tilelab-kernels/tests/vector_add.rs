//! GPU correctness test: cuTile vector_add vs the CPU reference.
//! Needs an sm_80+ GPU and CUDA 13.3+ (13.3 is the minimum for Hopper).

use cutile::prelude::*;
use tilelab_kernels::{elementwise::vector_add, with_big_stack};

fn run(len: usize, tile: usize) {
    with_big_stack(move || {
        let x_host = tilelab_ref::seeded_vec(len, 1);
        let y_host = tilelab_ref::seeded_vec(len, 2);
        let want = tilelab_ref::vector_add(&x_host, &y_host);

        let x = api::copy_host_vec_to_device(&Arc::new(x_host))
            .sync()
            .unwrap();
        let y = api::copy_host_vec_to_device(&Arc::new(y_host))
            .sync()
            .unwrap();
        let mut z = api::zeros::<f32>(&[len]).sync().unwrap();

        let _ = vector_add((&mut z).partition([tile]), &x, &y)
            .sync()
            .unwrap();
        let got: Vec<f32> = z.to_host_vec().sync().unwrap();

        // An f32 add is exact up to rounding of the same op on the CPU.
        tilelab_ref::assert_close(&got, &want, 0.0, 0.0);
    });
}

#[test]
fn vector_add_small() {
    run(1024, 128);
}

#[test]
fn vector_add_large() {
    run(1 << 24, 1024);
}

//! Elementwise kernels. `vector_add` is the hello-world of the playground.

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// `z = x + y`. Each tile block owns one `S`-sized slice of `z`, chosen by
    /// how the host partitions `z`; `x` and `y` are read-only and shared.
    #[cutile::entry()]
    pub fn vector_add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

pub use kernels::vector_add;

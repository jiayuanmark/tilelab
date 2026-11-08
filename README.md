# tilelab

A playground for writing GPU kernels in Rust with NVIDIA's
[cuTile Rust](https://github.com/NVIDIA/cuda-rust) (`cutile` crate), working
toward the kernels a small mixture-of-experts LLM needs. See
[docs/PLAN.md](docs/PLAN.md) for the roadmap.

## Layout

```
crates/tilelab-ref      CPU reference implementations + allclose checks (no CUDA)
crates/tilelab-kernels  cuTile kernels, GPU tests, and benchmark binaries
scripts/check_env.sh    checks GPU, driver, CUDA toolkit and tileiras
```

## Requirements

- An `sm_80`+ NVIDIA GPU. Tested targets: H100 (`sm_90`), B200 and GB200 (`sm_100`).
- CUDA **13.3+** (Hopper needs Tile IR 13.3; Blackwell works from 13.2).
- Rust stable 1.89+.

## Quick start

```bash
./scripts/check_env.sh
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3   # or set it in .cargo/config.toml

cargo test -p tilelab-ref                         # CPU only, runs anywhere
cargo test -p tilelab-kernels                     # needs a GPU
cargo run --release -p tilelab-kernels --bin bench_vector_add
```

The first run of each kernel JIT-compiles it through `tileiras`, so it is slower;
later runs hit the cache. `vector_add` should land near your GPU's peak memory
bandwidth (roughly 3 TB/s on H100 SXM, 7+ TB/s on B200) at the larger tile sizes.

## Adding a kernel

1. Write the CPU reference in `crates/tilelab-ref/src/lib.rs` with a unit test.
2. Write the kernel in a module under `crates/tilelab-kernels/src/`.
3. Add a GPU test in `crates/tilelab-kernels/tests/` that compares against the reference.
4. Add a benchmark in `crates/tilelab-kernels/src/bin/` using `cutile::bench::do_bench`.

`cutile` is pinned to an exact version in the workspace `Cargo.toml` because the
API still changes between minor releases.

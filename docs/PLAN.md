# Plan

The goal is to learn to write cuTile Rust kernels (first) and grow them into a
small MoE inference engine (eventually). Target GPUs: H100, B200, GB200.

1. **Repo and toolchain.** Rust workspace on `cutile`, CUDA 13.3+, a CPU
   reference crate, and a `vector_add` kernel with a correctness test and a
   bandwidth benchmark run on each GPU. *(this step)*
2. **Kernels an MoE layer needs.** GEMM, RMSNorm, softmax, SiLU, top-k
   routing, then grouped GEMM over experts. Each one gets a reference test
   and a benchmark against cuBLAS. References: Grout, mistral.rs
   `mistralrs-quant/src/cutile`, candle `candle-nn/src/moe/cutile.rs`,
   TileGym `ops/cutile_rs`.
3. **One real MoE block.** Load one layer of OLMoE-1B-7B (then
   Qwen3-30B-A3B) from safetensors and match the Hugging Face output. Then
   FP8 on H100/B200, and NVFP4 on Blackwell only.
4. **Later: a full decode loop.** Attention, KV cache, tokenizer, tok/s
   against Grout.

## Hardware notes

| GPU   | Arch | cuTile minimum Tile IR | FP8 | FP4 / block-scaled MMA |
|-------|------|------------------------|-----|------------------------|
| H100  | sm_90  | 13.3 | yes | no  |
| B200  | sm_100 | 13.2 | yes | yes (Tile IR 13.3) |
| GB200 | sm_100 | 13.2 | yes | yes (Tile IR 13.3) |

CUDA 13.3+ on every machine keeps things uniform.

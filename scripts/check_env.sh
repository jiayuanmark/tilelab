#!/usr/bin/env bash
# Prints what cuTile Rust needs: GPU compute capability, driver, CUDA toolkit
# and tileiras. Run this first on each new machine.
set -u
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mfail\033[0m  %s\n' "$*"; }

echo "GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv,noheader |
    while IFS=, read -r idx name cc drv mem; do
      cc=${cc// /}
      case "$cc" in
        9.0) ok "gpu$idx$name sm_90 (Hopper: needs Tile IR 13.3+, no FP4/block-scaled MMA)";;
        10.*|11.*|12.*) ok "gpu$idx$name sm_${cc/./} (Blackwell: FP8 + FP4/NVFP4 available)";;
        8.*) ok "gpu$idx$name sm_${cc/./} (Ampere/Ada)";;
        *) bad "gpu$idx$name sm_${cc/./} is below sm_80 and unsupported";;
      esac
      echo "        driver$drv, memory$mem"
    done
else
  bad "nvidia-smi not found"
fi

echo "CUDA toolkit"
root="${CUDA_TOOLKIT_PATH:-${CUDA_HOME:-}}"
if [ -z "$root" ]; then
  for d in /usr/local/cuda-13.5 /usr/local/cuda-13.4 /usr/local/cuda-13.3 /usr/local/cuda-13 /usr/local/cuda /opt/cuda; do
    [ -d "$d" ] && root=$d && break
  done
fi
if [ -n "$root" ] && [ -d "$root" ]; then
  ok "toolkit at $root"
  if [ -x "$root/bin/nvcc" ]; then
    ver=$("$root/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    case "$ver" in
      13.[3-9]*|1[4-9].*) ok "nvcc $ver";;
      13.2*) warn "nvcc $ver works for Ampere/Ada/Blackwell but Hopper needs 13.3+";;
      *) bad "nvcc $ver is too old; install CUDA 13.3+";;
    esac
  fi
  tileiras="${CUTILE_TILEIRAS_PATH:-$root/bin/tileiras}"
  [ -x "$tileiras" ] && ok "tileiras at $tileiras" || bad "tileiras not found (looked at $tileiras)"
else
  bad "no CUDA toolkit found; set CUDA_TOOLKIT_PATH"
fi

echo "Rust"
if command -v rustc >/dev/null; then ok "$(rustc --version) (needs 1.89+)"; else bad "rustc not found"; fi

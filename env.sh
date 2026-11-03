# source env.sh: CUDA toolkit matching torch 2.8.0+cu128, used to JIT-compile the perception kernels
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9   # RTX 4060 (Ada)
export MAX_JOBS=4                 # keep nvcc within WSL's RAM

# scripts import each other across the step folders
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _d in "$_root"/scripts/*/; do PYTHONPATH="${_d%/}:${PYTHONPATH:-}"; done
export PYTHONPATH="$_root/src:$PYTHONPATH"
unset _root _d

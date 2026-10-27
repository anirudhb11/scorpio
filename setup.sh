#!/bin/bash
# Build the virtual environment.
#
#   bash setup.sh
#
# Needs internet, the CUDA 13 toolkit, rust and protoc on PATH. The CUDA
# toolkit is also needed at runtime: triton and flashinfer JIT-compile kernels.
set -euo pipefail

VENV="${VENV:-$PWD/.venv}"
SGLANG_REF="${SGLANG_REF:-5af1f949c}"
PYVER="${PYVER:-3.12}"

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PROTOC="$(command -v protoc)"
export UV_TORCH_BACKEND=cu130

uv venv --python "$PYVER" "$VENV"
source "$VENV/bin/activate"

# sglang first: it pulls torch and flashinfer.
uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git@${SGLANG_REF}#subdirectory=python"

# Later transformers releases register a config that collides with sglang's
# own, which breaks the engine import.
uv pip install "transformers==5.12.1"

# Linear-attention kernels for Qwen3.5. causal-conv1d builds against the
# installed torch, so it needs nvcc and no build isolation.
uv pip install "flash-linear-attention[cuda,tilelang]==0.5.1"
uv pip install ninja setuptools wheel packaging
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install "causal-conv1d>=1.4.0" --no-build-isolation

uv pip install openai hydra-core matplotlib

# math-verify grades math answers by equivalence rather than string match.
uv pip install "math-verify==0.9.0" latex2sympy2_extended

# reasoning_gym grades the RG puzzles. Several games register only when
# magiccube and pycosat are installed; without them those games score 0.
uv pip install "reasoning_gym==0.1.25" magiccube pycosat

# nvidia-cutlass-dsl-libs-base and -cu13 both ship
# cutlass/_mlir/dialects/_gpu_ops_gen.py with different contents; only base's
# copy matches the shared library. Reinstall base last so it wins.
CUTLASS_VER=$(python -c "from importlib.metadata import version; print(version('nvidia-cutlass-dsl-libs-base'))")
uv pip install --force-reinstall --no-deps "nvidia-cutlass-dsl-libs-base==${CUTLASS_VER}"


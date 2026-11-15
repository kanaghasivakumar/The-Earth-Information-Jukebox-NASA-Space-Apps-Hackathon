#!/bin/bash
# Build the isolated MuseControlLite env "mcl" (never touches the shared jukebox env).
set -euo pipefail
export MAMBA_ROOT_PREFIX=/projects/e32706/omb8654/conda
MM=/projects/e32706/omb8654/bin/micromamba
export PIP_CACHE_DIR=/projects/e32706/omb8654/pip_cache_mcl
if [ ! -x $MAMBA_ROOT_PREFIX/envs/mcl/bin/python ]; then
  $MM create -y -n mcl -c conda-forge python=3.11 ffmpeg "numpy<2" scipy cython pip git
fi
PY=$MAMBA_ROOT_PREFIX/envs/mcl/bin/python
$PY -m pip install --no-input torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install --no-input "git+https://github.com/fundwotsai2001/diffusers.git@v0.31.0-release" \
   soundfile accelerate transformers==4.46.1 matplotlib librosa torchsde gdown huggingface_hub safetensors \
   sentencepiece einops pyloudnorm "numpy<2"
$PY -m pip install --no-input --no-build-isolation "git+https://github.com/YianLai0327/madmom.git" || echo "madmom failed (rhythm extraction only)"
$PY -c "import torch, diffusers, transformers, librosa; print('OK', torch.__version__, diffusers.__version__, transformers.__version__)"
rm -rf $PIP_CACHE_DIR

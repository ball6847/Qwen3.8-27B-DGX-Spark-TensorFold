# syntax=docker/dockerfile:1
# Qwen3.8-27B + TensorFold, self-contained: build and run it with docker alone, no helper script.
#
#   docker build -t tensorfold-qwen38-27b:v0.6.0 .
#
#   docker run -d --name qwen38-27b-tf --gpus all --ipc=host --network host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v "$HOME/.cache/huggingface/hub":/root/.cache/huggingface \
#     -v "$HOME/.cache/tensorfold-qwen38-27b":/cache \
#     tensorfold-qwen38-27b:v0.6.0 \
#     tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --host 0.0.0.0 --port 8888 \
#     --name Qwen3.8-27B --parallel 8 --context 262144 --temperature 1.0 --top-p 0.95 --top-k 20 \
#     --prefill-fp8 --vision --thinking
#
# The two mounts hold the checkpoint cache (~19 GB of model files) and the compiled CUDA kernels; both outlive the
# container, so only the first start pays the download and the compile. The server takes its settings at run time --
# `tensorfold serve` flags and TENSORFOLD_* environment variables -- so changing one of them never needs a rebuild.
#
# Build arguments, all optional:
#   BASE_IMAGE     the NVIDIA PyTorch container to start from (published for amd64 and arm64)
#   TF_SPEC        what pip installs as TensorFold (git+<repo>@<ref>)
#   PATCHES_HASH   12 hex chars identifying ./patches (ls patches/*.patch | xargs cat | sha256sum | cut -c1-12);
#                  pass a fresh value after editing the patches, so the label keeps matching them
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

# TensorFold itself, then transformers (the vision tower's modules) and PyAV (video decoding) for --vision.
ARG TF_SPEC=git+https://github.com/ashhart/TensorFold.git@v0.6.0
RUN pip install --no-cache-dir --upgrade "${TF_SPEC}" && \
    pip install --no-cache-dir "transformers==5.17.0" av && \
    tensorfold --version

# Local fixes in ./patches (unified diffs against site-packages, applied with patch -p0, in file name order),
# then a check that the patched modules still import.
COPY patches/ /opt/tf-patches/
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.server, tensorfold.vision.qwen_cuda, tensorfold.vision.videos, av; \
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel" && \
    cat /opt/tf-patches/*.patch 2>/dev/null | sha256sum | cut -c1-12 > /etc/tf-patches-hash

# scripts/prepare.sh recognizes a prebuilt image by this label and the <TF_VERSION>-<PATCHES_HASH> tag; the hash of
# the patches that really landed in the image is in /etc/tf-patches-hash (the label is a build argument).
ARG PATCHES_HASH=fb91a0019fdc
LABEL tf.patches=${PATCHES_HASH} \
      org.opencontainers.image.title="Qwen3.8-27B TensorFold" \
      org.opencontainers.image.description="TensorFold v0.6.0 with patches, on NVIDIA PyTorch: serves Qwen3.8-27B with vision and DFlash2 drafts" \
      org.opencontainers.image.source="https://github.com/ball6847/Qwen3.8-27B-DGX-Spark-TensorFold" \
      org.opencontainers.image.version="v0.6.0" \
      org.opencontainers.image.licenses="Apache-2.0"

# Where the mounts go: the checkpoint cache, and the kernels compiled on the first start.
ENV HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace

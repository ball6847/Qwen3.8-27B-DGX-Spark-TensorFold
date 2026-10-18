# syntax=docker/dockerfile:1
# Qwen3.8-27B + TensorFold, ready to serve. This image adds this project's serve defaults on top of the published
# TensorFold image, so a plain `docker run` starts the server -- no command line, no helper script.
#
#   docker build -t qwen38-27b:v0.6.0 .        # seconds: only the config changes, every layer comes from upstream
#
#   docker run -d --name qwen38-27b --gpus all --ipc=host --network host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v "$HOME/.cache/huggingface/hub":/root/.cache/huggingface \
#     -v "$HOME/.cache/tensorfold-qwen38-27b":/cache \
#     qwen38-27b:v0.6.0
#
# First start on a new machine: the entrypoint fills the checkpoint cache itself -- it pulls Vontra/
# Qwen3.8-27B-MLX-4bit (~15 GB) and z-lab/Qwen3.8-27B-DFlash2 (~3.6 GB) into the mounted cache if missing, then serves.
# No manual pull, no extra command. To skip drafts entirely: append --no-drafts (or -e TENSORFOLD_NO_DRAFTS=1).
#
# Changing a setting: every default below is overridable where it lives -- `-e` for the environment switches (e.g.
# -e TENSORFOLD_KV_DTYPE=bf16), and a `tensorfold serve ...` command replaces the CMD entirely for the flags (e.g.
# ... qwen38-27b:v0.6.0 tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --port 9000 --parallel 4 --no-drafts).
# -e TENSORFOLD_KV_POOL_GIB=78 pins the attention cache pool; without it the caches grow on demand (the sizing
# ./start.sh does is host-side, it cannot live in the image).
#
# Layers: every one of them is upstream's, pinned here by digest. Anyone who has pulled
# ghcr.io/miaai-lab/qwen3.8-27b-dgx-spark-tensorfold gets this image in seconds and downloads nothing new; a fresh
# machine downloads exactly upstream's bytes once. Rebuilding the TensorFold + patches layers from the NVIDIA base
# (needed only to change ./patches) is Dockerfile.full.

ARG BASE_IMAGE=ghcr.io/miaai-lab/qwen3.8-27b-dgx-spark-tensorfold:v0.6.0-fb91a0019fdc@sha256:043d192b9e1725f55c5439fc9672055de891433e5b31448ed8963918e3c460e9
FROM ${BASE_IMAGE}

# What scripts/config.sh serves with; each one is still overridable with -e at run time.
ENV TENSORFOLD_MAX_IMAGES=50 \
    TENSORFOLD_IMAGE_TOKENS=16384 \
    TENSORFOLD_VIDEO_TOKENS=16384 \
    TENSORFOLD_KV_DTYPE=fp8 \
    TENSORFOLD_MEMORY_RESERVE_GIB=0 \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    HF_HUB_OFFLINE=1

# Upstream's tf.patches label is inherited unchanged; these are this fork's identity.
LABEL org.opencontainers.image.title="Qwen3.8-27B TensorFold (ready to serve)" \
      org.opencontainers.image.description="Qwen3.8-27B on one DGX Spark: TensorFold v0.6.0 with patches fb91a0019fdc, serving with vision and DFlash2 drafts by default" \
      org.opencontainers.image.source="https://github.com/ball6847/Qwen3.8-27B-DGX-Spark-TensorFold" \
      org.opencontainers.image.version="v0.6.0"

# The entrypoint auto-fills the cache, then runs what ./start.sh runs as the image's own default. Extra flags after
# the image name are appended to serve (last flag wins); a full `tensorfold serve ...` replaces the defaults.
COPY --chmod=755 entrypoint.sh /usr/local/bin/tensorfold-entrypoint.sh

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/usr/local/bin/tensorfold-entrypoint.sh"]
CMD ["tensorfold", "serve", "Vontra/Qwen3.8-27B-MLX-4bit", \
     "--host", "0.0.0.0", "--port", "8888", \
     "--name", "Qwen3.8-27B", "--parallel", "8", "--context", "262144", \
     "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", \
     "--prefill-fp8", "--vision", "--thinking"]

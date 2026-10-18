#!/usr/bin/env bash
# TensorFold container entrypoint (Docker).
# - Ensures the model + drafter checkpoints are in the HF cache (pulls once if not)
# - Serves with the baked defaults
# - `docker run image <flags>` appends flags to serve (last flag wins)
# - `docker run image tensorfold …` / `bash` etc. pass through untouched
set -euo pipefail

HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HOME
HUB="$HF_HOME/hub"
MODEL="${TENSORFOLD_MODEL:-Vontra/Qwen3.8-27B-MLX-4bit}"
DRAFTER="${TENSORFOLD_DRAFTER:-z-lab/Qwen3.8-27B-DFlash2}"

# Cache may be mounted flat at $HF_HOME instead of the canonical $HF_HOME/hub
if [[ -d "$HF_HOME/models--${MODEL//\//--}" && ! -d "$HUB/models--${MODEL//\//--}" ]]; then
  export HF_HUB_CACHE="$HF_HOME"
fi

have() {
  local d="models--${1//\//--}"
  [[ -d "$HUB/$d" || -d "$HF_HOME/$d" ]]
}

fetch() {
  if have "$1"; then
    echo "[entrypoint] checkpoint cached: $1"
    return 0
  fi
  echo "[entrypoint] pulling $1 into $HUB (one-time, needs network) …"
  HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME="$HF_HOME" tensorfold pull "$1"
}

case "${1:-}" in
  ""|-*)
    # Default serve mode: baked CMD defaults + any extra flags the user passed
    has_nd=0
    for a in "$@"; do
      if [[ "$a" == "--no-drafts" ]]; then has_nd=1; fi
    done
    no_drafts=$has_nd
    if [[ "${TENSORFOLD_NO_DRAFTS:-0}" == "1" ]]; then no_drafts=1; fi

    fetch "$MODEL"
    if [[ $no_drafts == 1 ]]; then
      if [[ $has_nd == 0 ]]; then set -- "$@" --no-drafts; fi
    else
      fetch "$DRAFTER"
    fi

    exec tensorfold serve "$MODEL" \
      --host "${TENSORFOLD_HOST:-0.0.0.0}" --port "${TENSORFOLD_PORT:-8888}" \
      --name "Qwen3.8-27B" --parallel 8 --context 262144 \
      --temperature 1.0 --top-p 0.95 --top-k 20 \
      --prefill-fp8 --vision --thinking "$@"
    ;;
  tensorfold)
    # Full passthrough (the image's default CMD lands here): only pre-fill the
    # cache when the command is `tensorfold serve`.
    if [[ "${2:-}" == "serve" ]]; then
      repo=""
      for a in "${@:3}"; do
        if [[ "$a" != -* && -z "$repo" ]]; then repo="$a"; fi
      done
      if [[ -n "$repo" && -z "${TENSORFOLD_MODEL:-}" ]]; then MODEL="$repo"; fi
      has_nd=0
      for a in "$@"; do
        if [[ "$a" == "--no-drafts" ]]; then has_nd=1; fi
      done
      fetch "$MODEL"
      if [[ $has_nd == 0 && "${TENSORFOLD_NO_DRAFTS:-0}" != "1" ]]; then fetch "$DRAFTER"; fi
    fi
    exec "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

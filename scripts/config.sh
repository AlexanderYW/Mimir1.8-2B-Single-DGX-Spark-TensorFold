# shellcheck shell=bash disable=SC2034
# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `FORMAT=bf16 scripts/prepare.sh`
#   2. ./.env (KEY=value lines, read, never run)
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# Checkpoint: Danish Foundation Models' DFM-Mimir v1.8 (the XL EMA checkpoint at step 3.8M), HRM-Text: two 16-layer stacks
# (L and H, 30.7M parameters a layer) run H_cycles 2 x (L_cycles 3 + 1) = 8 times a token, so 128 layer applications, each
# with its own keys and values; multi-head attention (12 heads of 128) with a sigmoid output gate, weightless RMS norms,
# SwiGLU 1536 -> 4096; a PrefixLM: the whole prompt attends bidirectionally, the reply causally. 1.79B parameters,
# 3.57 GB of BF16 in one shard, Gemma 4's 262,144-token vocabulary and native chat template (thinking off, tool calls).
MODEL_ID="${MODEL_ID:-danish-foundation-models/DFM-Mimir-v1.8}"
_pin=1; [[ "$MODEL_ID" == danish-foundation-models/DFM-Mimir-v1.8 ]] || _pin=0
(( ! _pin )) || : "${MODEL_REVISION:=30485dbad6f3ce50d09eaa963b8a8fce53a4290c}"
MODEL_REVISION="${MODEL_REVISION:-}"
unset _pin
CKPT_GIB=4

# What is served: no NVFP4 release of Mimir v1.8 exists, so scripts/prepare.sh exports one (scripts/nvfp4_export.py) from
# the BF16 release into EXPORT_DIR: the stacks' 128 projections ModelOpt NVFP4 (weights only, round-to-nearest, blocks of
# 16), the embedding, z_L_init and lm_head BF16. FORMAT=bf16 exports the release's weights as stored (same folder layout).
FORMAT="${FORMAT:-nvfp4}"                 # nvfp4 | bf16
HEAD_FORMAT="${HEAD_FORMAT:-bf16}"        # bf16 | nvfp4: lm_head too (0.4B parameters; faster decode, unmeasured quality)
# The window the export names (max_position_embeddings): the model card's own evaluation serves 8192 (vLLM --hf-overrides,
# plain RoPE) over the 4096 the release names. A PrefixLM prompt runs whole in one chunk: 8192 at most.
WINDOW="${WINDOW:-8192}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"     # cpu | cuda: where the export's arithmetic runs (cpu leaves the GPU alone)
QUANT_LABEL="$( [[ "$FORMAT" == nvfp4 ]] && echo "NVFP4 stacks (ModelOpt layout, RTN, W4A16), BF16 embedding and ${HEAD_FORMAT^^} head" || echo "BF16 as released")"

# The engine: TensorFold's Zig engine (tensorfold-native), built from TF_REPO at TF_REF (the v1.3.4 release) with patches/*.patch
# applied in order (git apply in the checkout's root), with Zig ZIG_VERSION. Text only.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-12ebd414de443485d281768036ac4835e5abe8fe}"   # v1.3.4
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# libtfgrammar.so comes prebuilt from TensorFold's own release archive, checked against its SHA-256 (xgrammar's Apache-2.0
# notices ship with it). The dense engine serves no structured output yet, so it is only there for the server's tool parser.
TF_RELEASE_VERSION=1.3.4
TF_RELEASE_AARCH64_SHA256=02566a5297dd67db9b0b834737fe73e192907454896ee7d097d3a0dfca303937
TF_TAG="v1.3.4-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-mimir18:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE grammar-$TF_RELEASE_VERSION-$TF_RELEASE_AARCH64_SHA256"
  ) | sha256sum | cut -c1-12
}
CONTAINER_NAME="${CONTAINER_NAME:-mimir18-tf}"

SERVED_NAME="${SERVED_NAME:-mimir-1.8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8890}"
CONTEXT="${CONTEXT:-$WINDOW}"
# Requests decoded together. Every token keeps 128 layer slots of keys and values (12 heads of 128, bf16): 768 KiB a token,
# a 512-position page 384 MiB, so a full 8192-token window is 6.4 GiB. The engine admits each request inside the memory
# the weights leave and makes the rest wait, so PARALLEL is a cap (at most 128, a round's rows).
PARALLEL="${PARALLEL:-4}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
# The release's chat template renders thinking off; the recipe passes --no-thinking so it stays off.
THINKING="${THINKING:-0}"
# Context copies from the prompt's own text (the dense engine has no draft head); drafts are verified, so replies match
# DRAFTS=0 token for token.
DRAFTS="${DRAFTS:-1}"
# Sampling defaults (a request's own values win): greedy, as the release's evaluations and smoke checks decode.
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:-0}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-8}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

# WEIGHTS_GIB: the weights' device bytes (cuda_weights.zig's deviceBytes): NVFP4 stacks 0.54 GiB, bf16 embedding 0.75 GiB,
# bf16 head 0.75 GiB, the recurrent states and scratch rows of an 8192-row prompt about 0.9 GiB. Not yet measured on a Spark.
WEIGHTS_GIB="${WEIGHTS_GIB:-$( [[ "$FORMAT" == nvfp4 ]] && echo 3 || echo 5 )}"
MEM_NEED_GIB="${MEM_NEED_GIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-$TENSORFOLD_MEMORY_RESERVE_GIB}"
# kv_gib <tokens>: one sequence's caches over its 512-position pages (pages.zig): 128 slots x 2 x 1536 x bf16 a position.
kv_gib() {
  local ctx=${1:-$CONTEXT}
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  local pages=$(( (ctx + 16 + 511) / 512 ))
  echo $(( (pages * 402653184 + 1073741823) / 1073741824 ))
}
# mem_need_gib <context>: MemAvailable a start needs: weights, one full window, 2 GiB of slack, the floor.
mem_need_gib() {
  local ctx=${1:-$CONTEXT}
  echo $(( WEIGHTS_GIB + $(kv_gib "$ctx") + 2 + MEM_FLOOR_GIB ))
}

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
EXPORT_ROOT="${EXPORT_ROOT:-$HF_CACHE/tensorfold-models}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-mimir18}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/mimir18-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-mimir18/logs}"
MIN_FREE_GB="${MIN_FREE_GB:-$((CKPT_GIB * 3 + 4))}"   # the release, its export, a staging copy
IMAGE_FREE_GB="${IMAGE_FREE_GB:-10}"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
snapshot_rev() {
  if [[ -n "$MODEL_REVISION" ]]; then echo "$MODEL_REVISION"
  else cat "$(model_cache_dir)/refs/main" 2>/dev/null; fi
}
snapshot_dir() { echo "$(model_cache_dir)/snapshots/$(snapshot_rev)"; }
# The export's folder: what it was made from and how, so a changed setting or exporter makes a new one.
export_tag() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    echo "$MODEL_ID $(snapshot_rev) $FORMAT $HEAD_FORMAT $WINDOW"; cat scripts/nvfp4_export.py
  ) | sha256sum | cut -c1-8
}
export_dir() {
  local rev; rev=$(snapshot_rev)
  local head=""; [[ "$FORMAT" == nvfp4 ]] && head="-head-$HEAD_FORMAT"
  echo "$EXPORT_ROOT/${MODEL_ID##*/}-${FORMAT}${head}-w${WINDOW}-${rev:0:8}-$(export_tag)"
}
# The export folder as the container sees it (EXPORT_ROOT is mounted at /models).
export_in_container() { echo "/models/$(basename "$(export_dir)")"; }

PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local label
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  echo "model=$MODEL_ID@$(snapshot_rev) export=$(basename "$(export_dir)") image=$label patches=$(image_hash)"
}

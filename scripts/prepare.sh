#!/usr/bin/env bash
# Prepare this Spark to serve DFM-Mimir v1.8 ($QUANT_LABEL) with TensorFold's Zig engine (tensorfold-native, one GPU):
#   1. preflight: docker, the GPU, memory, disk
#   2. the image: tensorfold-native built from TensorFold (TF_REF) plus patches/*.patch with Zig, on NVIDIA's PyTorch
#      container (kernels are embedded in the binary), with libtfgrammar.so from the matching release archive
#   3. the BF16 release in the Hugging Face cache (~3.6 GB) at its pinned revision, its SHA-256 checked
#   4. the export (scripts/nvfp4_export.py, inside the image, no network, CPU unless EXPORT_DEVICE=cuda) into EXPORT_DIR
#   5. verify it: HRM-Text's config, every projection in the format asked for, the tokenizer and chat template
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch, --image-only to stop after the image, --reexport to export again.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REBUILD=0
IMAGE_ONLY=0
REEXPORT=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    --image-only) IMAGE_ONLY=1 ;;
    --reexport) REEXPORT=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done
[[ "$FORMAT" =~ ^(nvfp4|bf16)$ ]] || die "FORMAT is nvfp4 or bf16, not $FORMAT"
[[ "$HEAD_FORMAT" =~ ^(bf16|nvfp4)$ ]] || die "HEAD_FORMAT is bf16 or nvfp4, not $HEAD_FORMAT"
[[ "$WINDOW" =~ ^[1-9][0-9]*$ && "$WINDOW" -le 8192 ]] || die "WINDOW is a token count up to 8192 (the PrefixLM prompt chunk), not $WINDOW"
[[ "$EXPORT_DEVICE" =~ ^(cpu|cuda)$ ]] || die "EXPORT_DEVICE is cpu or cuda, not $EXPORT_DEVICE"

mkdir -p "$KERNEL_CACHE" "$STATE_DIR" "$HF_CACHE/hub" "$EXPORT_ROOT"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it"

log "Preflight checks"
_mem_here=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
if (( _mem_here < MEM_FLOOR_GIB + 8 )); then   # the export holds the release and its output in memory: ~8 GiB
  _msg="only ${_mem_here} GiB MemAvailable (the floor $MEM_FLOOR_GIB plus 8 for the export): stop other work first"
  if [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$_msg"; else die "$_msg (MEM_CHECK=0 continues anyway)"; fi
fi
log "Memory: ${_mem_here} GiB MemAvailable (floor $MEM_FLOOR_GIB)"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found"
elif ! nvidia-smi -L >/dev/null 2>&1; then warn "nvidia-smi failed: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime; --gpus all may fail"

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
rev=$(snapshot_rev)
need_ckpt=0
[[ -f "$(model_cache_dir)/snapshots/$rev/model.safetensors" ]] || need_ckpt=$((CKPT_GIB + 1))
[[ -f "$(export_dir)/export.json" && $REEXPORT -eq 0 ]] || need_ckpt=$((need_ckpt + CKPT_GIB * 2))
(( ! IMAGE_ONLY )) || need_ckpt=0
need_img=0; [[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_img=$IMAGE_FREE_GB
if [[ "$(stat -c %d "$HF_CACHE")" == "$(stat -c %d "$DOCKER_ROOT" 2>/dev/null)" ]]; then
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt + need_img )) ||
    die "only ${have} GB free under $HF_CACHE (also Docker's root), ~$((need_ckpt + need_img)) GB needed (checkpoint and export ${need_ckpt}, image ${need_img})"
else
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt )) || die "only ${have} GB free under $HF_CACHE, ~${need_ckpt} GB needed for the checkpoint and its export"
  have=$(free_gb "$DOCKER_ROOT"); (( have >= need_img )) ||
    die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the image needs ~${need_img} GB"
fi
log "Disk: $(free_gb "$HF_CACHE") GB free under $HF_CACHE"

if [[ $REBUILD -eq 1 || "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]] || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }
  log "Building $IMAGE (TensorFold ${TF_REF:0:12}, Zig $ZIG_VERSION, patches $PATCHES_HASH)"
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/patches"
  compgen -G 'patches/*.patch' >/dev/null && cp -- patches/*.patch "$ctx/patches/"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  # shellcheck disable=SC2086  # DOCKER_BUILD_ARGS is a list of flags, split on purpose
  docker build ${DOCKER_BUILD_ARGS:-} "${nocache[@]}" -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
    --build-arg TF_RELEASE_VERSION="$TF_RELEASE_VERSION" --build-arg TF_RELEASE_AARCH64_SHA256="$TF_RELEASE_AARCH64_SHA256" \
    --build-arg PATCHES_HASH="$PATCHES_HASH" \
    -f - "$ctx" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS grammar
ARG TF_RELEASE_VERSION
ARG TF_RELEASE_AARCH64_SHA256
RUN mkdir -p /opt/tfrel && cd /tmp && \
    curl -fsSL -o tf-rel.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v${TF_RELEASE_VERSION}/tensorfold-${TF_RELEASE_VERSION}-linux-aarch64.tar.gz" && \
    echo "${TF_RELEASE_AARCH64_SHA256}  tf-rel.tar.gz" | sha256sum -c - && \
    tar -xzf tf-rel.tar.gz -C /opt/tfrel --strip-components=1 \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/lib/libtfgrammar.so \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-Apache-2.0.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/xgrammar-NOTICE.txt \
      tensorfold-${TF_RELEASE_VERSION}-linux-aarch64/LICENSES/dlpack-Apache-2.0.txt && \
    test -s /opt/tfrel/lib/libtfgrammar.so

FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# fatbins + native only: the default install also builds the test binaries
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
# the engine looks for libtfgrammar.so in <binary dir>/../lib, i.e. next to native/bin
COPY --from=grammar /opt/tfrel/lib/libtfgrammar.so /opt/tensorfold/native/lib/libtfgrammar.so
COPY --from=grammar /opt/tfrel/LICENSES /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.kernels=embedded
ENV HF_HOME=/root/.cache/huggingface CUDA_CACHE_PATH=/cache/nv
WORKDIR /workspace
DOCKERFILE
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built with patches $PATCHES_HASH"
fi
docker run --rm --network none --entrypoint test "$IMAGE" -x /opt/tensorfold/native/bin/tensorfold-native ||
  die "$IMAGE has no tensorfold-native"
log "Image $IMAGE: tensorfold-native, kernel set present"
if (( IMAGE_ONLY )); then log "--image-only: done; run scripts/prepare.sh again for the checkpoint and its export"; exit 0; fi

command -v hf >/dev/null || warn "host 'hf' CLI not found, downloading from inside the container"
download() {   # download <repo> [revision]
  if command -v hf >/dev/null; then
    hf download "$1" ${2:+--revision "$2"} --cache-dir "$HF_CACHE/hub" >/dev/null
  else
    docker run --rm --user "$(id -u):$(id -g)" --network host --entrypoint python ${HF_TOKEN:+-e HF_TOKEN} \
      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} \
      -v "$HF_CACHE":/hf -e HF_HOME=/hf -e HOME=/tmp "$IMAGE" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)' "$1" "${2:-}"
  fi
}
dir=$(model_cache_dir)
pin=$MODEL_REVISION
snap="$dir/snapshots/${pin:-x}"
if [[ -n "$pin" && -f "$snap/model.safetensors" && -f "$snap/chat_template.jinja" ]]; then
  log "Checkpoint $MODEL_ID @ ${pin:0:8} already in $HF_CACHE/hub"
else
  log "Downloading $MODEL_ID${pin:+ @ ${pin:0:8}} into $HF_CACHE/hub (~3.6 GB)"
  download "$MODEL_ID" "$pin" || die "$MODEL_ID: the download failed"
  [[ -z "$pin" || -f "$dir/refs/main" ]] || { mkdir -p "$dir/refs"; printf %s "$pin" > "$dir/refs/main"; }
fi
rev=$(snapshot_rev)
snap="$dir/snapshots/$rev"
[[ -n "$rev" && -f "$snap/config.json" ]] || die "$MODEL_ID: no snapshot after the download"

log "Verifying the release"
CHECK_SRC_PY='
import hashlib, json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
if cfg.get("model_type") != "hrm_text":
    sys.exit("model_type is %r, not hrm_text" % cfg.get("model_type"))
for f in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if not os.path.isfile(os.path.join(d, f)):
        sys.exit("missing " + f)
want = None
ver = os.path.join(d, "model-verification.json")
if os.path.isfile(ver):
    want = json.load(open(ver)).get("sha256", {}).get("model.safetensors")
if want:
    h = hashlib.sha256()
    with open(os.path.join(d, "model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        sys.exit("model.safetensors sha256 %s, the release names %s" % (h.hexdigest(), want))
print("hrm_text, H %d x L %d, %d layers a stack, %d heads, vocab %d%s" % (cfg["H_cycles"], cfg["L_cycles"],
      cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["vocab_size"], ", sha256 matches the release" if want else ""))'
info=$(python3 -I -c "$CHECK_SRC_PY" "$snap" 2>&1) || die "the checkpoint at $snap is not ready: $info"
log "Release OK: $info"

out=$(export_dir)
if [[ -f "$out/export.json" && $REEXPORT -eq 0 ]]; then
  log "Export already made: $out"
else
  gpu=(); [[ "$EXPORT_DEVICE" == cuda ]] && gpu=(--gpus all)
  run_export=(docker run --rm --network none "${gpu[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp
              -v "$HF_CACHE":/hf:ro -v "$EXPORT_ROOT":/models -v "$PWD/scripts":/recipe:ro
              --entrypoint python3 "$IMAGE" -I /recipe/nvfp4_export.py)
  log "Checking the exporter (self-test)"
  "${run_export[@]}" --self-test
  log "Exporting $FORMAT (head $HEAD_FORMAT, window $WINDOW, on $EXPORT_DEVICE) into $out"
  "${run_export[@]}" --src "/hf/${snap#"$HF_CACHE"/}" --out "$(export_in_container)" \
    --format "$FORMAT" --head "$HEAD_FORMAT" --window "$WINDOW" --device "$EXPORT_DEVICE"
fi

log "Verifying the export"
CHECK_OUT_PY='
import json, os, struct, sys
d, fmt, head, window = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = json.load(open(os.path.join(d, "config.json")))
if cfg.get("model_type") != "hrm_text" or cfg.get("max_position_embeddings") != window:
    sys.exit("config.json is not the export asked for (model_type %r, window %r)" % (cfg.get("model_type"), cfg.get("max_position_embeddings")))
q = cfg.get("quantization_config")
if (fmt == "nvfp4") != bool(q) or (q and (q.get("quant_method"), q.get("quant_algo")) != ("modelopt", "NVFP4")):
    sys.exit("quantization_config %r does not match FORMAT=%s" % (q, fmt))
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "model.safetensors"):
    if not os.path.isfile(os.path.join(d, f)):
        sys.exit("missing " + f)
with open(os.path.join(d, "model.safetensors"), "rb") as f:
    h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
h.pop("__metadata__", None)
proj = [n for n in h if n.endswith(".weight") and "_module.layers." in n]
if len(proj) != 2 * cfg["num_hidden_layers"] * 4:
    sys.exit("%d projections, expected %d" % (len(proj), 2 * cfg["num_hidden_layers"] * 4))
want = "U8" if fmt == "nvfp4" else "BF16"
bad = [n for n in proj if h[n]["dtype"] != want]
if bad:
    sys.exit("%d projections not %s, e.g. %s" % (len(bad), want, bad[0]))
if fmt == "nvfp4":
    for n in proj:
        s, g = n + "_scale", n + "_scale_2"
        if h.get(s, {}).get("dtype") != "F8_E4M3" or h.get(g, {}).get("dtype") != "F32":
            sys.exit(n + ": no E4M3 block scales or F32 global scale")
hd = h["lm_head.weight"]["dtype"]
if hd != ("U8" if fmt == "nvfp4" and head == "nvfp4" else "BF16"):
    sys.exit("lm_head is %s, HEAD_FORMAT=%s" % (hd, head))
e = json.load(open(os.path.join(d, "export.json")))
print("%s, %d projections %s, head %s, window %d, %.2f GiB; spot-check relative error %.4f" % (fmt, len(proj), want, hd,
      window, os.path.getsize(os.path.join(d, "model.safetensors")) / 2**30, e.get("spot_check_max_relative_error", 0)))'
info=$(python3 -I -c "$CHECK_OUT_PY" "$out" "$FORMAT" "$HEAD_FORMAT" "$WINDOW" 2>&1) || die "the export at $out is not ready: $info (scripts/prepare.sh --reexport)"
log "Export OK: $info"
prepared_state > "$PREPARED_MARKER"
log "Done. Start the server with ./start.sh (port $PORT)."

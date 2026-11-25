#!/usr/bin/env python3
"""Export DFM-Mimir-v1.8 (HRM-Text, BF16) as a ModelOpt-layout NVFP4 checkpoint folder for TensorFold.

No NVFP4 release of Mimir v1.8 exists, so the recipe makes one. Every projection of the two stacks
(attn.gqkv_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj: 128 tensors, 0.98B of the 1.79B parameters) is
quantized weight-only, the way ModelOpt's NVFP4 export stores weights:

    weight           uint8  [n, k/2]   E2M1 codes, two a byte, the even input in the low nibble
    weight_scale     F8_E4M3 [n, k/16] one scale a block of 16 inputs, relative to weight_scale_2
    weight_scale_2   F32    []        amax(|W|) / (6 * 448)

    W ~= E2M1[code] * float(weight_scale) * weight_scale_2

Block scales are amax(block) / 6 / weight_scale_2 rounded to E4M3, and each value is rounded to the nearest E2M1
point after dividing by its dequantized block scale (round-to-nearest, no calibration data: RTN). No input_scale is
written: TensorFold's lane matmul runs these weights with bf16 rows (W4A16), so activations are not quantized.

The fused gate|q|k|v and gate|up tensors keep the release's own names and are quantized whole (one weight_scale_2 each),
since the engine reads them fused. The embedding, z_L_init and (by default) lm_head stay BF16, as ModelOpt leaves
lm_head and embeddings unquantized by default.

The folder also gets the tokenizer and chat template as released, a generation_config.json naming the stop ids
(<turn|> 106, <|tool_response> 50, <eos> 1), and a config.json with quantization_config. --window sets
max_position_embeddings (the model card's own evaluation serves 8192 with plain RoPE via vLLM's --hf-overrides; the
export records the original 4096 as hrm_trained_max_position_embeddings).

Usage (scripts/prepare.sh runs this inside the recipe's image, CPU by default):
    python3 -I nvfp4_export.py --src SNAPSHOT_DIR --out OUT_DIR [--head bf16|nvfp4] [--format nvfp4|bf16]
                               [--window 8192] [--device cpu|cuda]
    python3 -I nvfp4_export.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import warnings

import torch

# the source shards are mapped read-only and only ever read: torch.frombuffer's writability warning does not apply
warnings.filterwarnings("ignore", message="The given buffer is not writable")

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16

# the projections of both stacks; everything else is copied as stored
PROJ = re.compile(r"^model\.(L|H)_module\.layers\.\d+\.(attn\.gqkv_proj|attn\.o_proj|mlp\.gate_up_proj|mlp\.down_proj)\.weight$")
KEEP = {"model.embed_tokens.weight", "model.z_L_init", "lm_head.weight"}
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE")
STOP_IDS = {"<turn|>": 106, "<|tool_response>": 50, "<eos>": 1}

ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn}
ST_NAMES = {v: k for k, v in ST_DTYPES.items()}


# --------------------------------------------------------------------------- safetensors, without the package
def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def read_tensor(mm: memoryview, base: int, meta: dict) -> torch.Tensor:
    a, b = meta["data_offsets"]
    dtype = ST_DTYPES[meta["dtype"]]
    t = torch.frombuffer(mm[base + a: base + b], dtype=dtype) if b > a else torch.empty(0, dtype=dtype)
    return t.reshape(meta["shape"])


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    header: dict = {"__metadata__": metadata}
    at = 0
    order = sorted(tensors)
    for name in order:
        t = tensors[name]
        nbytes = t.numel() * t.element_size()
        header[name] = {"dtype": ST_NAMES[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + nbytes]}
        at += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # the data starts 8-byte aligned
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for name in order:
            t = tensors[name].contiguous()
            f.write(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- NVFP4
def _e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude code of |y| (|y| <= 6), a tie to the even code (mantissa bit 0), as cvt.rn rounds."""
    grid = E2M1.to(y.device)
    dist = (y.abs()[..., None] - grid).abs()
    best = dist == dist.amin(-1, keepdim=True)
    even = torch.arange(8, device=y.device) % 2 == 0
    # among the nearest (one, or two at a tie): the even code scores 2, an odd one 1; argmax takes the best
    return (best.to(torch.int8) * (1 + even.to(torch.int8))).argmax(-1)


def quantize_nvfp4(w: torch.Tensor, device: str = "cpu", rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ModelOpt NVFP4 weights of w [n, k]: codes u8 [n, k/2], E4M3 block scales [n, k/16], fp32 global [].

    The global scale is the whole tensor's; the codes are made `rows` rows at a time to bound the memory."""
    n, k = w.shape
    if k % BLOCK:
        raise ValueError(f"inputs {k} not a multiple of {BLOCK}")
    amax = w.abs().amax().to(device=device, dtype=torch.float32)
    g = (amax / (FP4_MAX * E4M3_MAX)).clamp_min(torch.finfo(torch.float32).tiny)
    packed = torch.empty(n, k // 2, dtype=torch.uint8)
    scales = torch.empty(n, k // BLOCK, dtype=torch.float8_e4m3fn)
    for r0 in range(0, n, rows):
        x = w[r0:r0 + rows].to(device=device, dtype=torch.float32)
        m = x.shape[0]
        blocks = x.reshape(m, k // BLOCK, BLOCK)
        scale = (blocks.abs().amax(-1) / FP4_MAX / g).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        deq = scale.to(torch.float32) * g  # each block's dequantized scale
        safe = torch.where(deq > 0, deq, torch.ones_like(deq))
        y = (blocks / safe[..., None]).clamp(-FP4_MAX, FP4_MAX)
        mag = _e2m1_codes(y)
        mag = torch.where(deq[..., None] > 0, mag, torch.zeros_like(mag))
        code = (mag | (((y < 0) & (mag != 0)).to(mag.dtype) << 3)).to(torch.uint8).reshape(m, k)
        packed[r0:r0 + m] = (code[:, 0::2] | (code[:, 1::2] << 4)).cpu()
        scales[r0:r0 + m] = scale.cpu()
    return packed, scales, g.reshape(()).cpu().to(torch.float32)


def dequantize_nvfp4(packed: torch.Tensor, scale: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """TensorFold's reference decode (tools/zig/ricursive_reference.py's nvfp4), in float64."""
    table = torch.cat((E2M1, -E2M1)).to(torch.float64)
    n = packed.shape[0]
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(n, -1).long()
    blocks = scale.to(torch.float64).repeat_interleave(BLOCK, dim=1)
    return table[codes] * blocks * g.to(torch.float64)


# --------------------------------------------------------------------------- export
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def export(src: Path, out: Path, head: str, fmt: str, window: int, device: str) -> None:
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("model_type") != "hrm_text":
        sys.exit(f"{src}: model_type is {cfg.get('model_type')!r}, not hrm_text")
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"{src}: no safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent))
    try:
        tensors: dict[str, torch.Tensor] = {}
        seen = 0
        err_max = 0.0
        t0 = time.time()
        for shard in shards:
            header, base = read_header(shard)
            header.pop("__metadata__", None)
            with open(shard, "rb") as f:
                mm = memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
                for name, meta in header.items():
                    t = read_tensor(mm, base, meta)
                    quant = fmt == "nvfp4" and (PROJ.match(name) or (name == "lm_head.weight" and head == "nvfp4"))
                    if quant:
                        if t.dtype != torch.bfloat16 or t.dim() != 2:
                            sys.exit(f"{name}: {t.dtype} {list(t.shape)}, expected a bf16 matrix")
                        packed, scale, g = quantize_nvfp4(t, device)
                        stem = name[: -len(".weight")]
                        tensors[stem + ".weight"] = packed
                        tensors[stem + ".weight_scale"] = scale
                        tensors[stem + ".weight_scale_2"] = g
                        if seen % 32 == 0:  # spot-check the round trip as the engine decodes it
                            d = dequantize_nvfp4(packed, scale, g)
                            ref = t.to(torch.float64)
                            rel = ((d - ref).norm() / ref.norm().clamp_min(1e-30)).item()
                            err_max = max(err_max, rel)
                        seen += 1
                    elif PROJ.match(name) or name in KEEP:
                        tensors[name] = t.clone()
                    else:
                        sys.exit(f"{name}: a tensor this export does not know (not HRM-Text v1.8's layout)")
                del mm
        want = 128 if fmt == "nvfp4" else 0
        if fmt == "nvfp4" and seen < want:
            sys.exit(f"quantized {seen} projections, expected at least {want}")
        meta = {"format": "pt", "producer": "mimir1.8-2b-tensorfold nvfp4_export.py", "scheme": fmt}
        write_safetensors(work / "model.safetensors", tensors, meta)
        del tensors

        q_layers = sorted({m.group(0)[: -len(".weight")] for shard in shards for m in
                           (PROJ.match(n) for n in read_header(shard)[0]) if m})
        out_cfg = dict(cfg)
        out_cfg["hrm_trained_max_position_embeddings"] = cfg.get("max_position_embeddings")
        out_cfg["max_position_embeddings"] = window
        out_cfg.pop("quantization_config", None)
        if fmt == "nvfp4":
            exclude = ["model.embed_tokens", "model.z_L_init"] + ([] if head == "nvfp4" else ["lm_head"])
            out_cfg["quantization_config"] = {
                "quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": BLOCK,
                "kv_cache_quant_algo": None, "exclude_modules": exclude,
                "weights_only": True, "calibration": "none (round-to-nearest)",
            }
            (work / "hf_quant_config.json").write_text(json.dumps({
                "producer": {"name": "mimir1.8-2b-tensorfold", "version": "1"},
                "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None, "group_size": BLOCK,
                                 "exclude_modules": exclude},
            }, indent=2) + "\n")
        (work / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")
        gen = {"bos_token_id": cfg.get("bos_token_id", 2), "pad_token_id": cfg.get("pad_token_id", 0),
               "eos_token_id": list(STOP_IDS.values())}
        (work / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
        for name in COPY_FILES:
            if (src / name).exists():
                shutil.copyfile(src / name, work / name)
        if not (work / "chat_template.jinja").exists():
            sys.exit(f"{src}: no chat_template.jinja")
        provenance = {
            "source": str(src), "source_sha256": {s.name: sha256(s) for s in shards},
            "format": fmt, "head": head if fmt == "nvfp4" else "bf16", "window": window,
            "quantized_projections": seen, "quantized_layers": q_layers if fmt == "nvfp4" else [],
            "spot_check_max_relative_error": err_max, "seconds": round(time.time() - t0, 1),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (work / "export.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if out.exists():
            shutil.rmtree(out)
        os.replace(work, out)
        print(f"exported {fmt} ({seen} projections quantized, head {provenance['head']}, window {window}) to {out} "
              f"in {provenance['seconds']} s; spot-check max relative error {err_max:.4f}")
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- self-test
def self_test() -> None:
    torch.manual_seed(0)
    # every E2M1 point times every power-of-two scale comes back exactly
    w = torch.cat((E2M1, -E2M1)).repeat(4, 2).to(torch.bfloat16)  # [4, 32]
    w[1] *= 0.25
    w[2] *= 8
    p, s, g = quantize_nvfp4(w)
    assert p.shape == (4, 16) and s.shape == (4, 2) and s.dtype == torch.float8_e4m3fn and g.dim() == 0
    back = dequantize_nvfp4(p, s, g).to(torch.bfloat16)  # the scales' product is 1 to within float64 rounding
    assert torch.equal(back, w), (back, w)
    # ties go to the even code: 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert _e2m1_codes(ties).tolist() == [0, 2, 2, 4, 4, 6, 6], _e2m1_codes(ties).tolist()
    assert _e2m1_codes(torch.tensor([0.76, 0.74, 3.4, 5.1, -6.0])).tolist() == [2, 1, 5, 7, 7]
    # row chunks share the tensor's global scale: chunked and whole agree bit for bit
    big = (torch.randn(300, 64) * 0.05).to(torch.bfloat16)
    a, b = quantize_nvfp4(big, rows=7), quantize_nvfp4(big, rows=4096)
    assert all(torch.equal(x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x, y.view(torch.uint8) if y.dtype == torch.float8_e4m3fn else y) for x, y in zip(a, b))
    # nibble order: input 0 in the low nibble of byte 0
    one = torch.zeros(1, 16, dtype=torch.bfloat16)
    one[0, 0] = 6.0
    one[0, 1] = -0.5
    p, s, g = quantize_nvfp4(one)
    assert p[0, 0].item() == (7 | (9 << 4)), hex(p[0, 0].item())
    # a zero block stays zero, its scale 0
    z = torch.zeros(2, 32, dtype=torch.bfloat16)
    z[1, 16:] = 1.0
    p, s, g = quantize_nvfp4(z)
    assert torch.equal(dequantize_nvfp4(p, s, g)[0], torch.zeros(32, dtype=torch.float64)) and s[0, 0].float() == 0
    # Gaussian weights: RTN NVFP4's relative error sits near 0.09-0.11
    w = (torch.randn(256, 1536) * 0.02).to(torch.bfloat16)
    p, s, g = quantize_nvfp4(w)
    ref = w.to(torch.float64)
    rel = ((dequantize_nvfp4(p, s, g) - ref).norm() / ref.norm()).item()
    assert 0.05 < rel < 0.14, rel
    # the safetensors writer reads back
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.safetensors"
        write_safetensors(path, {"a": p, "b": s, "c": g, "d": w}, {"format": "pt"})
        header, base = read_header(path)
        assert base % 8 == 0
        with open(path, "rb") as f:
            mm = memoryview(f.read())
        for name, t in {"a": p, "b": s, "c": g, "d": w}.items():
            got = read_tensor(mm, base, header[name])
            assert got.dtype == t.dtype and list(got.shape) == list(t.shape)
            assert torch.equal(got.reshape(-1).view(torch.uint8), t.contiguous().reshape(-1).view(torch.uint8)), name
    print(f"self-test ok: exact grid, nibble order, zero blocks, Gaussian relative error {rel:.4f}, safetensors round trip")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--format", choices=("nvfp4", "bf16"), default="nvfp4")
    ap.add_argument("--head", choices=("bf16", "nvfp4"), default="bf16")
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.src or not a.out:
        ap.error("--src and --out are required")
    if not 1 <= a.window <= 8192:
        ap.error("--window is 1 to 8192: a PrefixLM prompt runs whole in TensorFold's 8192-row chunk")
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))
    export(a.src.resolve(), a.out, a.head, a.format, a.window, a.device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check the engine's NVFP4 path: transformers' HrmTextForCausalLM with the export's NVFP4 projections decoded to BF16
(E2M1 code x E4M3 scale x f32 global, the ModelOpt layout), greedy, against the running server's greedy replies.

The two compute the same weights, so the replies should match token for token, ties aside. At the first token that
differs, this prints both tokens' logits from a teacher-forced transformers forward (equal logits are a tie).

Run it with the NVFP4 export served (./start.sh), in an environment with transformers 5.24+ on a GPU:
    python3 tools/nvfp4_reference.py --model <the BF16 release's snapshot> --export <the NVFP4 export folder>
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--export", required=True)
ap.add_argument("--url", default="http://127.0.0.1:%s" % os.environ.get("PORT", "8890"))
ap.add_argument("--tokens", type=int, default=128)
a = ap.parse_args()
M, X, url = a.model, a.export, a.url.rstrip("/")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
served = json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))["data"][0]["id"]
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
sd = dict(model.named_parameters()); n = 0
with safe_open(X + "/model.safetensors", "pt", device="cuda") as f:
    for k in f.keys():
        if not k.endswith(".weight") or f.get_slice(k).get_dtype() != "U8":
            continue
        w = f.get_tensor(k); s = f.get_tensor(k + "_scale").float(); g = f.get_tensor(k + "_scale_2").float()
        codes = torch.stack([w & 15, w >> 4], -1).reshape(w.shape[0], -1).long()
        v = E2M1[codes] * s.repeat_interleave(16, 1) * g
        v = v.to(torch.bfloat16); base = k[: -len(".weight")]
        if base.endswith("attn.gqkv_proj"):   # rows gate | q | k | v, as transformers' conversion chunks them
            parts = dict(zip(("gate_proj", "q_proj", "k_proj", "v_proj"), v.chunk(4, 0)))
            pre = base.replace("attn.gqkv_proj", "self_attn.")
        elif base.endswith("mlp.gate_up_proj"):
            parts = dict(zip(("gate_proj", "up_proj"), v.chunk(2, 0))); pre = base.replace("gate_up_proj", "")
        else:
            parts = {"": v}; pre = base.replace("attn.o_proj", "self_attn.o_proj")
        with torch.no_grad():
            for suffix, t in parts.items():
                sd[pre + suffix + ".weight"].copy_(t); n += 1
if not n:
    sys.exit("no NVFP4 projections in %s: is it the NVFP4 export?" % X)
print("decoded the export's NVFP4 projections into %d of transformers' weights" % n)
stop = [tok.convert_tokens_to_ids(t) for t in ("<turn|>", "<|tool_response>")]
for name, prompt in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, enable_thinking=False)["input_ids"].cuda()
    with torch.no_grad():
        gen = model.generate(input_ids=ids, token_type_ids=torch.ones_like(ids), max_new_tokens=a.tokens, do_sample=False,
                             eos_token_id=stop, pad_token_id=tok.pad_token_id)
    hf = [t for t in gen[0, ids.shape[1]:].tolist() if t not in stop]
    body = {"model": served, "max_tokens": a.tokens, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    eng = tok(r["choices"][0]["message"]["content"], add_special_tokens=False)["input_ids"]
    k = next((i for i, (a, b) in enumerate(zip(eng, hf)) if a != b), None)
    if k is None:
        print(f"{name:8s} same over {min(len(eng), len(hf))} tokens (engine {len(eng)}, HF {len(hf)})"); continue
    seq = torch.tensor([ids[0].tolist() + hf[:k]]).cuda(); tt = torch.zeros_like(seq); tt[0, :ids.shape[1]] = 1
    with torch.no_grad():
        lg = model(input_ids=seq, token_type_ids=tt).logits[0, -1].float()
    print(f"{name:8s} first differs at token {k} of {len(hf)}: HF {tok.decode([hf[k]])!r} {lg[hf[k]]:.3f} vs engine {tok.decode([eng[k]])!r} {lg[eng[k]]:.3f}; teacher-forced argmax {tok.decode([lg.argmax().item()])!r}")

#!/usr/bin/env python3
"""Greedy reference replies from Transformers' own HrmTextForCausalLM (BF16), for tools/check.py --reference.

The release is served PrefixLM: the whole rendered prompt is one bidirectional block (token_type_ids = 1 for every
prompt token), the reply causal, as its vLLM evaluation and the agent-tools pipeline run it. This script does the same
with transformers >= 5.24 (the hrm_text model), decoding greedily to <turn|> or <|tool_response>.

Run it in an environment with transformers 5.24+ and torch on a GPU (e.g. the DFM-Mimir agent-tools venv), when the
GPU is free:
    python3 tools/reference.py --model ~/.cache/huggingface/hub/models--danish-foundation-models--DFM-Mimir-v1.8/snapshots/30485dba... \
        --out work/reference.json [--tokens 128] [--attn sdpa]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the BF16 release's snapshot folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--attn", default="sdpa", help="sdpa or eager (FlashAttention cannot express the PrefixLM mask)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn).to(a.device).eval()
    stop = [tok.convert_tokens_to_ids(t) for t in ("<turn|>", "<|tool_response>")]
    out = {"model": os.path.abspath(a.model), "tokens": a.tokens, "attn": a.attn, "replies": {}}
    for name, prompt in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"].to(a.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(input_ids=ids, token_type_ids=torch.ones_like(ids), max_new_tokens=a.tokens,
                                 do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id)
        new = gen[0, ids.shape[1]:].tolist()
        while new and new[-1] in stop:
            new.pop()
        text = tok.decode(new, skip_special_tokens=True)
        out["replies"][name] = {"prompt_tokens": ids.shape[1], "ids": new, "text": text}
        print("%-8s %5d prompt tokens, %3d reply tokens, %.1fs: %r" % (name, ids.shape[1], len(new), time.time() - t0, text[:60]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode and prefill speed against a running server. Standard library only.

Decode: three request shapes at 1..N concurrent streams, thinking off, --tokens tokens a reply (ignore_eos):
  dansk  a Danish prose question (the language DFM-Mimir was trained for)
  prose  the same question in English
  code   a short coding task
A request's speed is the engine's own tokens_per_second (decode only, first token excluded) when the reply carries it,
else completion tokens over wall time; "aggregate" sums the concurrent requests. Each cell is the median of --reps runs.
"tok/round" is the context-copy drafts' accepted tokens a verify round (1.00 without drafts).

Prefill: prompts of about --sizes tokens, fresh random text every run. Mimir is PrefixLM, so a prompt always runs whole
in one bidirectional chunk and nothing is cached; the check for cached tokens stays as a guard. Rate = prompt tokens
the server prefilled (prompt minus cached) / the server's prefill seconds. The largest default leaves room for the
reply in the 8,192-token window.

Usage: python3 tools/bench.py [--url http://127.0.0.1:8890] [--streams 1,2,4] [--shapes dansk,prose,code]
                              [--tokens 256] [--reps 3] [--temperature 0] [--sizes 1024,2048,4096,8000]
                              [--decode-only | --prefill-only] [--label NAME] [--json out.json]
API_URL or PORT in the environment set the URL; TENSORFOLD_API_KEY is sent when set.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request

DANSK = ("Forklar, hvordan en hashtabel virker, og hvorfor opslag i gennemsnit er hurtige. "
         "Skriv fire almindelige afsnit på dansk.")
PROSE = "Explain how a hash map works and why lookups are fast on average. Write four plain paragraphs."
CODE = ("Write a Python class LRUCache with get and put in O(1) using an OrderedDict, with type hints. "
        "Then write five pytest tests for it. Output only code.")
SHAPES = {"dansk": DANSK, "prose": PROSE, "code": CODE}
MAX_SHARED_PREFIX = 64   # a fresh prompt that reuses more than this hit the prompt cache

WORDS = ("skov hav by vej lys tid ord bog dag nat vind sol regn sne fjeld å bro mark hus tag dør vindue kat hund fugl "
         "fisk sten træ blad rod frø korn brød mælk ost æble pære kirke skole marked havn skib tog bil cykel sti "
         "river forest city street light time word book day night wind sun rain snow mountain bridge field house "
         "roof door window cat dog bird fish stone tree leaf root seed grain bread milk cheese apple pear church "
         "school market harbour ship train car bicycle path").split()


def post(url, body, timeout=3600):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if os.environ.get("TENSORFOLD_API_KEY"):
        req.add_header("Authorization", "Bearer " + os.environ["TENSORFOLD_API_KEY"])
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d: %s" % (e.code, e.read().decode(errors="replace")[:500]))
    return reply, time.monotonic() - t0


def stats(reply, wall):
    tf = reply.get("tensorfold") or {}
    usage = reply.get("usage") or {}
    n = usage.get("completion_tokens") or 0
    s = reply.get("speculative") or tf.get("speculative") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = tf.get("cached")
    return {"tps": float(tf.get("tokens_per_second") or 0) or (n / wall if wall else 0),
            "ttft": float(tf.get("time_to_first_token") or 0), "tokens": n,
            "per_round": float(s.get("tokens_per_round") or 0),
            "prompt": usage.get("prompt_tokens") or 0, "cached": cached,
            "prefill_s": float(tf.get("prefill_seconds") or tf.get("prefill_s") or 0), "wall": wall}


def decode_one(url, model, prompt, tokens, temperature, seed):
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    if temperature > 0:
        body["seed"] = seed
    return stats(*post(url, body))


TOKENS_A_WORD = 1.28   # Gemma 4's tokenizer on these words and their punctuation (1,000 words: 1,305 tokens)


def text_of(tokens, seed):
    rng = random.Random(seed)
    words = []
    tokens = int(tokens / TOKENS_A_WORD)
    while len(words) < tokens:
        n = rng.randint(6, 14)
        words += [rng.choice(WORDS) for _ in range(n)]
        words[-1] += "."
    return " ".join(words[:tokens])


def prefill_one(url, model, tokens, seed):
    body = {"model": model, "max_tokens": 8, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text_of(tokens, seed) + "\n\nOpsummer teksten ovenfor i én sætning."}]}
    return stats(*post(url, body))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("API_URL") or "http://127.0.0.1:%s" % os.environ.get("PORT", "8890"))
    ap.add_argument("--streams", default="1,2,4")
    ap.add_argument("--shapes", default="dansk,prose,code")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sizes", default="1024,2048,4096,8000")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", help="also write the results to this file")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    rows = []

    if not a.prefill_only:
        print("%s %s  decode, T=%g, %d tokens a reply, median of %d" % (model, a.label, a.temperature, a.tokens, a.reps))
        print("%-6s %7s %13s %13s %8s %10s" % ("shape", "streams", "aggregate", "per request", "TTFT s", "tok/round"))
        for shape in a.shapes.split(","):
            prompt = SHAPES[shape]
            decode_one(url, model, prompt, 32, a.temperature, 0)   # warm: graphs, the prompt cache
            for n in [int(x) for x in a.streams.split(",")]:
                aggs, pers, ttfts, rounds = [], [], [], []
                for rep in range(a.reps):
                    with cf.ThreadPoolExecutor(n) as ex:
                        res = list(ex.map(lambda i: decode_one(url, model, prompt, a.tokens, a.temperature, 1000 * rep + i), range(n)))
                    aggs.append(sum(r["tps"] for r in res))
                    pers.append(statistics.mean(r["tps"] for r in res))
                    ttfts.append(statistics.mean(r["ttft"] for r in res))
                    rounds.append(statistics.mean(r["per_round"] for r in res))
                row = {"kind": "decode", "label": a.label, "shape": shape, "streams": n, "temperature": a.temperature,
                       "tokens": a.tokens, "aggregate": statistics.median(aggs), "per_request": statistics.median(pers),
                       "ttft": statistics.median(ttfts), "tokens_per_round": statistics.median(rounds)}
                rows.append(row)
                print("%-6s %7d %9.1f t/s %9.1f t/s %8.2f %10.2f" % (shape, n, row["aggregate"], row["per_request"],
                                                                     row["ttft"], row["tokens_per_round"]), flush=True)

    if not a.decode_only:
        print("\n%s %s  cold prefill, median of %d fresh prompts" % (model, a.label, a.reps))
        seed = int(time.time())
        for size in [int(x) for x in a.sizes.split(",")]:
            rs = [prefill_one(url, model, size, seed + size + 7919 * k) for k in range(a.reps)]
            if any(r["cached"] is None or not r["prefill_s"] for r in rs):
                raise SystemExit("the reply carries no cached-token count or prefill seconds: cannot time prefill")
            hits = [r["cached"] for r in rs if r["cached"] > MAX_SHARED_PREFIX]
            if hits:
                raise SystemExit("prompts resumed %s cached tokens: not cold" % hits)
            r = sorted(rs, key=lambda x: x["prefill_s"])[len(rs) // 2]
            done = r["prompt"] - r["cached"]
            row = {"kind": "prefill", "label": a.label, "prompt_tokens": r["prompt"], "cached": r["cached"],
                   "prefill_s": r["prefill_s"], "rate": done / r["prefill_s"], "ttft": r["ttft"] or r["wall"]}
            rows.append(row)
            print("prefill %7s tok: %7.2f s  %6.0f tok/s  (%d cached; runs %s s)" % (
                format(r["prompt"], ","), r["prefill_s"], row["rate"], r["cached"],
                ", ".join("%.2f" % x["prefill_s"] for x in rs)), flush=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

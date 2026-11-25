#!/usr/bin/env python3
"""Correctness checks against a running server. Standard library only.

  1. the served model id
  2. drafted greedy replies equal the same request with "draft": false (drafts may change speed, never tokens)
  3. replies decoded concurrently equal the ones decoded alone
  4. a long prompt (past 1,024 tokens: one PrefixLM chunk) finds the code word hidden in its middle
  5. a tool call comes back as tool_calls (Gemma 4's <|tool_call> markers, parsed by the server)
  6. with --reference work/reference.json (tools/reference.py): each reply against Transformers' BF16 reply. A BF16
     export should match token for token (allow a late divergence from a near-tie); NVFP4 is reported, not judged

Usage: python3 tools/check.py [--url http://127.0.0.1:8890] [--tokens 128] [--reference work/reference.json]
API_URL or PORT in the environment set the URL; TENSORFOLD_API_KEY is sent when set.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS  # noqa: E402


def request(url, path, body=None, timeout=900):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    key = os.environ.get("TENSORFOLD_API_KEY")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d from %s: %s" % (e.code, path, e.read().decode(errors="replace")[:500]))


def chat(url, model, prompt, tokens, draft=True, **extra):
    body = {"model": model, "max_tokens": tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}], **extra}
    if not draft:
        body["draft"] = False
    return request(url, "/v1/chat/completions", body)


def content(reply):
    return reply["choices"][0]["message"].get("content") or ""


def fingerprint(reply):
    """The reply's token hash when the engine reports one, else a hash of its text."""
    tf = reply.get("tensorfold") or {}
    if tf.get("token_sha"):
        return str(tf["token_sha"])
    return hashlib.sha256(content(reply).encode()).hexdigest()[:16]


def spec(reply):
    s = reply.get("speculative") or (reply.get("tensorfold") or {}).get("speculative") or {}
    if not s:
        return ""
    return " (accepted %s of %s drafted)" % (s.get("accepted"), s.get("drafted"))


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("API_URL") or "http://127.0.0.1:%s" % os.environ.get("PORT", "8890"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reference", help="tools/reference.py's JSON")
    ap.add_argument("--skip-tools", action="store_true")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    failed = 0

    model = request(url, "/v1/models")["data"][0]["id"]
    print("model: %s at %s" % (model, url))

    solo, replies = {}, {}
    for name, prompt in PROMPTS:
        d = chat(url, model, prompt, a.tokens)
        s = chat(url, model, prompt, a.tokens, draft=False)
        solo[name], replies[name] = fingerprint(d), d
        ok = fingerprint(d) == fingerprint(s)
        failed += not ok
        print("%-8s drafted == serial: %s, %d prompt tokens%s" % (name, "PASS" if ok else "FAIL", d["usage"]["prompt_tokens"], spec(d)))

    with cf.ThreadPoolExecutor(len(PROMPTS)) as ex:
        futs = {name: ex.submit(chat, url, model, prompt, a.tokens) for name, prompt in PROMPTS}
        for name, f in futs.items():
            ok = fingerprint(f.result()) == solo[name]
            failed += not ok
            print("%-8s concurrent == alone: %s" % (name, "PASS" if ok else "FAIL"))

    long_reply = content(replies["lang"])
    ok = replies["lang"]["usage"]["prompt_tokens"] > 1024 and "BLÅBÆR-47" in long_reply.upper().replace(" ", "")
    failed += not ok
    print("long prompt (%d tokens) finds its code word: %s (%r)" % (replies["lang"]["usage"]["prompt_tokens"], "PASS" if ok else "FAIL", long_reply[:40]))

    if not a.skip_tools:
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Hent aktuelt vejr for en by / Get current weather for a city.",
            "parameters": {"type": "object", "required": ["city"], "properties": {
                "city": {"type": "string", "description": "Bynavn, fx 'Aarhus'"},
                "days": {"type": "integer", "description": "Antal prognosedage"}}}}}]
        r = request(url, "/v1/chat/completions", {
            "model": model, "max_tokens": 256, "temperature": 0, "tools": tools,
            "messages": [{"role": "user", "content": "Hvordan bliver vejret i Aarhus de næste 2 dage?"}]})
        calls = r["choices"][0]["message"].get("tool_calls") or []
        ok = False
        if calls:
            args = json.loads(calls[0]["function"]["arguments"])
            ok = calls[0]["function"]["name"] == "get_weather" and "aarhus" in str(args.get("city", "")).lower()
        failed += not ok
        print("tool call: %s %s" % ("PASS" if ok else "FAIL", json.dumps(calls, ensure_ascii=False)[:200] if calls else repr(content(r)[:200])))

    if a.reference:
        ref = json.load(open(a.reference))
        print("\nagainst Transformers' BF16 replies (%s, %d tokens):" % (ref.get("attn"), ref.get("tokens")))
        exact = 0
        for name, _ in PROMPTS:
            want = ref["replies"][name]["text"]
            got = content(chat(url, model, dict(PROMPTS)[name], ref.get("tokens", a.tokens)))
            n = common_prefix(got, want)
            same = got.strip() == want.strip()
            exact += same
            print("%-8s %s: %d of %d characters agree%s" % (name, "same" if same else "differs", n, len(want),
                                                            "" if same else " (then %r vs %r)" % (got[n:n + 24], want[n:n + 24])))
        print("%d of %d replies identical to the reference" % (exact, len(PROMPTS)))

    print("\n%s" % ("all checks passed" if not failed else "%d check(s) FAILED" % failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

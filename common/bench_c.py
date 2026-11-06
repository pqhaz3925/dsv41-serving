"""Decode speed at low concurrency. bench_c.py <port> <keyfile> [levels=1,2,3,4,5] [max_tokens=2000] [rounds=2]
Streams /v1/chat/completions; per request: tokens / (last token time - first token time). Prints per level:
mean per-stream tok/s and aggregate (sum of tokens / wall time of the decode phase)."""
import json, sys, time, threading, http.client

port, key = int(sys.argv[1]), open(sys.argv[2]).read().strip()
levels = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3,4,5").split(",")]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 2
model = sys.argv[6] if len(sys.argv) > 6 else "heretic"
PROMPTS = [
    "Write a long, detailed essay about the history of the printing press and its effects on European society.",
    "Write a complete Python implementation of a red-black tree with insert, delete, search and in-order iteration, with docstrings and tests.",
    "Explain step by step how TCP congestion control works (slow start, AIMD, fast retransmit, CUBIC, BBR), with worked numeric examples.",
    "Write a Rust HTTP/1.1 server from scratch using only std: request parsing, routing, keep-alive and a thread pool. Full code.",
    "Write a detailed travel guide for two weeks in Japan, day by day, with costs, transport and food recommendations.",
    "Refactor this idea into TypeScript: an event-sourced shopping cart with commands, events, projections and unit tests. Full code.",
]

def one(i, out):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Authorization": "Bearer " + key, "content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        out[i] = ("ERR", r.status, r.read()[:200]); return
    first = last = None; n = 0; usage = None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        ch = d.get("choices") or []
        if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
            t = time.time(); first = first or t; last = t; n += 1
    toks = (usage or {}).get("completion_tokens", n)
    out[i] = (toks, first, last)

for lvl in levels:
    for rd in range(rounds):
        out = {}
        ts = [threading.Thread(target=one, args=(rd * 7 + i, out)) for i in range(lvl)]
        t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
        errs = [v for v in out.values() if v[0] == "ERR"]
        if errs: print(f"c={lvl} round {rd}: errors {errs[:2]}", flush=True); continue
        per = [v[0] / max(v[2] - v[1], 1e-3) for v in out.values()]
        span = max(v[2] for v in out.values()) - min(v[1] for v in out.values())
        agg = sum(v[0] for v in out.values()) / span
        print(f"c={lvl} round {rd}: per-stream mean {sum(per)/len(per):.0f} (min {min(per):.0f}) tok/s, aggregate {agg:.0f} tok/s, tokens {[v[0] for v in out.values()]}", flush=True)

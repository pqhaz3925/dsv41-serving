"""How long a prompt-cache point survives on a loaded replica. For each delay a unique ~40k-token prompt is sent
once, then again after that delay; SGLang's /generate reports cached_tokens. Prints counts only.
Usage: cache_ttl_probe.py <port> [delays...]"""
import json
import sys
import threading
import time
import urllib.request
import uuid

PORT = int(sys.argv[1])
DELAYS = [int(x) for x in sys.argv[2:]] or [10, 30, 60, 120, 240]
KEY = open("/mnt/nvme/ds/kit/api_key").read().strip()
FILLER = " ".join(f"item {i} value {i * 7919 % 10007} note alpha beta gamma." for i in range(4000))


def gen(text):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", method="POST",
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                                 data=json.dumps({"text": text, "sampling_params": {"max_new_tokens": 1}}).encode())
    t = time.time()
    m = json.load(urllib.request.urlopen(req, timeout=900))["meta_info"]
    return m.get("prompt_tokens"), m.get("cached_tokens"), time.time() - t


def run(delay, out):
    text = f"probe {uuid.uuid4().hex}\n" + FILLER
    p, c, d = gen(text)
    sent = time.time()
    time.sleep(max(0, delay - (time.time() - sent)))
    p2, c2, d2 = gen(text)
    out[delay] = (p, c, d, p2, c2, d2, time.time() - sent - d2)


out = {}
th = [threading.Thread(target=run, args=(d, out)) for d in DELAYS]
for t in th:
    t.start()
for t in th:
    t.join()
for d in DELAYS:
    p, c, dur, p2, c2, dur2, gap = out[d]
    print(f"delay {d:>4}s (real gap {gap:5.0f}s): first {p} tok cached {c} ({dur:.1f}s) | repeat cached {c2}/{p2} "
          f"({100 * c2 / p2:.0f}%, {dur2:.1f}s)", flush=True)

"""Where does the prompt cache break in a multi-turn tool loop through /v1/messages?
Sends a real conversation (tool call -> tool result -> tool call ...) straight to one replica and, for every
turn, finds that replica's prefill log line whose new+cached tokens equal the prompt size. Prints counts only,
never model text. Usage: cache_tools_probe.py [port, default 30007] [container, default sg-rep45] [turns, default 4]"""
import json
import random
import subprocess
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30007
CONT = sys.argv[2] if len(sys.argv) > 2 else "sg-rep45"
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
KEY = open("/mnt/nvme/ds/kit/api_key").read().strip()

rnd = random.Random(time.time())
words = [''.join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(3, 9))) for _ in range(3000)]
filler = " ".join(rnd.choice(words) for _ in range(9000))  # unique per run, ~20k tokens, no earlier cache

TOOLS = [{"name": "read_file", "description": "Read a file from the project.",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
SYSTEM = "You are a coding agent. Always use the read_file tool to look at files before answering. Notes:\n" + filler
messages = [{"role": "user", "content": "Read src/a.py, then src/b.py, then src/c.py, then src/d.py, one tool call at a time."}]


def call(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def cached_for(prompt, prev, since):
    """Exact match on a single-request batch; otherwise (batched with other agents) the batch whose cached
    count lies between 0.9 x the previous turn's prompt and this prompt, the size a hit on our prefix has."""
    out = subprocess.run(["sudo", "docker", "logs", "--since", since, CONT], capture_output=True, text=True)
    exact, near = None, None
    for l in (out.stdout + out.stderr).splitlines():
        if "TP0]" not in l or "Prefill batch" not in l:
            continue
        try:
            seqs = int(l.split("#new-seq: ")[1].split(",")[0])
            new = int(l.split("#new-token: ")[1].split(",")[0])
            cached = int(l.split("#cached-token: ")[1].split(",")[0])
        except (IndexError, ValueError):
            continue
        if seqs == 1 and abs(new + cached - prompt) <= 2:
            exact = (new, cached, "exact")
        elif prev and 0.9 * prev <= cached <= prompt:
            near = (new, cached, f"batched x{seqs}")
    return exact or near


prev = 0
for turn in range(1, TURNS + 1):
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 2))
    body = {"model": "heretic", "max_tokens": 4000, "system": SYSTEM, "tools": TOOLS, "messages": messages,
            "thinking": {"type": "enabled", "budget_tokens": 2000}, "temperature": 1.0, "top_p": 0.95}
    r = call(body)
    prompt = r["usage"]["input_tokens"]
    time.sleep(2)
    hit = cached_for(prompt, prev, since)
    kinds = [b.get("type") for b in r.get("content", [])]
    if hit:
        print(f"turn {turn}: prompt {prompt}, cached {hit[1]} ({100 * hit[1] / prompt:.1f}%) [{hit[2]}] | reply blocks {kinds}", flush=True)
    else:
        print(f"turn {turn}: prompt {prompt}, no batch with a hit near {prev} tokens (miss or not found) | reply blocks {kinds}", flush=True)
    prev = prompt
    uses = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
    if not uses:
        print("no tool call, stopping", flush=True)
        break
    messages.append({"role": "assistant", "content": r["content"]})  # exactly as returned, the way Claude Code does
    messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"],
                                                  "content": f"# file {turn}\n" + " ".join(rnd.choice(words) for _ in range(300))}
                                                 for u in uses]})

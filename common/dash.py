"""Small realtime dashboard for the SGLang replicas on this box.
Follows each sg-* container's log live (docker logs -f) and rocm-smi every 2 s, and pushes the state to the page
once a second over Server-Sent Events. Serves on 127.0.0.1:PORT only (open it through an SSH tunnel, there is no auth).
Usage: dash.py [port, default 8090]"""
import collections
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
RATE_WINDOW = 10      # seconds over which prefill rate and cache hit are averaged
DECODE_WINDOW = 5     # seconds of decode lines that count as "now"
LOCK = threading.Lock()
REPS = {}             # name -> live counters, filled by the log followers
GPUS = {}
HISTORY = collections.deque(maxlen=900)  # 15 min at 1 s
NUM = {k: re.compile(p) for k, p in {
    "running": r"#running-req: (\d+)", "queue": r"#queue-req: (\d+)",
    "kv": r"full token usage: ([\d.]+)", "kv2": r"token usage: ([\d.]+)",
    "new": r"#new-token: (\d+)", "cached": r"#cached-token: (\d+)",
    "tps": r"gen throughput \(token/s\): ([\d.]+)", "accept": r"accept len: ([\d.]+)"}.items()}


def num(key, line):
    m = NUM[key].search(line)
    return float(m.group(1)) if m else None


def gpus_of(name):
    gp = re.search(r"(?:rep|vl-)(\d+)", name)
    return [int(c) for c in gp.group(1)] if gp else []


RANK = re.compile(r"(DP(\d+) )?TP(\d+)\]")


def follow(name):
    """SGLang: one thread per container, reads its log as it is written. With DP attention every DP rank logs
    its own batches (DP0 TP0, DP1 TP1); their running/queue/KV are summed."""
    rec = {"name": name, "engine": "sglang", "gpus": gpus_of(name), "running": 0, "queue": 0, "kv": 0.0,
           "ranks": {}, "prefills": collections.deque(), "decodes": collections.deque(), "last": 0.0}
    with LOCK:
        REPS[name] = rec
    proc = subprocess.Popen(["sudo", "docker", "logs", "-f", "--since", f"{RATE_WINDOW}s", name],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        m = RANK.search(line)
        if not m or (m.group(1) is None and m.group(3) != "0"):
            continue
        pre, dec = "Prefill batch" in line, "Decode batch" in line
        if not (pre or dec):
            continue
        now = time.time()
        with LOCK:
            rk = rec["ranks"].setdefault(m.group(2) or "0", {"running": 0, "queue": 0, "kv": 0.0})
            r, q = num("running", line), num("queue", line)
            if r is not None:
                rk["running"] = int(r)
            if q is not None:
                rk["queue"] = int(q)
            kv = num("kv", line)
            rk["kv"] = kv if kv is not None else (num("kv2", line) or rk["kv"])
            rec["running"] = sum(x["running"] for x in rec["ranks"].values())
            rec["queue"] = sum(x["queue"] for x in rec["ranks"].values())
            rec["kv"] = max(x["kv"] for x in rec["ranks"].values())
            rec["last"] = now
            if pre:
                rec["prefills"].append((now, num("new", line) or 0, num("cached", line) or 0))
            else:
                t = num("tps", line)
                if t is not None and t >= 30:  # lower values are idle gaps
                    rec["decodes"].append((now, t, num("accept", line)))
    proc.wait()
    with LOCK:
        REPS.pop(name, None)


VMET = {"running": "vllm:num_requests_running", "queue": "vllm:num_requests_waiting",
        "kv": "vllm:kv_cache_usage_perc", "prompt": "vllm:prompt_tokens_total", "gen": "vllm:generation_tokens_total",
        "hits": "vllm:prefix_cache_hits_total", "queries": "vllm:prefix_cache_queries_total",
        "drafts": "vllm:spec_decode_num_drafts_total", "accepted": "vllm:spec_decode_num_accepted_tokens_total"}
KEY = open("/mnt/nvme/ds/kit/api_key").read().strip() if __import__("os").path.exists("/mnt/nvme/ds/kit/api_key") else ""


def vllm_port(name):
    out = subprocess.run(["sudo", "docker", "inspect", "-f", "{{json .Args}}", name],
                         capture_output=True, text=True, timeout=20).stdout
    args = json.loads(out or "[]")
    return int(args[args.index("--port") + 1]) if "--port" in args else None


def poll_vllm(name):
    """vLLM: reads /metrics once a second; rates come from the change of its counters over the window."""
    import urllib.request
    port = vllm_port(name)
    if not port:
        return
    rec = {"name": name, "engine": "vllm", "gpus": gpus_of(name), "running": 0, "queue": 0, "kv": 0.0,
           "samples": collections.deque(), "last": 0.0}
    with LOCK:
        REPS[name] = rec
    fails = 0
    while fails < 30:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {KEY}"})
            text = urllib.request.urlopen(req, timeout=5).read().decode()
            vals = {}
            for line in text.splitlines():
                if line.startswith("vllm:"):
                    metric = line.split("{", 1)[0].split(" ", 1)[0]
                    for k, mname in VMET.items():
                        if metric == mname:
                            vals[k] = vals.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
            now = time.time()
            with LOCK:
                rec["running"] = int(vals.get("running", 0))
                rec["queue"] = int(vals.get("queue", 0))
                rec["kv"] = vals.get("kv", 0.0)
                rec["last"] = now
                rec["samples"].append((now, vals))
                while rec["samples"] and rec["samples"][0][0] < now - RATE_WINDOW - 1:
                    rec["samples"].popleft()
            fails = 0
        except Exception:
            fails += 1
        time.sleep(1)
    with LOCK:
        REPS.pop(name, None)


def watch_containers():
    while True:
        try:
            names = subprocess.run(["sudo", "docker", "ps", "--format", "{{.Names}}"],
                                   capture_output=True, text=True, timeout=20).stdout.split()
            for n in names:
                if n.startswith("sg-") and n not in REPS:
                    threading.Thread(target=follow, args=(n,), daemon=True).start()
                elif n.startswith("vl-") and n not in REPS:
                    threading.Thread(target=poll_vllm, args=(n,), daemon=True).start()
        except Exception as e:
            print("docker ps error:", type(e).__name__, e, flush=True)
        time.sleep(5)


def watch_gpus():
    while True:
        try:
            out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--json"],
                                 capture_output=True, text=True, timeout=20).stdout
            res = {}
            for card, v in json.loads(out).items():
                if card.startswith("card"):
                    res[int(card[4:])] = {"use": int(float(v.get("GPU use (%)", 0) or 0)),
                                          "vram": int(float(v.get("GPU Memory Allocated (VRAM%)", 0) or 0))}
            with LOCK:
                GPUS.clear()
                GPUS.update(res)
        except Exception as e:
            print("rocm-smi error:", type(e).__name__, e, flush=True)
        time.sleep(2)


def snapshot():
    """Current state of every replica, computed from the rolling windows."""
    now = time.time()
    reps = []
    with LOCK:
        for name in sorted(REPS, key=lambda n: (gpus_of(n) or [99])[0]):
            rec = REPS[name]
            if rec["engine"] == "vllm":
                reps.append(vllm_snapshot(rec, now))
                continue
            while rec["prefills"] and rec["prefills"][0][0] < now - RATE_WINDOW:
                rec["prefills"].popleft()
            while rec["decodes"] and rec["decodes"][0][0] < now - max(RATE_WINDOW, DECODE_WINDOW):
                rec["decodes"].popleft()
            new = sum(p[1] for p in rec["prefills"])
            cached = sum(p[2] for p in rec["prefills"])
            recent = [d for d in rec["decodes"] if d[0] >= now - DECODE_WINDOW]
            acc = [d[2] for d in recent if d[2] is not None]
            idle = now - rec["last"] > DECODE_WINDOW
            reps.append({"name": name, "engine": "sglang", "gpus": rec["gpus"], "queue": rec["queue"], "kv": rec["kv"],
                         "running": 0 if idle and not recent else rec["running"],
                         "decode": round(sum(d[1] for d in recent) / len(recent)) if recent else 0,
                         "prefill": round(new / RATE_WINDOW),
                         "cached": round(100 * cached / (new + cached)) if new + cached else None,
                         "accept": round(sum(acc) / len(acc), 2) if acc else None})
        gpus = dict(GPUS)
    return {"ts": now, "replicas": reps, "gpus": gpus}


def vllm_snapshot(rec, now):
    """Rates from counter deltas: prefill = prompt tokens not served from the prefix cache, over RATE_WINDOW;
    decode over DECODE_WINDOW; accept len = 1 + accepted draft tokens per draft step."""
    s = rec["samples"]
    out = {"name": rec["name"], "engine": "vllm", "gpus": rec["gpus"], "queue": rec["queue"], "kv": rec["kv"],
           "running": rec["running"], "decode": 0, "prefill": 0, "cached": None, "accept": None}
    if len(s) < 2:
        return out
    t1, v1 = s[-1]
    t0, v0 = s[0]
    td, vd = next(((t, v) for t, v in s if t >= t1 - DECODE_WINDOW), s[0])
    d = lambda a, b, k: max(0.0, a.get(k, 0) - b.get(k, 0))
    if t1 > t0:
        q, h = d(v1, v0, "queries"), d(v1, v0, "hits")
        out["prefill"] = round(max(0.0, d(v1, v0, "prompt") - h) / (t1 - t0))
        out["cached"] = round(100 * h / q) if q else None
    if t1 > td:
        out["decode"] = round(d(v1, vd, "gen") / (t1 - td))
        dr = d(v1, v0, "drafts")
        out["accept"] = round(1 + d(v1, v0, "accepted") / dr, 2) if dr else None
    return out


def ticker():
    while True:
        s = snapshot()
        point = {"ts": s["ts"], "decode": sum(r["decode"] for r in s["replicas"]),
                 "prefill": sum(r["prefill"] for r in s["replicas"]),
                 "queue": sum(r["queue"] for r in s["replicas"])}
        with LOCK:
            HISTORY.append(point)
            LATEST.update(s, point=point)
            TICK.notify_all()
        time.sleep(1)


LATEST = {}
TICK = threading.Condition(LOCK)

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSeek load</title><style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1d21;--mute:#6b7280;--line:#e5e7eb;--acc:#2563eb;--pre:#7c3aed;--warn:#d97706;--bad:#dc2626;--ok:#16a34a}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--mute:#9aa3b2;--line:#262b36;--acc:#60a5fa;--pre:#a78bfa;--warn:#f59e0b;--bad:#f87171;--ok:#4ade80}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 4px;display:flex;align-items:center;gap:8px}.sub{color:var(--mute);margin-bottom:16px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--bad)}.dot.on{background:var(--ok)}
.tot{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px}
.k{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.k b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.k span{color:var(--mute);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
.r{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.r h2{font-size:15px;margin:0 0 8px;display:flex;justify-content:space-between}
.row{display:flex;justify-content:space-between;padding:2px 0;font-variant-numeric:tabular-nums}.row span{color:var(--mute)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:2px 0 6px}.bar i{display:block;height:100%;background:var(--acc);transition:width .4s}
canvas{width:100%;height:140px;background:var(--card);border:1px solid var(--line);border-radius:10px;margin-top:14px}
.q0{color:var(--ok)}.q1{color:var(--warn)}.q2{color:var(--bad)}
</style></head><body>
<h1><span class="dot" id="dot"></span>DeepSeek load</h1><div class="sub" id="sub">connecting…</div>
<div class="tot" id="tot"></div><div class="grid" id="grid"></div>
<canvas id="ch"></canvas><div class="sub">last 15 min, 1 s steps: <span style="color:var(--acc)">decode tok/s</span> · <span style="color:var(--pre)">prefill tok/s</span> · <span style="color:var(--warn)">queued requests</span></div>
<script>
const $=id=>document.getElementById(id), css=v=>getComputedStyle(document.documentElement).getPropertyValue(v);
const qc=q=>q==0?'q0':q<8?'q1':'q2', H=[];
function k(v,l){return `<div class="k"><b>${v}</b><span>${l}</span></div>`}
function row(l,v,c=''){return `<div class="row"><span>${l}</span><b class="${c}">${v}</b></div>`}
function bar(p){return `<div class="bar"><i style="width:${Math.min(100,p)}%"></i></div>`}
function draw(){const c=$('ch'),d=devicePixelRatio||1,W=c.clientWidth,Ht=c.clientHeight;c.width=W*d;c.height=Ht*d;
 const x=c.getContext('2d');x.scale(d,d);if(H.length<2)return;
 const t0=H[0].ts,t1=H[H.length-1].ts;let y=14;
 for(const [key,col] of [['decode','--acc'],['prefill','--pre'],['queue','--warn']]){const m=Math.max(1,...H.map(p=>p[key]));
  x.strokeStyle=css(col);x.lineWidth=1.5;x.beginPath();
  H.forEach((p,i)=>{const X=6+(W-12)*(p.ts-t0)/(t1-t0||1),Y=Ht-8-(Ht-48)*p[key]/m;i?x.lineTo(X,Y):x.moveTo(X,Y)});x.stroke();
  x.fillStyle=css(col);x.font='11px system-ui';x.fillText(`${key} max ${Math.round(m).toLocaleString()}`,8,y);y+=14}}
function render(s){const R=s.replicas,sum=f=>R.reduce((a,r)=>a+f(r),0),cached=R.filter(r=>r.cached!=null);
 $('sub').textContent=`${R.length} replicas · ${new Date(s.ts*1000).toLocaleTimeString()}`;
 $('tot').innerHTML=k(sum(r=>r.running),'generating')+k(`<span class="${qc(sum(r=>r.queue))}">${sum(r=>r.queue)}</span>`,'queued')
  +k(sum(r=>r.decode).toLocaleString(),'decode tok/s (sum)')+k(sum(r=>r.prefill).toLocaleString(),'prefill tok/s (10 s)')
  +k(cached.length?Math.round(cached.reduce((a,r)=>a+r.cached,0)/cached.length)+'%':'–','prompt cache hit');
 $('grid').innerHTML=R.map(r=>{const g=r.gpus.map(i=>s.gpus[i]).filter(Boolean);
  return `<div class="r"><h2>${r.name} <small style="color:var(--mute);font-weight:400">${r.engine}</small><span style="color:var(--mute);font-weight:400">GPU ${r.gpus.join(',')}</span></h2>
  ${row('generating',r.running)}${row('queued',r.queue,qc(r.queue))}
  ${row('decode tok/s',r.decode)}${row('accept len',r.accept??'–')}
  ${row('prefill tok/s',r.prefill.toLocaleString())}${row('cache hit',r.cached==null?'–':r.cached+'%')}
  ${row('KV used',Math.round(r.kv*100)+'%')}${bar(r.kv*100)}
  ${g.map((x,i)=>row('GPU '+r.gpus[i]+' busy',x.use+'%')+bar(x.use)).join('')}</div>`}).join('');
 draw()}
const es=new EventSource('api/stream');
es.addEventListener('history',e=>{H.length=0;H.push(...JSON.parse(e.data))});
es.onmessage=e=>{const s=JSON.parse(e.data);H.push(s.point);while(H.length>900)H.shift();render(s)};
es.onopen=()=>$('dot').classList.add('on');
es.onerror=()=>{$('dot').classList.remove('on');$('sub').textContent='disconnected, reconnecting…'};
addEventListener('resize',draw);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with LOCK:
                    hist = json.dumps(list(HISTORY))
                self.wfile.write(f"event: history\ndata: {hist}\n\n".encode())
                self.wfile.flush()
                while True:
                    with TICK:
                        TICK.wait(timeout=5)
                        data = json.dumps(LATEST)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True


for target in (watch_containers, watch_gpus, ticker):
    threading.Thread(target=target, daemon=True).start()
print(f"dashboard on 127.0.0.1:{PORT}", flush=True)
Server(("127.0.0.1", PORT), Handler).serve_forever()

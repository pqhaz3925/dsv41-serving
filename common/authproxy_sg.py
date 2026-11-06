"""Forward requests to the local vLLM server, turning an Anthropic-style `x-api-key`
header into the `Authorization: Bearer` header vLLM expects, and filling in sampling
settings the client did not send. Responses are streamed through unchanged. Nothing is logged.
Optional key files: the server's key, and a priority key. A request with the priority key is sent upstream
with the server's key and SGLang's `x-override-priority: 10` header (it takes effect only when the server
runs with --enable-priority-scheduling and SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1).
Upstream: one port, a comma list, or @file with one port per line (re-read when it changes). With several upstreams each
client session (metadata.user_id, else the first message) sticks to the replica it started on, so its prompt cache stays
warm; new sessions go to the replica with the fewest requests in flight; a refused connection falls through to the next.
Usage: authproxy_sg.py <listen_port> <upstream_port|p1,p2|@file> [key_file priority_key_file]"""
import collections
import http.client
import json
import os
import select
import socket
import zlib
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN, UPSTREAM_SPEC = int(sys.argv[1]), sys.argv[2]
LOCK = threading.Lock()
ACTIVE = collections.Counter()           # requests in flight per upstream port
STICKY = collections.OrderedDict()       # session key -> port
_cache = {"mtime": None, "ports": []}


def upstreams():
    if not UPSTREAM_SPEC.startswith("@"):
        return [int(x) for x in UPSTREAM_SPEC.split(",") if x.strip()]
    path = UPSTREAM_SPEC[1:]
    try:
        mtime = os.stat(path).st_mtime
        if mtime != _cache["mtime"]:
            ports = [int(x) for x in open(path).read().split() if x.strip().isdigit()]
            if ports:
                _cache.update(mtime=mtime, ports=ports)
    except OSError:
        pass
    return list(_cache["ports"])


PREFIXES = collections.defaultdict(collections.OrderedDict)  # port -> prefix hash -> None (LRU)
PREFIX_CAP = 200000


def prefix_hashes(request):
    """Cumulative hashes of the conversation: system+tools, then each message in turn. A replica that saw
    hash k already holds that prefix in its prompt cache."""
    h = zlib.crc32(json.dumps([request.get("system"), request.get("tools")], sort_keys=True).encode())
    out = [h]
    for m in request.get("messages") or []:
        h = zlib.crc32(json.dumps(m, sort_keys=True).encode(), h)
        out.append(h)
    return out


def order_by_prefix(hashes):
    """Cache-aware choice: the replica holding the longest prefix of this conversation, unless it is much
    busier than the least busy one; then the least busy. The chosen replica learns all the prefixes."""
    ports = upstreams()
    with LOCK:
        least = min(ports, key=lambda p: ACTIVE[p])
        best, depth = least, 0
        for p in ports:
            seen = PREFIXES[p]
            d = 0
            for i in range(len(hashes) - 1, -1, -1):  # longest first
                if hashes[i] in seen:
                    d = i + 1
                    break
            if d > depth or (d == depth and d and ACTIVE[p] < ACTIVE[best]):
                best, depth = p, d
        spilled = depth > 0 and ACTIVE[best] > ACTIVE[least] + SPILL
        if depth == 0 or spilled:
            best = least
        seen = PREFIXES[best]
        known = next((i + 1 for i in range(len(hashes) - 1, -1, -1) if hashes[i] in seen), 0)
        for x in hashes:
            seen[x] = None
            seen.move_to_end(x)
        while len(seen) > PREFIX_CAP:
            seen.popitem(last=False)
        # Only counts, no content: how much of this conversation the chosen replica has already seen.
        print(f"route :{best} known {known}/{len(hashes)}{' spilled' if spilled else ''}", flush=True)
        ACTIVE[best] += 1  # reserved here, so requests arriving together see each other's load
    return [best] + [p for p in ports if p != best]


def order(session):
    """Upstream ports to try, best first."""
    ports = upstreams()
    with LOCK:
        if session is not None:
            port = STICKY.get(session)
            # A session whose replica is much busier than the least busy one moves there: one prompt-cache
            # miss is cheaper than waiting behind a queue of long prefills.
            if port in ports and ACTIVE[port] > min(ACTIVE[p] for p in ports) + SPILL:
                port = None
            if port not in ports:
                bound = collections.Counter(STICKY.values())
                port = min(ports, key=lambda p: (ACTIVE[p], bound[p], p))
                STICKY[session] = port
            STICKY.move_to_end(session)
            while len(STICKY) > 20000:
                STICKY.popitem(last=False)
        else:
            port = min(ports, key=lambda p: ACTIVE[p])
        ACTIVE[port] += 1  # reserved here, see order_by_prefix
    return [port] + [p for p in ports if p != port]
KEY = open(sys.argv[3]).read().strip() if len(sys.argv) > 4 else None
PRIORITY_KEY = open(sys.argv[4]).read().strip() if len(sys.argv) > 4 else None
SKIP_REQUEST = {"host", "content-length", "connection", "accept-encoding"}
SKIP_RESPONSE = {"transfer-encoding", "content-length", "connection"}
# DeepSeek recommends temperature 1.0 and top_p 0.95 for this model; lower temperatures make it loop.
SAMPLING_DEFAULTS = {"temperature": 1.0, "top_p": 0.95}
# A session moves off its replica on an imbalance (busier than the least busy one by more than SPILL);
# SPILL=0 (few concurrent streams) balances strictly and keeps affinity only while loads are equal;
# raise it when many agents share a long system prompt and cache hits matter more than queueing.
# under even overload moving would just trade a queue for a prompt-cache miss.
SPILL = int(os.environ.get("SPILL", "0"))
STICKY_FILE = os.environ.get("STICKY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sticky.json"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in SKIP_REQUEST}
        session = hashes = None
        if body and self.path.rstrip("/").endswith(("/messages", "/chat/completions")):
            # The model's generation_config.json makes vLLM default to temperature 1.0 with
            # repetition_penalty 1.1, and with those its output falls apart after about a
            # thousand tokens. Clients that send no sampling settings get sane ones here.
            try:
                request = json.loads(body)
                # Only the sampling numbers a client sent are printed, never any content.
                print("client sent:", {name: request.get(name) for name in ("temperature", "top_p", "top_k", "max_tokens")},
                      "| thinking:", (request.get("thinking") or {}).get("type"), flush=True)
                meta = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
                first = (request.get("messages") or [None])[0]
                # Claude Code sends the same user_id for its main thread and every subagent; the first
                # message tells those conversations apart, so each one gets its own replica.
                session = f"{meta.get('user_id')}|{zlib.crc32(json.dumps(first)[:4000].encode())}"
                # Claude Code's first system block is a per-session billing header. The DeepSeek template puts the
                # system prompt before the tools, so this one block made every agent's prompt differ from token 0
                # and no two sessions shared a cached prefix. It means nothing to the model: drop it.
                if isinstance(request.get("system"), list):
                    kept = [b for b in request["system"] if not (isinstance(b, dict)
                            and str(b.get("text", "")).startswith("x-anthropic-billing-header"))]
                    if len(kept) != len(request["system"]):
                        request["system"] = kept
                        print("dropped billing header", flush=True)
                hashes = prefix_hashes(request)
                # Only lengths and checksums of the system blocks, never their text: shows which blocks are
                # shared by every agent and which differ per session.
                sysb = request.get("system")
                sysb = [sysb] if isinstance(sysb, str) else (sysb or [])
                print("sysblocks", " ".join(
                    f"{len(t)}:{zlib.crc32(t.encode()):08x}" for t in
                    ((b if isinstance(b, str) else str(b.get("text", ""))) for b in sysb if isinstance(b, (str, dict)))),
                    "| tools", zlib.crc32(json.dumps(request.get("tools"), sort_keys=True).encode()) & 0xffffffff, flush=True)
                for name, value in SAMPLING_DEFAULTS.items():
                    request.setdefault(name, value)
                # SGLang opens the reasoning block by default but only separates it from the
                # answer when the request asks for thinking; without this the reasoning and the
                # closing tag end up in the text.
                if self.path.rstrip("/").endswith("/messages"):
                    # The budget must stay below max_tokens or SGLang answers 400 (Claude Code's short side
                    # requests ask for a few hundred tokens); tiny requests get thinking off instead.
                    limit = int(request.get("max_tokens") or 32000)
                    if limit > 2048:
                        request.setdefault("thinking", {"type": "enabled", "budget_tokens": min(16000, limit - 1024)})
                    else:
                        request.setdefault("thinking", {"type": "disabled"})
                    # DeepSeek Harness sends {"type": "enabled"} with the effort in output_config and
                    # no budget; SGLang rejects that, so give it the whole output as the budget.
                    thinking = request.get("thinking")
                    if isinstance(thinking, dict) and thinking.get("type") == "enabled" and "budget_tokens" not in thinking:
                        thinking["budget_tokens"] = max(1024, int(request.get("max_tokens") or 32000) - 1)
                    # Newer Claude Code sends display values SGLang does not know (400 "Input should be
                    # 'summarized' or 'omitted'"); the field only affects how reasoning is shown.
                    if isinstance(thinking, dict) and thinking.get("display") not in (None, "summarized", "omitted"):
                        thinking.pop("display")
                body = json.dumps(request).encode()
            except ValueError:
                pass
        key = self.headers.get("x-api-key")
        bearer = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        if PRIORITY_KEY and PRIORITY_KEY in (key, bearer):
            headers = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "x-api-key")}
            headers["Authorization"] = f"Bearer {KEY}"
            headers["x-override-priority"] = "10"
        elif key and not any(k.lower() == "authorization" for k in headers):
            headers["Authorization"] = f"Bearer {key}"
        response = upstream = port = None
        attempt = 0
        for port in (order_by_prefix(hashes) if hashes else order(session)):
            # Load is counted from the routing decision, so requests arriving together, and those still
            # queued for prefill (no response headers yet), both count. The first port was reserved there.
            attempt += 1
            if attempt > 1:
                with LOCK:
                    ACTIVE[port] += 1
            upstream = http.client.HTTPConnection("127.0.0.1", port, timeout=7200)
            try:
                upstream.request(self.command, self.path, body=body, headers=headers)
                self.watch_client(upstream, port)
                response = upstream.getresponse()
                break
            except (ConnectionRefusedError, ConnectionResetError, OSError) as error:
                self.stop_watch()
                upstream.close()
                with LOCK:
                    ACTIVE[port] -= 1
                if self.client_gone:  # the watcher aborted it; nobody is waiting for an answer
                    self.close_connection = True
                    return
                if session is not None:
                    with LOCK:
                        STICKY.pop(session, None)
                print(f"upstream :{port} failed ({type(error).__name__}), trying next", flush=True)
                response = None
        if response is None:
            self.send_error(502, "no upstream answered")
            self.close_connection = True
            return
        try:
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in SKIP_RESPONSE:
                    self.send_header(name, value)
            # No length is known for a stream: the body ends when the connection closes.
            self.send_header("Connection", "close")
            self.end_headers()
            if response.status >= 400:
                data = response.read()
                self.wfile.write(data)
                try:  # only the error type and message, never request content
                    err = json.loads(data)
                    err = err.get("error", err) if isinstance(err, dict) else {}
                    print(f"upstream :{port} {response.status} {self.path}: {str(err.get('type'))[:40]} {str(err.get('message'))[:200]}", flush=True)
                except ValueError:
                    print(f"upstream :{port} {response.status} {self.path} (non-JSON body)", flush=True)
            else:
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.stop_watch()
            with LOCK:
                ACTIVE[port] -= 1
            upstream.close()
            self.close_connection = True

    client_gone = False
    done = None

    def stop_watch(self):
        if self.done is not None:
            self.done.set()

    def watch_client(self, upstream, port):
        """While a request is queued in SGLang nothing flows back, so a client that gives up (Claude Code
        times out and retries) goes unnoticed and every retry leaves a dead copy in the queue. A watcher
        thread shuts the upstream socket when the client disconnects; SGLang then aborts the request."""
        self.done = threading.Event()
        client, sock = self.connection, upstream.sock

        def watch():
            while not self.done.is_set():
                try:
                    ready, _, _ = select.select([client], [], [], 2)
                    if ready and not client.recv(1, socket.MSG_PEEK):
                        break
                    if ready:  # the client sent more bytes; nothing to do but avoid spinning
                        self.done.wait(2)
                except (OSError, ValueError):
                    break
            else:
                return
            if self.done.is_set():  # the answer was already delivered; the client just closed after it
                return
            self.client_gone = True
            print(f"client left during request on :{port}; aborting upstream", flush=True)
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        threading.Thread(target=watch, daemon=True).start()

    do_GET = do_POST = forward

    def log_message(self, *args):
        pass


class Server(ThreadingHTTPServer):
    allow_reuse_port = True  # a replacement can bind the same port while this one drains
    request_queue_size = 1024  # Python's default listen backlog is 5: bursts of agents got ConnectionRefused
    daemon_threads = False  # server_close() then waits for the requests in flight


def save_sticky():
    """Session -> replica map survives proxy restarts, so a restart does not throw away warm prompt caches."""
    while True:
        threading.Event().wait(30)
        with LOCK:
            data = json.dumps(list(STICKY.items()))
        tmp = STICKY_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write(data)
        os.replace(tmp, STICKY_FILE)


try:
    with open(STICKY_FILE) as f:
        STICKY.update((k, int(v)) for k, v in json.load(f))
except (OSError, ValueError):
    pass
threading.Thread(target=save_sticky, daemon=True).start()
server = Server(("127.0.0.1", LISTEN), Handler)
signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
print(f"listening on :{LISTEN} -> {upstreams()}, {len(STICKY)} sticky sessions loaded", flush=True)
server.serve_forever()
print("draining: no new requests, finishing the ones in flight", flush=True)
server.server_close()
print("drained, exiting", flush=True)

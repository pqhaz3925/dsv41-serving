# DeepSeek-V4.1-Flash-1017 serving receipts: 8x B300 and 8x MI355X

What we measured serving DeepSeek-V4.1-Flash-1017 (our abliterated build
[`pqhaz/DeepSeek-V4.1-Flash-1017-heretic`](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic), same
architecture and shapes as the original) to **Claude Code agents**: long contexts (often 100k–300k tokens), tool
loops, thinking on, through an Anthropic `/v1/messages` front end. Scripts in this repo are the ones that ran.

Short version:

- **Agent load (tens of concurrent Claude Code agents):** vLLM 4×TP2 on MI355X held 91–100 % prefix-cache hit and no
  queue; SGLang on the same traffic held 4–40 % and queues of 10–30. Most of that gap was one header (see "Gotchas").
- **Few streams on B300 (1–5 at once):** vLLM TP4 with the AgentX-style flags beat SGLang TP4 at every level,
  ~900 tok/s for one stream.
- **Few streams on MI355X:** the opposite — SGLang TP4 (v0.5.23 image) was ~30 % faster per stream than vLLM TP4.

These are one person's runs on rented boxes, not a controlled benchmark. Per-stream speed depends heavily on how many
DSpark draft tokens get accepted, which depends on the text (code ≫ prose). Spread between identical rounds is large.

## 8x B300 (SXM6, NVLink NV18), 2026-11-06

Two replicas, TP4 each (GPUs 0–3, 4–7). DSpark 5, 1M context, vision on.

| | image | script |
|---|---|---|
| vLLM | `vllm/vllm-openai:cu134-nightly` (0.33.0rc1.dev806) | [`b300/vllm_b300.sh`](b300/vllm_b300.sh) |
| SGLang | `lmsysorg/sglang:dev-dsv41` | [`b300/sg_b300.sh`](b300/sg_b300.sh) |

Per-stream decode, tok/s, `common/bench_c.py` (prose and code prompts, 2000-token answers, thinking off, warm round):

| streams | SGLang TP4+EP4 | SGLang TP4 | **vLLM TP4** | vLLM aggregate |
|---|---|---|---|---|
| 1 | ~450 (live median) | — | **903** | 903 |
| 2 | 512 | 541 | **672** | 1128 |
| 3 | 471 | 501 | **555** | 1379 |
| 4 | 426 | 415 | **493** | 1390 |
| 5 | 420 | 422 | **477** | 1698 |

EP on/off made no measurable difference for SGLang at these concurrencies.

Live agent traffic on the two vLLM replicas (~1–2 h): 13.4k requests, 2.3 B prompt tokens, 98–99 % prefix-cache
hit, DSpark 4.4–4.5 tokens per step (SGLang on the same traffic: ~3.6), single-stream peaks ~1000 tok/s.

vLLM notes:
- First start takes ~40 min (FlashInfer autotune at every cudagraph size). Copy the replica's cache dir
  (`/root/.cache` → we mount `$DS/jit/<name>`) to the next replica: second start was 18 min.
- vLLM's Rust frontend has no `/v1/messages`; we put a small Anthropic→OpenAI translator in front of each replica.

## 8x MI355X, 2026-10-24 … 11-02

| | image |
|---|---|
| vLLM | `vllm/vllm-openai-rocm:nightly-rocm100-edad8b03c5285a1c12e7e4f83c1af709b319a01b` ([`mi355x/vllm_amd.sh`](mi355x/vllm_amd.sh)) |
| SGLang | `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021`, later `v0.5.23-rocm10-mi35x-20261102` ([`mi355x/replica_amd.sh`](mi355x/replica_amd.sh), DP2 variant [`mi355x/replica_dp2.sh`](mi355x/replica_dp2.sh)) |

**Agent load, 4× vLLM TP2** (1M context needs `engram-config {"cpu_offload":true}` and
`--gpu-memory-utilization 0.92` on two cards):
- One hour on one box: 18.8k requests, 2.52 B prompt tokens (mean ~134k in, ~178 out), **98.8 % prefix-cache hit**,
  DSpark ~4.96 tokens per step, 0 preemptions.
- Whole-box decode peaks (1–10 s windows) on three boxes: 2.4k, 4.1k, 5.5k tok/s; per replica up to ~1.4k.
- Same traffic on 4× SGLang TP2: prefix-cache hit 4–40 %, queues 10–30 per replica; SGLang DP2 did not change
  per-replica prompt throughput (~45k tok/s per 2 cards either way).

**Few streams, 2× TP4, through the proxy, 2000-token answers** (per-stream mean, two rounds):

| streams | SGLang v0.5.23 | vLLM |
|---|---|---|
| 1 | 360–794 | 235–549 |
| 5 | 514–596 | 288–325 |

v0.5.23 was ~10–15 % faster per stream than v0.5.22; the cookbook's low-concurrency flags (PDI 192, lpm, …) added
nothing on top. Earlier single-stream essay numbers on TP4: SGLang 372–415, vLLM 334–387.

## Gotchas that cost us hours

1. **Claude Code's first system block is a per-session `x-anthropic-billing-header`.** The DeepSeek chat template
   puts the system prompt *before* the tools, so that block makes every agent's prompt differ from token ~0 and no two
   sessions share a cached prefix. Drop it in front of the server (done in [`common/authproxy_sg.py`](common/authproxy_sg.py)).
   It means nothing to the model.
2. **Route by conversation prefix, not by session id.** Routers often strip `metadata.user_id`; then all subagents
   collapse into a few "sessions". The proxy hashes system+tools and each message cumulatively and sends a request to
   the replica that has seen the longest prefix.
3. **Python's `ThreadingHTTPServer` listen backlog is 5.** Under bursts of agents the kernel dropped connections
   (`ConnectionRefused` upstream). Set `request_queue_size` (we use 1024).
4. **MI355X: replicas died with `RPC call to sample_tokens timed out`** after the kernel stalled
   `amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu` for >2 min: page cache had eaten all free RAM and khugepaged kept moving
   the pinned Engram pages. Before starting: `transparent_hugepage/defrag=never`, `khugepaged/defrag=0`,
   `vm.min_free_kbytes=16777216`, one `drop_caches`. ROCm images also need containerd on the big disk
   ([`mi355x/dockerroot.sh`](mi355x/dockerroot.sh)).
5. **vLLM `--language-model-only` turns vision off**; the checkpoint has a vision tower and it fits with offload.
6. vLLM's Rust frontend crashes at start on an empty `--api-key` (`invalid type: sequence, expected a string`).
7. SGLang with `--tokenizer-worker-num > 1` refuses `--api-key` ("not supported in multi-tokenizer mode").
8. SGLang's default `--swa-prefix-tails` resolved to 128 here; at agent concurrency use thousands (4096 fit at mem
   0.80 on MI355X TP2, 8192 did not).
9. Don't benchmark on a replica that is serving users — it halves their speed for the duration.

## Files

- `b300/` — vLLM and SGLang launch scripts for B300.
- `mi355x/` — vLLM and SGLang (TP2, DP2) launch scripts for MI355X, containerd-on-NVMe helper.
- `common/authproxy_sg.py` — key check, priority key, sampling defaults, billing-header drop, prefix-aware routing
  over several replicas, client-disconnect abort, sticky map persisted across restarts.
- `common/bench_c.py` — per-stream and aggregate decode at given concurrency levels.
- `common/dash.py` — live dashboard (vLLM `/metrics` or SGLang logs + `rocm-smi`), SSE, 127.0.0.1 only.
- `common/cache_ttl_probe.py`, `common/cache_tools_probe.py` — how long prefix-cache entries live under load, and
  where the cache breaks in a multi-turn tool loop.

Scripts assume the model at `$DS/models/heretic` and the server key in `$K`. MIT.

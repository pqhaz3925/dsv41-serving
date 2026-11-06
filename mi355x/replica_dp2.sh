#!/usr/bin/env bash
# SGLang replica of the heretic in the cookbook's DP2 layout for 32-64 concurrent agent sessions per replica:
# two GPUs, attention data-parallel across both, MoE tensor-parallel. No monokernel (it refuses INDEX_K_PAGE8).
# GPUS (e.g. 0,1), PORT, NAME. Key in $K. MEM (default 0.85), TAILS (default 4096), PDI (default 16), EXTRA. Prints READY/FAIL.
DS=/mnt/nvme/ds; IMG=${IMG:-lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021}; GPUS=${GPUS:-0,1}; PORT=${PORT:-30008}; NAME=${NAME:-sg-rep01}
sudo docker rm -f $NAME >/dev/null 2>&1
sudo docker run -d --name $NAME --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64g --ipc private -p 127.0.0.1:$PORT:$PORT -v $DS/models:/models -e HIP_VISIBLE_DEVICES=$GPUS \
  -v $DS/jitcache/$NAME:/root/.cache \
  -e SGLANG_DEFAULT_THINKING=1 -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
  -e SGLANG_USE_AITER=1 -e SGLANG_USE_ROCM700A=0 -e TORCH_BLAS_PREFER_HIPBLASLT=1 -e TRITON_HIP_USE_ASYNC_COPY=0 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 -e SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true -e SGLANG_OPT_HIP_FLYDSL_MXFP8=1 \
  -e SGLANG_HIP_FP4_INDEXER_ROWGROUP=1 -e SGLANG_HIP_DSV41_INDEX_K_PAGE8=1 -e HSA_NO_SCRATCH_RECLAIM=0 -e GPU_MAX_HW_QUEUES=5 \
  -e SGLANG_SHARED_EXPERT_TP1=1 -e SGLANG_DP_USE_GATHERV=1 -e SGLANG_DP_USE_REDUCE_SCATTER=1 -e SGLANG_AITER_CAPTURE_COPY_IN=1 \
  -e SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1 -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false \
  $IMG sglang serve --model-path /models/heretic --served-model-name heretic --host 0.0.0.0 --port $PORT --api-key "$K" \
  --trust-remote-code --tp 2 --attn-dp-size 2 --ep-size 1 --enable-dp-lm-head \
  --enable-prefill-delayer --enable-dp-attention-local-control-broadcast \
  --prefill-delayer-token-usage-low-watermark 0.7 --load-balance-method total_requests \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --enforce-shared-experts-fusion --fp8-gemm-backend aiter --enable-decoder-swa-bounded-replay \
  --cuda-graph-backend-prefill disabled --mem-fraction-static ${MEM:-0.85} --chunked-prefill-size 32768 \
  --prefill-decode-interval ${PDI:-16} --swa-prefix-tails ${TAILS:-4096} --max-running-requests 64 --cuda-graph-max-bs-decode 64 \
  --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600 ${EXTRA:-} >/dev/null || { echo "FAIL docker run"; exit 1; }
t0=$(date +%s)
until curl -sf -m 5 127.0.0.1:$PORT/health >/dev/null; do
  sudo docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "FAIL $NAME exited"; sudo docker logs --tail 80 $NAME 2>&1 | grep -iE 'error|exception|memory|unrecognized|invalid' | grep -v server_args | tail -6 | cut -c1-300; exit 1; }
  sleep 10
done
echo "READY $NAME $(( $(date +%s) - t0 ))s"

#!/usr/bin/env bash
# Extra SGLang replica of the heretic in its own network namespace (two host-network SGLang servers on one MI355X box crash).
# GPUS (default 4,5), PORT (default 30007, published on 127.0.0.1), NAME. Key in $K. Prints READY/FAIL.
DS=/mnt/nvme/ds; IMG=${IMG:-lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021}; GPUS=${GPUS:-4,5}; PORT=${PORT:-30007}; NAME=${NAME:-sg-rep45}
TP=$(echo $GPUS | tr ',' '\n' | wc -l)
sudo docker rm -f $NAME >/dev/null 2>&1
sudo docker run -d --name $NAME --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64g --ipc private -p 127.0.0.1:$PORT:$PORT -v $DS/models:/models -e HIP_VISIBLE_DEVICES=$GPUS \
  -e SGLANG_DEFAULT_THINKING=1 -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
  -e SGLANG_USE_AITER=1 -e SGLANG_USE_ROCM700A=0 -e TORCH_BLAS_PREFER_HIPBLASLT=1 -e TRITON_HIP_USE_ASYNC_COPY=0 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 -e SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true -e SGLANG_OPT_HIP_FLYDSL_MXFP8=1 \
  -e SGLANG_HIP_FP4_INDEXER_ROWGROUP=1 -e HSA_NO_SCRATCH_RECLAIM=0 -e GPU_MAX_HW_QUEUES=2 \
  -e SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1 -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false \
  $IMG sglang serve --model-path /models/heretic --served-model-name heretic --host 0.0.0.0 --port $PORT --api-key "$K" \
  --trust-remote-code --tp $TP --ep-size 1 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --monokernel-backend amd --enforce-shared-experts-fusion --fp8-gemm-backend aiter --enable-decoder-swa-bounded-replay \
  --cuda-graph-backend-prefill disabled --mem-fraction-static ${MEM:-0.80} --chunked-prefill-size ${CHUNK:-32768} --prefill-decode-interval ${PDI:-16} \
  --schedule-policy ${SCHED:-fcfs} --swa-prefix-tails ${TAILS:-4096} --max-running-requests ${MAXRUN:-32} --cuda-graph-max-bs-decode 64 \
  --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600 \
  ${PRIO:---enable-priority-scheduling --default-priority-value 0 --retraction-policy priority} ${EXTRA:-} >/dev/null || { echo "FAIL docker run"; exit 1; }
t0=$(date +%s)
until curl -sf -m 5 127.0.0.1:$PORT/health >/dev/null; do
  sudo docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "FAIL $NAME exited"; sudo docker logs --tail 60 $NAME 2>&1 | grep -iE 'error|exception|memory|unrecognized|invalid' | grep -v server_args | tail -6 | cut -c1-300; exit 1; }
  sleep 10
done
echo "READY $NAME $(( $(date +%s) - t0 ))s"

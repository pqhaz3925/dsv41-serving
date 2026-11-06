#!/usr/bin/env bash
# vLLM TP4 heretic on B300, AgentX-style low-latency recipe (vllm-openai:cu134-nightly). Key in $K.
#   GPUS=4,5,6,7 PORT=30006 NAME=vl-b [MAXSEQ=16] [EXTRA=...] vllm_b300.sh   -> READY/FAIL
DS=/mnt/nvme/ds; IMG=${IMG:-vllm/vllm-openai:cu134-nightly}; GPUS=${GPUS:-4,5,6,7}; PORT=${PORT:-30006}; NAME=${NAME:-vl-b}
TP=$(echo $GPUS | tr ',' '\n' | wc -l)
sudo docker rm -f $NAME >/dev/null 2>&1
sudo docker run -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size 32g --ipc=host --network host \
  -v $DS/models:/models -v $DS/jit/$NAME:/root/.cache \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_RUST_FRONTEND=1 \
  --entrypoint vllm $IMG serve /models/heretic --served-model-name heretic --host 127.0.0.1 --port $PORT --api-key "$K" \
  --trust-remote-code --tokenizer-mode deepseek_v41 --tensor-parallel-size $TP --kv-cache-dtype fp8 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --gpu-memory-utilization ${MEM:-0.93} --max-model-len ${CTX:-1048576} \
  --engram-config '{"cpu_offload":true,"use_thp":true}' \
  --kernel-config '{"enable_flashinfer_autotune":true,"sparse_indexer_topk_backend":"deep_select"}' \
  --max-num-seqs ${MAXSEQ:-16} --prefill-schedule-interval 16 --max-num-batched-tokens 4096 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092]}' \
  --max-cudagraph-capture-size 4092 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  ${EXTRA:-} >/dev/null || { echo "FAIL docker run $NAME"; exit 1; }
t0=$(date +%s)
until curl -sf -m 5 127.0.0.1:$PORT/health >/dev/null; do
  sudo docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "FAIL $NAME exited"; sudo docker logs --tail 120 $NAME 2>&1 | grep -iE 'error|exception|invalid|unrecogn' | grep -v -i 'api.key\|non-default' | tail -6 | cut -c1-300; exit 1; }
  sleep 10
done
echo "READY $NAME $(( $(date +%s) - t0 ))s"

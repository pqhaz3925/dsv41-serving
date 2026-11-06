#!/usr/bin/env bash
# vLLM ROCm (Artificial Analysis "fastest single agent" config, 2026-10-12): TP4, DSpark 5, mono decode. Key in $K.
DS=/mnt/nvme/ds; GPUS=${GPUS:-4,5,6,7}; PORT=${PORT:-30006}; NAME=${NAME:-vl-heretic}
IMG=${IMG:-vllm/vllm-openai-rocm:nightly-rocm100-edad8b03c5285a1c12e7e4f83c1af709b319a01b}
TP=$(echo $GPUS | tr ',' '\n' | wc -l)
sudo docker rm -f $NAME >/dev/null 2>&1
sudo docker run -d --name $NAME --device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined \
  --shm-size 64g --ipc=host --network host -v $DS/models:/models -e HIP_VISIBLE_DEVICES=$GPUS \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_USE_RUST_FRONTEND=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 -e VLLM_ROCM_MONO_DECODE=1 -e OMP_NUM_THREADS=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e AITER_TRITON_LOG_LEVEL=ERROR --entrypoint vllm $IMG serve /models/heretic --served-model-name heretic --host 127.0.0.1 --port $PORT \
  --api-key "$K" --tensor-parallel-size $TP --trust-remote-code \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --moe-backend aiter --block-size 64 --max-model-len 1048576 --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --max-cudagraph-capture-size 384 --gpu-memory-utilization 0.92 --engram-config '{"cpu_offload":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}' \
  ${EXTRA:-} >/dev/null && echo "STARTED $NAME"
t0=$(date +%s)
until curl -sf -m 5 127.0.0.1:$PORT/health >/dev/null; do
  sudo docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "FAIL vllm exited"; sudo docker logs --tail 80 $NAME 2>&1 | grep -iE 'error|exception|unrecogn|invalid' | grep -v -i 'api.key\|non-default' | tail -6 | cut -c1-300; exit 1; }
  sleep 10
done
echo "READY vllm $(( $(date +%s) - t0 ))s"

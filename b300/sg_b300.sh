#!/usr/bin/env bash
# One SGLang replica of the heretic on B300 (host network, 127.0.0.1:PORT). Key in $K.
#   GPUS=0,1,2,3 PORT=30005 NAME=sg-a [EP=4] [IMG=lmsysorg/sglang:dev-dsv41] [MEM=0.70] [CHUNK=4096] [EXTRA=...] sg_b300.sh
# Prints READY <name> <s> or FAIL.
DS=/mnt/nvme/ds; IMG=${IMG:-lmsysorg/sglang:dev-dsv41}; GPUS=${GPUS:-0,1,2,3}; PORT=${PORT:-30005}; NAME=${NAME:-sg-a}
TP=$(echo $GPUS | tr ',' '\n' | wc -l); EP=${EP:-1}
sudo docker rm -f $NAME >/dev/null 2>&1
sudo docker run -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size 32g --ipc=host --network host \
  -v $DS/models:/models -v $DS/jit/$NAME:/root/.cache \
  -e SGLANG_DEFAULT_THINKING=true -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1 \
  $IMG sglang serve --model-path /models/heretic --served-model-name heretic --host 127.0.0.1 --port $PORT --api-key "$K" \
  --trust-remote-code --tp-size $TP --ep-size $EP --context-length ${CONTEXT:-1048576} --mem-fraction-static ${MEM:-0.70} \
  --speculative-algorithm DSPARK --chunked-prefill-size ${CHUNK:-4096} \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
  --enable-priority-scheduling --default-priority-value 0 --retraction-policy priority ${EXTRA:-} >/dev/null || { echo "FAIL docker run $NAME"; exit 1; }
t0=$(date +%s)
until curl -sf -m 5 127.0.0.1:$PORT/health >/dev/null; do
  sudo docker ps --format '{{.Names}}' | grep -qx $NAME || { echo "FAIL $NAME exited"; sudo docker logs --tail 80 $NAME 2>&1 | grep -iE 'error|exception|memory|unrecognized|invalid' | grep -v server_args | tail -6 | cut -c1-300; exit 1; }
  sleep 10
done
echo "READY $NAME $(( $(date +%s) - t0 ))s"

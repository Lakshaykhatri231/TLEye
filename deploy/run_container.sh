#!/bin/bash
# Start the demo container for image tleye:<commit>. Settings (e.g. TLEYE_DTYPE) come from /etc/tleye.env, written at first boot.
# If the server has an IAM role (tleye-ec2-logs), the container's logs go to CloudWatch Logs (group /tleye/demo); otherwise they stay local.
set -euo pipefail
C=$1
TOKEN=$(curl -sf -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
md() { curl -sf -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
LOGS=()
if [ -n "$TOKEN" ] && md iam/info >/dev/null; then
  LOGS=(--log-driver awslogs --log-opt awslogs-region="$(md placement/region)" --log-opt awslogs-group=/tleye/demo
        --log-opt awslogs-create-group=true --log-opt awslogs-stream="app-${C:0:7}")
fi
docker run -d --name tleye --restart unless-stopped -p 80:8000 --env-file /etc/tleye.env "${LOGS[@]}" \
  -e TLEYE_DEMO=1 -e TLEYE_HOSTED=1 -e TLEYE_MODEL=lora -e TLEYE_COMMIT="$C" \
  -v /opt/models/lora:/app/models/lora:ro -v /opt/hf:/home/tleye/.cache/huggingface "tleye:$C"

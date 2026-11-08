#!/bin/bash
# Start the demo container for image tleye:<commit>. Settings (e.g. TLEYE_DTYPE) come from /etc/tleye.env, written at first boot.
set -euo pipefail
C=$1
docker run -d --name tleye --restart unless-stopped -p 80:8000 --env-file /etc/tleye.env \
  -e TLEYE_DEMO=1 -e TLEYE_HOSTED=1 -e TLEYE_MODEL=lora -e TLEYE_COMMIT="$C" \
  -v /opt/models/lora:/app/models/lora:ro -v /opt/hf:/home/tleye/.cache/huggingface "tleye:$C"

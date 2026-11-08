#!/bin/bash
# Runs once on the new EC2 server (cloud-init). __COMMIT__ is replaced by deploy/aws.sh with the exact git commit to serve.
set -euxo pipefail
dnf install -y docker git
systemctl enable --now docker
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile    # headroom during the image build
git clone https://github.com/Lakshaykhatri231/TLEye.git /opt/tleye
cd /opt/tleye && git checkout __COMMIT__
docker build -t tleye --build-arg WITH_LORA=1 --build-arg WITH_DEMO=1 .
mkdir -p /opt/models/lora /opt/hf && chown 10001 /opt/models/lora /opt/hf
# the verdict adapter at its pinned Hugging Face revision, checksums verified
docker run --rm -v /opt/models/lora:/app/models/lora tleye python -m tleye.models.fetch --dest /app/models/lora
# demo at /, API at /health and /check; certificates stay OFF (they would run visitors' code on this server)
docker run -d --name tleye --restart unless-stopped -p 80:8000 --memory 3500m \
  -e TLEYE_DEMO=1 -e TLEYE_HOSTED=1 -e TLEYE_MODEL=lora \
  -v /opt/models/lora:/app/models/lora:ro -v /opt/hf:/home/tleye/.cache/huggingface tleye

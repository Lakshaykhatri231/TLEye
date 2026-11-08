#!/bin/bash
# Runs once on the new EC2 server (cloud-init). __COMMIT__ is replaced by deploy/aws.sh with the exact git commit to serve.
set -euxo pipefail
dnf install -y docker git
systemctl enable --now docker
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile    # 4 GB of disk as extra memory: a 2 GB server needs it for the build and the model
echo '/swapfile none swap defaults 0 0' >> /etc/fstab
git clone https://github.com/Lakshaykhatri231/TLEye.git /opt/tleye
cd /opt/tleye && git checkout __COMMIT__
docker build -t tleye --build-arg WITH_LORA=1 --build-arg WITH_DEMO=1 .
mkdir -p /opt/models/lora /opt/hf && chown 10001 /opt/models/lora /opt/hf
# the verdict adapter at its pinned Hugging Face revision, checksums verified
docker run --rm -v /opt/models/lora:/app/models/lora tleye python -m tleye.models.fetch --dest /app/models/lora
# demo at /, API at /health and /check; certificates stay OFF (they would run visitors' code on this server)
docker run -d --name tleye --restart unless-stopped -p 80:8000 \
  -e TLEYE_DEMO=1 -e TLEYE_HOSTED=1 -e TLEYE_MODEL=lora -e TLEYE_DTYPE=__DTYPE__ \
  -v /opt/models/lora:/app/models/lora:ro -v /opt/hf:/home/tleye/.cache/huggingface tleye

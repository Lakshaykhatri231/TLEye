#!/bin/bash
# Runs once on the new EC2 server (cloud-init). __COMMIT__ / __DTYPE__ are filled in by deploy/aws.sh.
set -euxo pipefail
dnf install -y docker git jq
mkdir -p /etc/docker && echo '{"log-driver": "json-file", "log-opts": {"max-size": "10m", "max-file": "3"}}' > /etc/docker/daemon.json   # logs rotate, disk cannot fill
systemctl enable --now docker
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile    # 4 GB of disk as extra memory for a 2 GB server
echo '/swapfile none swap defaults 0 0' >> /etc/fstab
echo "TLEYE_DTYPE=__DTYPE__" > /etc/tleye.env
git clone https://github.com/Lakshaykhatri231/TLEye.git /opt/tleye
cd /opt/tleye && git checkout __COMMIT__
docker build -t tleye:__COMMIT__ --build-arg WITH_LORA=1 --build-arg WITH_DEMO=1 .
mkdir -p /opt/models/lora /opt/hf && chown 10001 /opt/models/lora /opt/hf
# the verdict adapter at its pinned Hugging Face revision, checksums verified
docker run --rm -v /opt/models/lora:/app/models/lora tleye:__COMMIT__ python -m tleye.models.fetch --dest /app/models/lora
# demo at /, API at /health and /check; certificates stay OFF (they would run visitors' code on this server)
bash /opt/tleye/deploy/run_container.sh __COMMIT__
echo __COMMIT__ > /opt/tleye-deployed
# continuous deployment: check GitHub every 5 minutes (deploy/updater.sh); its output goes to the journal (journalctl -u tleye-update)
cat > /etc/systemd/system/tleye-update.service <<'UNIT'
[Unit]
Description=Deploy the latest green TLEye commit
After=docker.service network-online.target
[Service]
Type=oneshot
ExecStart=/bin/bash /opt/tleye/deploy/updater.sh
UNIT
cat > /etc/systemd/system/tleye-update.timer <<'UNIT'
[Unit]
Description=Check GitHub for a new TLEye commit every 5 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload && systemctl enable --now tleye-update.timer

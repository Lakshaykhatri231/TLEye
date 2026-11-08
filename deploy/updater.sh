#!/bin/bash
# Continuous deployment, pull-based (no AWS keys on GitHub): run every 5 minutes by a systemd timer on the server.
# Deploys origin/main only after ALL its CI jobs finished green; if the new container fails its health check, rolls back.
set -uo pipefail
exec 9>/run/tleye-update.lock; flock -n 9 || exit 0             # a build can outlast the 5-minute interval
API=https://api.github.com/repos/Lakshaykhatri231/TLEye
cd /opt/tleye
cur=$(cat /opt/tleye-deployed)
git fetch -q origin main || { echo "fetch failed"; exit 0; }
new=$(git rev-parse origin/main)
[ "$new" = "$cur" ] && exit 0
runs=$(curl -sf "$API/commits/$new/check-runs?per_page=100") || { echo "GitHub API unavailable"; exit 0; }
# only the three CI jobs count (the uptime check also reports on main, and must not block the commit that fixes an outage)
ci='[.check_runs[] | select(.name == "test" or .name == "sandbox" or .name == "service")]'
ok=$(jq "$ci | map(select(.status == \"completed\" and .conclusion == \"success\")) | length" <<<"$runs")
bad=$(jq "$ci | map(select(.status == \"completed\" and .conclusion != \"success\")) | length" <<<"$runs")
[ "$bad" -gt 0 ] && { echo "CI failed for ${new:0:7}: not deploying"; exit 0; }
[ "$ok" = 3 ] || { echo "waiting for CI on ${new:0:7} ($ok/3 green)"; exit 0; }
echo "deploying ${new:0:7} (was ${cur:0:7})"
git checkout -q "$new"
if ! docker build -q -t "tleye:$new" --build-arg WITH_LORA=1 --build-arg WITH_DEMO=1 .; then
  echo "build failed; staying on ${cur:0:7}"; git checkout -q "$cur"; exit 1
fi
docker stop tleye >/dev/null && docker rename tleye tleye-old     # one model at a time fits in 2 GB: about a minute of downtime
bash /opt/tleye/deploy/run_container.sh "$new" >/dev/null
for _ in $(seq 1 60); do curl -sf -m 5 localhost/health >/dev/null && { healthy=1; break; }; sleep 5; done
if [ "${healthy:-0}" = 1 ]; then
  docker rm -f tleye-old >/dev/null; echo "$new" > /opt/tleye-deployed
  docker image ls --format '{{.Repository}}:{{.Tag}}' tleye | grep -v ":$new$" | xargs -r docker rmi >/dev/null 2>&1 || true
  echo "deployed ${new:0:7}"
else
  echo "new version unhealthy: rolling back to ${cur:0:7}"; docker logs --tail 20 tleye || true
  docker rm -f tleye >/dev/null; docker rename tleye-old tleye; docker start tleye >/dev/null; git checkout -q "$cur"; exit 1
fi

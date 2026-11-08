#!/bin/bash
# Continuous deployment, pull-based (no AWS keys on GitHub): run every 5 minutes by a systemd timer on the server.
# Deploys origin/main only after ALL its CI jobs finished green; if the new container fails its health check, rolls back.
set -uo pipefail
exec 9>/run/tleye-update.lock; flock -n 9 || exit 0             # a build can outlast the 5-minute interval
# with an IAM role, every message also goes to CloudWatch Logs (group /tleye/demo, stream "updater"); the journal keeps a copy either way
log() {
  echo "$*"
  local ev; ev=$(jq -cn --arg m "$*" --argjson t "$(date +%s%3N)" '[{timestamp: $t, message: $m}]')
  put() { aws logs put-log-events --log-group-name /tleye/demo --log-stream-name updater --log-events "$ev" >/dev/null 2>&1; }
  put || { aws logs create-log-stream --log-group-name /tleye/demo --log-stream-name updater >/dev/null 2>&1 && put; } || true
}
export AWS_DEFAULT_REGION=$(TOKEN=$(curl -sf -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60"); \
  curl -sf -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region || echo ap-southeast-2)
API=https://api.github.com/repos/Lakshaykhatri231/TLEye
cd /opt/tleye
cur=$(cat /opt/tleye-deployed)
git fetch -q origin main || { log "fetch failed"; exit 0; }
new=$(git rev-parse origin/main)
[ "$new" = "$cur" ] && exit 0
runs=$(curl -sf "$API/commits/$new/check-runs?per_page=100") || { log "GitHub API unavailable"; exit 0; }
# only the three CI jobs count (the uptime check also reports on main, and must not block the commit that fixes an outage)
ci='[.check_runs[] | select(.name == "test" or .name == "sandbox" or .name == "service")]'
ok=$(jq "$ci | map(select(.status == \"completed\" and .conclusion == \"success\")) | length" <<<"$runs")
bad=$(jq "$ci | map(select(.status == \"completed\" and .conclusion != \"success\")) | length" <<<"$runs")
[ "$bad" -gt 0 ] && { log "CI failed for ${new:0:7}: not deploying"; exit 0; }
[ "$ok" = 3 ] || { log "waiting for CI on ${new:0:7} ($ok/3 green)"; exit 0; }
log "deploying ${new:0:7} (was ${cur:0:7})"
git checkout -q "$new"
if ! docker build -q -t "tleye:$new" --build-arg WITH_LORA=1 --build-arg WITH_DEMO=1 .; then
  log "build failed; staying on ${cur:0:7}"; git checkout -q "$cur"; exit 1
fi
docker stop tleye >/dev/null && docker rename tleye tleye-old     # one model at a time fits in 2 GB: about a minute of downtime
bash /opt/tleye/deploy/run_container.sh "$new" >/dev/null
for _ in $(seq 1 60); do curl -sf -m 5 localhost/health >/dev/null && { healthy=1; break; }; sleep 5; done
if [ "${healthy:-0}" = 1 ]; then
  docker rm -f tleye-old >/dev/null; echo "$new" > /opt/tleye-deployed
  docker image ls --format '{{.Repository}}:{{.Tag}}' tleye | grep -v ":$new$" | xargs -r docker rmi >/dev/null 2>&1 || true
  log "deployed ${new:0:7}"
else
  log "new version unhealthy: rolling back to ${cur:0:7}"; docker logs --tail 20 tleye || true
  docker rm -f tleye >/dev/null; docker rename tleye-old tleye; docker start tleye >/dev/null; git checkout -q "$cur"; exit 1
fi

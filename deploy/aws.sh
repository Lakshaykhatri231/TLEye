#!/bin/bash
# Public demo on one EC2 server.   deploy/aws.sh up | status | down        (uses your `aws configure` credentials and region)
# up: creates a security group (port 80 only, no SSH) and one server that builds and runs the pushed commit. down: removes both.
set -euo pipefail
NAME=tleye-demo
TYPE=${TYPE:-t4g.medium}               # ARM, 2 vCPU, 4 GB: the LoRA service needs about 2.4 GB
DISK_GB=${DISK_GB:-20}
REGION=${REGION:-$(aws configure get region)}   # new-experience AWS projects only allow their home region (e.g. REGION=ap-southeast-2)
ec2() { aws ec2 --region "$REGION" "$@"; }

instance_id() { ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
                  --query 'Reservations[].Instances[].InstanceId' --output text; }
sg_id() { ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" --query 'SecurityGroups[].GroupId' --output text 2>/dev/null; }

case "${1:-}" in
up)
  [ -n "$(instance_id)" ] && { echo "already running: $(instance_id); use status or down"; exit 1; }
  COMMIT=$(git rev-parse HEAD)
  [ "$(git ls-remote origin refs/heads/main | cut -f1)" = "$COMMIT" ] || { echo "push first: the server builds the commit from GitHub"; exit 1; }
  VPC=$(ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  SG=$(sg_id)
  if [ -z "$SG" ]; then
    SG=$(ec2 create-security-group --group-name "$NAME" --description "TLEye demo: HTTP only" --vpc-id "$VPC" --query GroupId --output text)
    ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null
  fi
  ARCH=$([[ "$TYPE" == *g.* ]] && echo arm64 || echo x86_64)
  AMI=$(ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-kernel-6.*-$ARCH" "Name=state,Values=available" \
          --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  UD=$(mktemp); sed "s/__COMMIT__/$COMMIT/" "$(dirname "$0")/user_data.sh" > "$UD"
  ID=$(ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --security-group-ids "$SG" --user-data "file://$UD" \
        --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true}" \
        --metadata-options HttpTokens=required \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=commit,Value=$COMMIT}]" \
        ${DRYRUN:+--dry-run} --query 'Instances[0].InstanceId' --output text)
  rm -f "$UD"
  ec2 wait instance-running --instance-ids "$ID"
  IP=$(ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "server $ID ($TYPE, $REGION) at http://$IP  commit ${COMMIT:0:7}"
  echo "the first start builds the image and downloads the models: about 10-15 minutes. Check with: deploy/aws.sh status" ;;
status)
  ID=$(instance_id); [ -z "$ID" ] && { echo "no server"; exit 0; }
  IP=$(ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "server $ID at http://$IP"
  curl -s -m 5 "http://$IP/health" && echo || echo "not answering yet (still building, or failed: see the boot log below)"
  ec2 get-console-output --instance-id "$ID" --latest --output text 2>/dev/null | tail -5 || true ;;
down)
  ID=$(instance_id)
  [ -n "$ID" ] && { ec2 terminate-instances --instance-ids $ID >/dev/null; ec2 wait instance-terminated --instance-ids $ID; echo "terminated $ID"; }
  SG=$(sg_id); [ -n "$SG" ] && { ec2 delete-security-group --group-id "$SG"; echo "deleted security group $SG"; }
  echo "nothing left running" ;;
*) echo "usage: deploy/aws.sh up | status | down"; exit 2 ;;
esac

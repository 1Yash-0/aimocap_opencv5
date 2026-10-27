#!/usr/bin/env bash
# Cloud-init bootstrap for the AIMoCap web instance (Ubuntu 24.04, x86_64).
# Placeholders __BUCKET__ and __REGION__ are filled in by deploy.py.
set -euo pipefail
exec > >(tee -a /var/log/aimocap-bootstrap.log) 2>&1
echo "bootstrap start $(date -Is)"

BUCKET="__BUCKET__"
REGION="__REGION__"
HOME_DIR=/opt/aimocap
APP=$HOME_DIR/app
VENV=$HOME_DIR/venv

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip nginx ffmpeg curl unzip libgl1 libglib2.0-0t64
snap install aws-cli --classic || true
AWS=$(command -v aws || echo /snap/bin/aws)

id aimocap >/dev/null 2>&1 || useradd --system --home-dir "$HOME_DIR" --create-home --shell /usr/sbin/nologin aimocap
mkdir -p "$APP" "$HOME_DIR/data/opencap_trials" "$HOME_DIR/showcase" "$HOME_DIR/runs" /var/log/aimocap

# Code, models, trials and precomputed results from the deployment bucket.
"$AWS" s3 cp "s3://$BUCKET/code/aimocap.tar.gz" /tmp/aimocap.tar.gz --region "$REGION"
rm -rf "$APP.new" && mkdir -p "$APP.new" && tar -xzf /tmp/aimocap.tar.gz -C "$APP.new"
rm -rf "$APP" && mv "$APP.new" "$APP"
"$AWS" s3 sync "s3://$BUCKET/models/" "$APP/models/" --region "$REGION" --only-show-errors
"$AWS" s3 sync "s3://$BUCKET/trials/" "$HOME_DIR/data/opencap_trials/" --region "$REGION" --only-show-errors
"$AWS" s3 sync "s3://$BUCKET/showcase/" "$HOME_DIR/showcase/" --region "$REGION" --only-show-errors

# Python environment. rtmlib is installed without dependency resolution because
# its declared requirements conflict with the pinned OpenCV 5 and ONNX Runtime.
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$APP/requirements.txt"
"$VENV/bin/pip" install --no-deps rtmlib==0.0.16 tqdm==4.70.1
if "$VENV/bin/pip" show opencv-python-headless >/dev/null 2>&1; then
  "$VENV/bin/pip" uninstall -y opencv-python-headless
  "$VENV/bin/pip" install --force-reinstall --no-deps opencv-python==5.0.0.93
fi
"$VENV/bin/python" -c "import cv2; assert cv2.__version__.startswith('5'), cv2.__version__; print('OpenCV', cv2.__version__)"

# Service environment; the password comes from SSM Parameter Store.
PASSWORD=$("$AWS" ssm get-parameter --name /aimocap/password --with-decryption \
  --query Parameter.Value --output text --region "$REGION")
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
ITYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type)
umask 077
cat > /etc/aimocap.env <<EOF
AIMOCAP_PASSWORD=$PASSWORD
AIMOCAP_DATA_DIR=$HOME_DIR/data/opencap_trials
AIMOCAP_SHOWCASE_DIR=$HOME_DIR/showcase
AIMOCAP_RUNS_DIR=$HOME_DIR/runs
AIMOCAP_LOG_DIR=/var/log/aimocap
AIMOCAP_S3_BUCKET=$BUCKET
AIMOCAP_CLOUDWATCH=1
AIMOCAP_HOURLY_USD=__HOURLY_USD__
AIMOCAP_REPO_URL=__REPO_URL__
AIMOCAP_PYTHON=$VENV/bin/python
AWS_DEFAULT_REGION=$REGION
MPLBACKEND=Agg
EOF
umask 022
chown -R aimocap:aimocap "$HOME_DIR" /var/log/aimocap
echo "instance type $ITYPE"

cat > /etc/systemd/system/aimocap.service <<EOF
[Unit]
Description=AIMoCap web service
After=network-online.target
Wants=network-online.target

[Service]
User=aimocap
Group=aimocap
WorkingDirectory=$APP
EnvironmentFile=/etc/aimocap.env
ExecStart=$VENV/bin/python -m service --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
NoNewPrivileges=true
ProtectSystem=full
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/nginx/sites-available/aimocap <<'EOF'
server {
    listen 80 default_server;
    server_name _;
    client_max_body_size 5m;
    gzip on;
    gzip_types text/css application/javascript application/json text/plain;
    access_log /var/log/nginx/aimocap_access.log;
    error_log /var/log/nginx/aimocap_error.log;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_buffering off;
    }
}
EOF
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/aimocap /etc/nginx/sites-enabled/aimocap
nginx -t

# CloudWatch agent: service, nginx and bootstrap logs, plus memory and CPU metrics.
curl -fsSL -o /tmp/cwagent.deb https://amazoncloudwatch-agent.s3.amazonaws.com/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E /tmp/cwagent.deb
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'EOF'
{
  "agent": {"metrics_collection_interval": 60, "run_as_user": "root"},
  "logs": {"logs_collected": {"files": {"collect_list": [
    {"file_path": "/var/log/aimocap/service.log", "log_group_name": "/aimocap/service", "log_stream_name": "{instance_id}/service"},
    {"file_path": "/var/log/nginx/aimocap_access.log", "log_group_name": "/aimocap/service", "log_stream_name": "{instance_id}/nginx-access"},
    {"file_path": "/var/log/nginx/aimocap_error.log", "log_group_name": "/aimocap/service", "log_stream_name": "{instance_id}/nginx-error"},
    {"file_path": "/var/log/aimocap-bootstrap.log", "log_group_name": "/aimocap/service", "log_stream_name": "{instance_id}/bootstrap"}
  ]}}},
  "metrics": {"namespace": "AIMoCap/Host", "metrics_collected": {
    "mem": {"measurement": ["mem_used_percent"]},
    "cpu": {"measurement": ["cpu_usage_active"], "totalcpu": true}
  }}
}
EOF
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

systemctl daemon-reload
systemctl enable --now aimocap.service
systemctl restart nginx
for _ in $(seq 1 30); do curl -fsS http://127.0.0.1/healthz && break; sleep 2; done
touch "$HOME_DIR/READY"
echo "bootstrap done $(date -Is)"

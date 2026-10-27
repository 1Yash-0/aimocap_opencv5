# AWS deployment

`deploy.py` deploys the web service to a single EC2 instance behind CloudFront. It uses
boto3 (already in `requirements.txt`) and works the same on Windows, Linux and macOS.

## What it creates

| resource | name | notes |
|---|---|---|
| S3 bucket | `aimocap-<account>-<region>` | public access blocked, SSE-S3, `runs/` expire after 7 days |
| SSM parameter | `/aimocap/password` | SecureString, random, read by the instance at boot |
| IAM role and instance profile | `aimocap-ec2` | S3 on the bucket only, the one parameter path, one log group, `AIMoCap*` metrics, SSM managed instance core |
| security group | `aimocap-web` | TCP 80 from the CloudFront origin-facing prefix list only; no SSH |
| EC2 instance | `aimocap-web` | `m7i-flex.large`, Ubuntu 24.04, encrypted gp3, IMDSv2 required |
| CloudFront distribution | comment `aimocap` | HTTPS, caching disabled, viewer headers forwarded |
| CloudWatch log group | `/aimocap/service` | service, nginx and bootstrap logs, 14-day retention |

Every resource is tagged `Project=aimocap`. The instance bootstrap is `user_data.sh`.

## Prerequisites

- AWS CLI v2 profile with permissions for EC2, S3, IAM, SSM, CloudWatch Logs and CloudFront.
- A local checkout with the models (`python scripts/fetch_models.py`), prepared trials in
  `data/opencap_trials/` and precomputed results in `showcase/` (see the main README).

## Commands

Run from the repository root:

```
python deploy/aws/deploy.py plan  --profile <profile>     # read-only: what would be created
python deploy/aws/deploy.py up    --profile <profile>     # create or update everything
python deploy/aws/deploy.py push  --profile <profile>     # new code bundle, restart (add --data for models/trials/showcase)
python deploy/aws/deploy.py status --profile <profile> --logs
python deploy/aws/deploy.py stop  --profile <profile>     # stop the instance when idle
python deploy/aws/deploy.py start --profile <profile>
python deploy/aws/deploy.py down  --profile <profile> [--delete-bucket]
```

`up` prints the HTTPS URL. The instance needs 5 to 10 minutes to install, and a new
CloudFront distribution 5 to 15 minutes to deploy. Read the password with:

```
aws ssm get-parameter --name /aimocap/password --with-decryption --query Parameter.Value --output text --profile <profile> --region ap-southeast-2
```

The user name is `judge`.

## Cost

The instance type is Free-tier eligible; charges otherwise follow the EC2 on-demand rate
for the region (see the AWS pricing page). Stop the instance when it is not needed;
`start` re-points CloudFront at the new public DNS name. Pass `--hourly-usd` to `up` to
show an estimated cost per run in the web interface.

## Security

- HTTPS to the viewer through CloudFront; the origin accepts traffic only from CloudFront.
- No SSH port; administration through SSM Run Command (`push`, `status --logs`).
- The password lives in SSM Parameter Store and is written to `/etc/aimocap.env` (mode 0600).
- The instance role cannot read or write anything outside its bucket, parameter, log
  group and metric namespace.
- The service accepts no uploads; it runs fixed presets on the bundled trials.

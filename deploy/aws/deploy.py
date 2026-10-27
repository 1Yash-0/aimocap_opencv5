"""Deploy the AIMoCap web service to AWS.

Creates (idempotently) an S3 bucket, an SSM password parameter, a least-privilege
instance role, a security group open only to CloudFront, one EC2 instance and a
CloudFront distribution in front of it. All resources are tagged
``Project=aimocap`` and named ``aimocap-*``.

Usage (from the repository root)::

    python deploy/aws/deploy.py plan                  # show what would happen
    python deploy/aws/deploy.py up                    # create or update everything
    python deploy/aws/deploy.py push                  # upload a new code bundle and restart
    python deploy/aws/deploy.py status
    python deploy/aws/deploy.py stop | start          # cost control
    python deploy/aws/deploy.py down [--delete-bucket]

Options: ``--profile`` and ``--region`` (default: AWS_PROFILE / ap-southeast-2),
``--instance-type`` (default m7i-flex.large), ``--hourly-usd`` (shown as a cost
estimate in the web UI), ``--yes`` (no confirmation prompt).
"""

from __future__ import annotations

import argparse
import io
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME = "aimocap"
TAGS = [{"Key": "Project", "Value": NAME}]
PASSWORD_PARAM = "/aimocap/password"
LOG_GROUP = "/aimocap/service"
AMI_PARAM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
CLOUDFRONT_PREFIX_LIST = "com.amazonaws.global.cloudfront.origin-facing"
CACHE_POLICY_DISABLED = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"  # Managed-CachingDisabled
ORIGIN_POLICY_ALL_VIEWER = "b689b0a8-53d0-40ab-baf2-68738e2966ac"  # Managed-AllViewerExceptHostHeader
MODEL_FILES = (
    "rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
    "yolox_nano.onnx",
)
HAND_MODEL_DIR = "rtmpose_hand"
TRIAL_SKIP_DIRS = {"syncd"}  # reference clips, not used by the service
SHOWCASE_FILES = (
    "trial.json", "calibration.json", "calibration_check.json", "multipose.npz", "sync.json",
    "skeleton3d.npz", "motion.bvh", "overlay_2d.mp4", "skeleton_vs_mocap.mp4", "metrics.json", "events.jsonl",
    "character.mp4",
)


def log(message: str) -> None:
    print(f"[deploy] {message}", flush=True)


class Deployment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session = boto3.Session(profile_name=args.profile, region_name=args.region)
        self.region = args.region
        self.account = self.session.client("sts").get_caller_identity()["Account"]
        self.bucket = f"{NAME}-{self.account}-{self.region}"
        self.s3 = self.session.client("s3")
        self.ec2 = self.session.client("ec2")
        self.iam = self.session.client("iam")
        self.ssm = self.session.client("ssm")
        self.logs = self.session.client("logs")
        self.cf = self.session.client("cloudfront")

    # -- lookups ---------------------------------------------------------------

    def instance(self) -> dict | None:
        res = self.ec2.describe_instances(Filters=[
            {"Name": "tag:Project", "Values": [NAME]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ])
        found = [i for r in res["Reservations"] for i in r["Instances"]]
        return found[0] if found else None

    def distribution(self) -> dict | None:
        for page in self.cf.get_paginator("list_distributions").paginate():
            for item in page.get("DistributionList", {}).get("Items", []):
                if item.get("Comment") == NAME:
                    return item
        return None

    def bucket_exists(self) -> bool:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            return True
        except ClientError:
            return False

    # -- plan ------------------------------------------------------------------

    def plan(self) -> list[str]:
        steps = []
        steps.append(("reuse" if self.bucket_exists() else "create") + f" S3 bucket {self.bucket} "
                     "(public access blocked, SSE-S3, runs/ expire after 7 days)")
        steps.append(f"upload code bundle (git HEAD), models, trials and showcase to s3://{self.bucket}/")
        steps.append(f"ensure SSM SecureString {PASSWORD_PARAM}")
        steps.append(f"ensure IAM role and instance profile {NAME}-ec2 (bucket, parameter, logs, metrics only)")
        steps.append(f"ensure security group {NAME}-web (TCP 80 from the CloudFront origin-facing prefix list only)")
        inst = self.instance()
        steps.append(f"reuse instance {inst['InstanceId']} ({inst['State']['Name']})" if inst else
                     f"launch EC2 {self.args.instance_type} (Ubuntu 24.04, gp3 {self.args.volume_gb} GB encrypted, IMDSv2)")
        dist = self.distribution()
        steps.append(f"reuse CloudFront {dist['DomainName']}" if dist else "create CloudFront distribution (HTTPS)")
        steps.append(f"ensure CloudWatch log group {LOG_GROUP} (14-day retention)")
        return steps

    # -- resources -------------------------------------------------------------

    def ensure_bucket(self) -> None:
        if not self.bucket_exists():
            log(f"creating bucket {self.bucket}")
            self.s3.create_bucket(Bucket=self.bucket,
                                  CreateBucketConfiguration={"LocationConstraint": self.region})
        self.s3.put_public_access_block(Bucket=self.bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        self.s3.put_bucket_encryption(Bucket=self.bucket, ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})
        self.s3.put_bucket_lifecycle_configuration(Bucket=self.bucket, LifecycleConfiguration={"Rules": [
            {"ID": "expire-runs", "Status": "Enabled", "Filter": {"Prefix": "runs/"}, "Expiration": {"Days": 7}}]})
        self.s3.put_bucket_tagging(Bucket=self.bucket, Tagging={"TagSet": TAGS})

    def _upload(self, local: Path, key: str) -> None:
        try:
            head = self.s3.head_object(Bucket=self.bucket, Key=key)
            if head["ContentLength"] == local.stat().st_size and key.split("/")[0] != "code":
                return
        except ClientError:
            pass
        log(f"upload {key} ({local.stat().st_size / 1e6:.1f} MB)")
        self.s3.upload_file(str(local), self.bucket, key)

    def upload_code(self) -> None:
        bundle = subprocess.run(["git", "archive", "--format=tar.gz", "HEAD"], cwd=REPO_ROOT,
                                check=True, capture_output=True).stdout
        self.s3.upload_fileobj(io.BytesIO(bundle), self.bucket, "code/aimocap.tar.gz")
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                check=True, capture_output=True, text=True).stdout.strip()
        log(f"uploaded code bundle {commit} ({len(bundle) / 1e6:.1f} MB)")

    def upload_data(self) -> None:
        models = REPO_ROOT / "models"
        for rel in MODEL_FILES:
            self._upload(models / rel, f"models/{rel}")
        for path in sorted((models / HAND_MODEL_DIR).rglob("end2end.onnx")):
            self._upload(path, f"models/{path.relative_to(models).as_posix()}")
        trials = REPO_ROOT / "data" / "opencap_trials"
        showcase = REPO_ROOT / "showcase"
        for trial_dir in sorted(p for p in showcase.iterdir() if p.is_dir()):
            for name in SHOWCASE_FILES:
                if (trial_dir / name).is_file():
                    self._upload(trial_dir / name, f"showcase/{trial_dir.name}/{name}")
            source = trials / trial_dir.name
            for path in sorted(source.rglob("*")):
                rel = path.relative_to(source)
                if path.is_file() and rel.parts[0] not in TRIAL_SKIP_DIRS:
                    self._upload(path, f"trials/{trial_dir.name}/{rel.as_posix()}")

    def ensure_password(self) -> None:
        try:
            self.ssm.get_parameter(Name=PASSWORD_PARAM)
        except ClientError:
            log(f"creating {PASSWORD_PARAM}")
            self.ssm.put_parameter(Name=PASSWORD_PARAM, Value=secrets.token_urlsafe(15), Type="SecureString",
                                   Tags=TAGS)

    def ensure_role(self) -> str:
        role = f"{NAME}-ec2"
        trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "sts:AssumeRole",
                 "Principal": {"Service": "ec2.amazonaws.com"}}]}
        try:
            self.iam.get_role(RoleName=role)
        except ClientError:
            log(f"creating role {role}")
            self.iam.create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps(trust), Tags=TAGS,
                                 Description="AIMoCap web instance")
        arn = f"arn:aws:s3:::{self.bucket}"
        policy = {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": arn},
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"{arn}/*"},
            {"Effect": "Allow", "Action": ["ssm:GetParameter"],
             "Resource": f"arn:aws:ssm:{self.region}:{self.account}:parameter/aimocap/*"},
            {"Effect": "Allow", "Action": ["kms:Decrypt"], "Resource": "*",
             "Condition": {"StringEquals": {"kms:ViaService": f"ssm.{self.region}.amazonaws.com"}}},
            {"Effect": "Allow", "Action": ["cloudwatch:PutMetricData"], "Resource": "*",
             "Condition": {"StringLike": {"cloudwatch:namespace": "AIMoCap*"}}},
            {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
             "Resource": f"arn:aws:logs:{self.region}:{self.account}:log-group:{LOG_GROUP}*"},
            {"Effect": "Allow", "Action": ["ec2:DescribeTags"], "Resource": "*"},
        ]}
        self.iam.put_role_policy(RoleName=role, PolicyName=f"{NAME}-runtime", PolicyDocument=json.dumps(policy))
        self.iam.attach_role_policy(RoleName=role, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
        try:
            self.iam.get_instance_profile(InstanceProfileName=role)
        except ClientError:
            self.iam.create_instance_profile(InstanceProfileName=role, Tags=TAGS)
            self.iam.add_role_to_instance_profile(InstanceProfileName=role, RoleName=role)
            log("waiting for the instance profile to propagate")
            time.sleep(15)
        return role

    def ensure_security_group(self) -> str:
        vpc = self.ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
        groups = self.ec2.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [f"{NAME}-web"]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
        if groups:
            return groups[0]["GroupId"]
        prefix = self.ec2.describe_managed_prefix_lists(Filters=[
            {"Name": "prefix-list-name", "Values": [CLOUDFRONT_PREFIX_LIST]}])["PrefixLists"][0]["PrefixListId"]
        log(f"creating security group {NAME}-web")
        group = self.ec2.create_security_group(
            GroupName=f"{NAME}-web", Description="AIMoCap web origin, CloudFront only", VpcId=vpc,
            TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS}])["GroupId"]
        self.ec2.authorize_security_group_ingress(GroupId=group, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 80, "ToPort": 80,
            "PrefixListIds": [{"PrefixListId": prefix, "Description": "CloudFront origin-facing"}]}])
        return group

    def user_data(self) -> str:
        text = (Path(__file__).parent / "user_data.sh").read_text(encoding="utf-8")
        return (text.replace("__BUCKET__", self.bucket).replace("__REGION__", self.region)
                .replace("__HOURLY_USD__", str(self.args.hourly_usd))
                .replace("__REPO_URL__", self.args.repo_url).replace("\r\n", "\n"))

    def ensure_instance(self, role: str, group: str) -> dict:
        inst = self.instance()
        if inst is None:
            ami = self.ssm.get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
            log(f"launching {self.args.instance_type} from {ami}")
            inst = self.ec2.run_instances(
                ImageId=ami, InstanceType=self.args.instance_type, MinCount=1, MaxCount=1,
                IamInstanceProfile={"Name": role}, SecurityGroupIds=[group], UserData=self.user_data(),
                MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2, "HttpEndpoint": "enabled"},
                BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {
                    "VolumeSize": self.args.volume_gb, "VolumeType": "gp3", "Encrypted": True,
                    "DeleteOnTermination": True}}],
                TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS + [{"Key": "Name", "Value": f"{NAME}-web"}]},
                                   {"ResourceType": "volume", "Tags": TAGS}],
            )["Instances"][0]
        elif inst["State"]["Name"] in ("stopped", "stopping"):
            log(f"starting {inst['InstanceId']}")
            self.ec2.get_waiter("instance_stopped").wait(InstanceIds=[inst["InstanceId"]])
            self.ec2.start_instances(InstanceIds=[inst["InstanceId"]])
        self.ec2.get_waiter("instance_running").wait(InstanceIds=[inst["InstanceId"]])
        return self.instance()

    def ensure_distribution(self, origin_dns: str) -> dict:
        dist = self.distribution()
        if dist is not None:
            if dist["Origins"]["Items"][0]["DomainName"] != origin_dns:
                self._set_origin(dist["Id"], origin_dns)
            return dist
        log("creating CloudFront distribution")
        config = {
            "CallerReference": f"{NAME}-{int(time.time())}", "Comment": NAME, "Enabled": True,
            "PriceClass": "PriceClass_100", "HttpVersion": "http2", "IsIPV6Enabled": True,
            "Origins": {"Quantity": 1, "Items": [{
                "Id": "ec2", "DomainName": origin_dns,
                "CustomOriginConfig": {"HTTPPort": 80, "HTTPSPort": 443, "OriginProtocolPolicy": "http-only",
                                       "OriginReadTimeout": 60, "OriginKeepaliveTimeout": 5,
                                       "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]}}}]},
            "DefaultCacheBehavior": {
                "TargetOriginId": "ec2", "ViewerProtocolPolicy": "redirect-to-https", "Compress": True,
                "CachePolicyId": CACHE_POLICY_DISABLED, "OriginRequestPolicyId": ORIGIN_POLICY_ALL_VIEWER,
                "AllowedMethods": {"Quantity": 7, "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
                                   "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}}},
        }
        res = self.cf.create_distribution_with_tags(DistributionConfigWithTags={
            "DistributionConfig": config, "Tags": {"Items": TAGS}})
        return res["Distribution"]

    def _set_origin(self, dist_id: str, origin_dns: str) -> None:
        current = self.cf.get_distribution_config(Id=dist_id)
        config = current["DistributionConfig"]
        config["Origins"]["Items"][0]["DomainName"] = origin_dns
        log(f"pointing CloudFront at {origin_dns}")
        self.cf.update_distribution(Id=dist_id, IfMatch=current["ETag"], DistributionConfig=config)

    def ensure_log_group(self) -> None:
        try:
            self.logs.create_log_group(logGroupName=LOG_GROUP, tags={"Project": NAME})
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise
        self.logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=14)

    def run_on_instance(self, script: str, timeout: int = 900) -> str:
        inst = self.instance()
        if inst is None:
            raise SystemExit("no instance")
        cmd = self.ssm.send_command(InstanceIds=[inst["InstanceId"]], DocumentName="AWS-RunShellScript",
                                    Parameters={"commands": [script]}, TimeoutSeconds=timeout)["Command"]["CommandId"]
        for _ in range(timeout // 5):
            time.sleep(5)
            try:
                out = self.ssm.get_command_invocation(CommandId=cmd, InstanceId=inst["InstanceId"])
            except ClientError:
                continue
            if out["Status"] not in ("Pending", "InProgress", "Delayed"):
                return f"{out['Status']}\n{out['StandardOutputContent']}\n{out['StandardErrorContent']}"
        return "timed out waiting for the command"

    # -- commands ----------------------------------------------------------------

    def up(self) -> None:
        self.ensure_bucket()
        self.upload_code()
        self.upload_data()
        self.ensure_password()
        self.ensure_log_group()
        role = self.ensure_role()
        group = self.ensure_security_group()
        inst = self.ensure_instance(role, group)
        dist = self.ensure_distribution(inst["PublicDnsName"])
        log(f"instance {inst['InstanceId']} {inst['PublicDnsName']}")
        log(f"URL https://{dist['DomainName']}  (user: judge)")
        log("the instance needs about 5-10 minutes to install; CloudFront about 5-15 minutes to deploy")
        log(f"password: aws ssm get-parameter --name {PASSWORD_PARAM} --with-decryption "
            f"--query Parameter.Value --output text --region {self.region}")

    def push(self) -> None:
        self.upload_code()
        if self.args.data:
            self.upload_data()
        script = (
            "set -e; A=$(command -v aws || echo /snap/bin/aws); "
            f"$A s3 cp s3://{self.bucket}/code/aimocap.tar.gz /tmp/aimocap.tar.gz --region {self.region}; "
            "rm -rf /opt/aimocap/app.new && mkdir -p /opt/aimocap/app.new && "
            "tar -xzf /tmp/aimocap.tar.gz -C /opt/aimocap/app.new && "
            "cp -r /opt/aimocap/app/models /opt/aimocap/app.new/ && "
            "rm -rf /opt/aimocap/app && mv /opt/aimocap/app.new /opt/aimocap/app && "
            "/opt/aimocap/venv/bin/pip install -q -r /opt/aimocap/app/requirements.txt && "
            f"$A s3 sync s3://{self.bucket}/showcase/ /opt/aimocap/showcase/ --region {self.region} --only-show-errors && "
            f"$A s3 sync s3://{self.bucket}/trials/ /opt/aimocap/data/opencap_trials/ --region {self.region} --only-show-errors && "
            "chown -R aimocap:aimocap /opt/aimocap && systemctl restart aimocap && sleep 3 && "
            "curl -fsS http://127.0.0.1/healthz"
        )
        print(self.run_on_instance(script))

    def status(self) -> None:
        inst = self.instance()
        dist = self.distribution()
        print(f"bucket       {self.bucket} ({'exists' if self.bucket_exists() else 'missing'})")
        print(f"instance     {inst['InstanceId'] + ' ' + inst['State']['Name'] if inst else 'none'}")
        print(f"cloudfront   {dist['DomainName'] + ' ' + dist['Status'] if dist else 'none'}")
        if inst and inst["State"]["Name"] == "running" and self.args.logs:
            print(self.run_on_instance("tail -n 30 /var/log/aimocap-bootstrap.log; "
                                       "systemctl is-active aimocap; tail -n 20 /var/log/aimocap/service.log", 120))

    def stop(self) -> None:
        inst = self.instance()
        if inst:
            self.ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
            log(f"stopping {inst['InstanceId']}")

    def start(self) -> None:
        inst = self.instance()
        if inst:
            self.ec2.start_instances(InstanceIds=[inst["InstanceId"]])
            self.ec2.get_waiter("instance_running").wait(InstanceIds=[inst["InstanceId"]])
            inst = self.instance()
            dist = self.distribution()
            if dist:
                self.ensure_distribution(inst["PublicDnsName"])
            log(f"running {inst['PublicDnsName']}")

    def down(self) -> None:
        dist = self.distribution()
        if dist:
            current = self.cf.get_distribution_config(Id=dist["Id"])
            if current["DistributionConfig"]["Enabled"]:
                current["DistributionConfig"]["Enabled"] = False
                self.cf.update_distribution(Id=dist["Id"], IfMatch=current["ETag"],
                                            DistributionConfig=current["DistributionConfig"])
            log("waiting for CloudFront to disable (can take 15 minutes)")
            self.cf.get_waiter("distribution_deployed").wait(Id=dist["Id"])
            etag = self.cf.get_distribution_config(Id=dist["Id"])["ETag"]
            self.cf.delete_distribution(Id=dist["Id"], IfMatch=etag)
        inst = self.instance()
        if inst:
            self.ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
            self.ec2.get_waiter("instance_terminated").wait(InstanceIds=[inst["InstanceId"]])
        for group in self.ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [f"{NAME}-web"]}])[
                "SecurityGroups"]:
            self.ec2.delete_security_group(GroupId=group["GroupId"])
        role = f"{NAME}-ec2"
        for call in (lambda: self.iam.remove_role_from_instance_profile(InstanceProfileName=role, RoleName=role),
                     lambda: self.iam.delete_instance_profile(InstanceProfileName=role),
                     lambda: self.iam.delete_role_policy(RoleName=role, PolicyName=f"{NAME}-runtime"),
                     lambda: self.iam.detach_role_policy(
                         RoleName=role, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"),
                     lambda: self.iam.delete_role(RoleName=role),
                     lambda: self.ssm.delete_parameter(Name=PASSWORD_PARAM),
                     lambda: self.logs.delete_log_group(logGroupName=LOG_GROUP)):
            try:
                call()
            except ClientError:
                pass
        if self.args.delete_bucket and self.bucket_exists():
            bucket = self.session.resource("s3").Bucket(self.bucket)
            bucket.objects.all().delete()
            bucket.delete()
        log("teardown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy AIMoCap to AWS")
    parser.add_argument("command", choices=["plan", "up", "push", "status", "stop", "start", "down"])
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--instance-type", default="m7i-flex.large")
    parser.add_argument("--volume-gb", type=int, default=30)
    parser.add_argument("--hourly-usd", type=float, default=0.0)
    parser.add_argument("--repo-url", default="https://github.com/1Yash-0/aimocap_opencv5")
    parser.add_argument("--data", action="store_true", help="push: also upload models, trials and showcase")
    parser.add_argument("--logs", action="store_true", help="status: fetch logs from the instance")
    parser.add_argument("--delete-bucket", action="store_true", help="down: also delete the bucket")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    deployment = Deployment(args)
    if args.command in ("plan", "up", "down"):
        print(f"account {deployment.account}, region {deployment.region}")
        for step in deployment.plan() if args.command != "down" else ["delete every aimocap-* resource"]:
            print(f"  - {step}")
        if args.command == "plan":
            return
        if not args.yes and input("Proceed? [yes/no] ").strip().lower() != "yes":
            sys.exit("aborted")
    getattr(deployment, args.command)()


if __name__ == "__main__":
    main()

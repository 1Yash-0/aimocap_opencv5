"""Render the architecture diagram used in the submission (docs/architecture.png)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
CV = "#e7f0fb"
AWS = "#fff3e0"
OWN = "#eef6ee"
EDGE = "#3a4552"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=10.5, weight="bold", color="#1b1f24")
    ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=8.6, color="#2d3640", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=EDGE))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center", fontsize=8, color="#545d68")


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.6), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(0.2, 7.35, "AIMoCap architecture", fontsize=15, weight="bold", color="#1b1f24")

    # Pipeline row
    ax.text(0.2, 6.75, "Pipeline (runs on the EC2 instance CPU)", fontsize=10, color="#545d68")
    y, h, w = 4.75, 1.8, 2.05
    box(ax, 0.2, y, w, h, "Input", "3-5 phone videos\nunsynchronized\n+ checkerboard clip", OWN)
    box(ax, 2.55, y, w, h, "Calibration", "OpenCV 5\nfindChessboardCorners\ncornerSubPix\nsolvePnPGeneric (IPPE)", CV)
    box(ax, 4.9, y, w, h, "2D pose", "OpenCV 5 cv2.dnn\nYOLOX person detector\nRTMW-x 133 keypoints\nRTMPose hand model", CV)
    box(ax, 7.25, y, w, h, "Synchronization", "epipolar search +\nmulti-view reprojection\non co-visible frames\nundistortPoints", OWN)
    box(ax, 9.6, y, w, h, "Triangulation", "soft camera trust\nrobust fusion\nKalman/RTS smoother\n3D confidence", OWN)
    box(ax, 11.95, y, 1.85, h, "Export", "BVH skeleton\nmetrics.json\nvideos", OWN)
    for x in (2.25, 4.6, 6.95, 9.3, 11.65):
        arrow(ax, x, y + h / 2, x + 0.3, y + h / 2)

    # AWS row
    ax.text(0.2, 3.95, "AWS deployment (ap-southeast-2)", fontsize=10, color="#545d68")
    y2, h2 = 1.75, 2.0
    box(ax, 0.2, y2, 2.2, h2, "Browser", "judge opens the\nHTTPS link\n(password protected)", OWN)
    box(ax, 2.9, y2, 2.4, h2, "Amazon CloudFront", "HTTPS endpoint\ncaching disabled\nforwards to EC2", AWS)
    box(ax, 5.8, y2, 3.0, h2, "Amazon EC2", "m7i-flex.large, Ubuntu 24.04\nnginx + FastAPI service\njob runner -> pipeline\nIMDSv2, no SSH", AWS)
    box(ax, 9.3, y2 + 1.05, 2.1, 0.95, "Amazon S3", "code, models, data,\nresults (7-day expiry)", AWS)
    box(ax, 11.7, y2 + 1.05, 2.1, 0.95, "SSM", "password (SecureString)\nRun Command admin", AWS)
    box(ax, 9.3, y2 - 0.05, 2.1, 0.95, "CloudWatch", "logs, RunSeconds,\nStageSeconds metrics", AWS)
    box(ax, 11.7, y2 - 0.05, 2.1, 0.95, "IAM role", "least privilege:\nbucket, parameter, logs", AWS)
    arrow(ax, 2.4, y2 + h2 / 2, 2.9, y2 + h2 / 2, "HTTPS")
    arrow(ax, 5.3, y2 + h2 / 2, 5.8, y2 + h2 / 2, "HTTP")
    arrow(ax, 8.8, y2 + 1.5, 9.3, y2 + 1.5)
    arrow(ax, 8.8, y2 + 0.45, 9.3, y2 + 0.45)
    ax.text(5.55, y2 - 0.35, "security group: port 80 from the CloudFront origin-facing prefix list only",
            ha="center", fontsize=8, color="#545d68")
    arrow(ax, 7.3, y2 + h2, 7.3, 4.7, "")
    ax.text(7.45, 4.2, "runs the pipeline", fontsize=8, color="#545d68")

    # Legend
    for i, (c, t) in enumerate(((CV, "OpenCV 5"), (AWS, "AWS service"), (OWN, "AIMoCap code"))):
        ax.add_patch(FancyBboxPatch((0.2 + i * 2.2, 0.35), 0.35, 0.3, boxstyle="round,pad=0.01", fc=c, ec=EDGE))
        ax.text(0.65 + i * 2.2, 0.5, t, va="center", fontsize=9)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()

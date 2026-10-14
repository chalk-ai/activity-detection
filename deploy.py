#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute"]
# ///
"""Deploy the activity-detection demo (server.py, pipeline.py, assets/) as a GPU ScalingGroup.

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l4]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = "/models"

# Download the detector and CLIP weights at build time so a cold replica never waits on them.
PREFETCH = (
    "import torchvision; from transformers import CLIPModel, CLIPProcessor; "
    "torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn("
    "weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT); "
    "CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); "
    "CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
)


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="activity-detection")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0"])
        .pip_install(
            [
                "transformers>=4.57,<5",
                "silero-vad==6.2.3",
                "av>=14,<17",
                "numpy<3",
                "pillow",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
                "python-multipart==0.0.20",
            ]
        )
        .env({"TORCH_HOME": f"{MODELS}/torch", "HF_HOME": f"{MODELS}/hf"})
        .run_commands(f'python -c "{PREFETCH}"')
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "pipeline.py"), "/app/pipeline.py", strategy="copy")
        # The page, the recording and its labels exceed the 32 KiB inline-copy budget.
        .add_local_dir(str(HERE / "assets"), "/assets", strategy="volume")
        .workdir("/app")
    )

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        cpu="4",
        memory="16Gi",
        env={"PORT": "8080", "ASSETS_DIR": "/assets", "HF_HUB_OFFLINE": "1"},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        # Ready only after the models load and the bundled recording has been analyzed.
        readiness_probe=ReadinessProbe.http(
            "/status", period_seconds=5, failure_threshold=240
        ),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()

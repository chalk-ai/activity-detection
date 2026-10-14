#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", "transformers>=4.57,<5",
#   "silero-vad==6.2.3", "av>=14,<17", "numpy<3", "pillow",
# ]
# ///
"""Fit the fusion weights on one labeled recording and report accuracy on another.

    ./calibrate.py media/calib.mp4 media/calib.labels.json assets/demo.mp4 assets/demo.labels.json

Writes weights.json (copy it to assets/ to deploy it). Transition shots (standing up, sitting down) are excluded from both the
fit and the evaluation, since their "true" label is a matter of taste.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import pipeline

FIT_STEPS = 4000
L2 = 0.01


def labeled_matrix(
    signals: dict[str, np.ndarray], labels: dict
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(signals["person"])):
        t = i / pipeline.SAMPLE_FPS
        seg = next((g for g in labels["segments"] if g["start"] <= t < g["end"]), None)
        if seg is None or seg["label"] == "transition":
            continue
        xs.append([signals[k][i] for k in pipeline.SIGNALS])
        ys.append(1.0 if seg["label"] == "active" else 0.0)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.zeros(x.shape[1] + 1)
    xb = np.hstack([np.ones((len(x), 1)), x])
    for _ in range(FIT_STEPS):
        p = 1 / (1 + np.exp(-xb @ w))
        grad = xb.T @ (p - y) / len(y) + L2 * np.r_[0, w[1:]]
        w -= 0.5 * grad
    return w


def main() -> None:
    calib_video, calib_labels, demo_video, demo_labels = map(Path, sys.argv[1:5])
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models: pipeline.Models | None = None

    def raw_signals(video: Path) -> dict[str, np.ndarray]:
        """Raw per-sample signals, cached next to the video so re-fitting the fusion is instant."""
        nonlocal models
        cache = video.with_suffix(".signals.npz")
        if cache.exists():
            return dict(np.load(cache))
        models = models or pipeline.Models(device)
        raw, _ = pipeline.extract_raw(models, pipeline.decode(video, 600))
        np.savez(cache, **raw)
        return raw

    calib_sig = pipeline.windowed(raw_signals(calib_video))
    x, y = labeled_matrix(calib_sig, json.loads(calib_labels.read_text()))
    w = fit_logistic(x, y)
    weights = {
        "bias": round(float(w[0]), 4),
        **{k: round(float(v), 4) for k, v in zip(pipeline.SIGNALS, w[1:])},
    }
    print("weights", weights)
    Path("weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    for name, video, labels in [
        ("calib", calib_video, calib_labels),
        ("demo", demo_video, demo_labels),
    ]:
        sig = pipeline.windowed(raw_signals(video))
        _, state, segs = pipeline.smooth_and_segment(pipeline.fuse(sig, weights))
        result = {"state": [int(v) for v in state]}
        print(
            name,
            pipeline.evaluate(result, json.loads(labels.read_text())),
            [(s["label"], s["start"], s["end"]) for s in segs],
        )


if __name__ == "__main__":
    main()

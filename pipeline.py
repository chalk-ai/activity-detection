"""Activity detection for a livestream recording: is the streamer active, or on a break?

Four signals per sample (SAMPLE_FPS per second of video), each in [0, 1]:
  person  -- best "person" score from a COCO object detector
  scene   -- CLIP zero-shot: "someone at the desk streaming" vs "empty chair / be-right-back card / idle"
  speech  -- Silero VAD speech probability over the sample's audio window
  motion  -- frame-difference energy against the previous sample
Each signal is summarized over a trailing window (see WINDOWS), a logistic model fuses them into
P(active), and an EMA plus hysteresis turns that into active/break segments.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image

SAMPLE_FPS = 4
AUDIO_SR = 16_000
# How each raw signal is summarized over a trailing window. Presence (person, scene) is a short
# mean so breaks register within seconds. Speech is a longer max -- "spoke recently" -- which is
# what separates a streamer scrolling a phone (present, silent) from one between sentences.
WINDOWS: dict[str, tuple[str, float]] = {
    "person": ("mean", 2.0),
    "scene": ("mean", 2.0),
    "speech": ("max", 6.0),
    "motion": ("mean", 4.0),
}
EMA_SECONDS = 1.5
ENTER_ACTIVE, ENTER_BREAK = 0.6, 0.4
MIN_SEGMENT_SECONDS = 2.0
SIGNALS = ("person", "scene", "speech", "motion")

ACTIVE_PROMPTS = [
    "a person sitting at a desk talking to a webcam",
    "a gamer wearing headphones playing a video game at a computer",
    "a smiling streamer talking into a microphone",
]
BREAK_PROMPTS = [
    "an empty gaming chair in a room with nobody in it",
    "a 'be right back' screen with large text",
    "a person leaning back looking down at a phone, not paying attention",
]


@dataclasses.dataclass
class Decoded:
    frames: list[Image.Image]
    audio: np.ndarray  # mono float32 at AUDIO_SR
    duration_s: float


def decode(path: Path, max_seconds: float) -> Decoded:
    """Decode frames at SAMPLE_FPS and mono 16 kHz audio, stopping at max_seconds."""
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None:
            raise ValueError("no video stream")
        next_t = 0.0
        for frame in container.decode(vstream):
            t = float(frame.time or 0.0)
            if t > max_seconds:
                break
            if t + 1e-6 >= next_t:
                img = frame.to_image()
                img.thumbnail((640, 640))
                frames.append(img)
                next_t += 1.0 / SAMPLE_FPS
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        astream = next((s for s in container.streams if s.type == "audio"), None)
        if astream is not None:
            resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_SR)
            for frame in container.decode(astream):
                if float(frame.time or 0.0) > max_seconds:
                    break
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
    audio = (
        np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    )
    duration = len(frames) / SAMPLE_FPS
    return Decoded(
        frames=frames, audio=audio[: int(duration * AUDIO_SR)], duration_s=duration
    )


class Models:
    def __init__(self, device: str) -> None:
        super().__init__()
        import torchvision
        from silero_vad import load_silero_vad
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.detector = (
            torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights
            )
            .eval()
            .to(device)
        )
        self.det_tf = weights.transforms()
        self.clip = (
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        )
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        with torch.no_grad():
            text = self.clip_proc(
                text=ACTIVE_PROMPTS + BREAK_PROMPTS, return_tensors="pt", padding=True
            ).to(device)
            emb = self.clip.get_text_features(**text)
            self.text_emb = emb / emb.norm(dim=-1, keepdim=True)
        self.vad = load_silero_vad()

    @torch.no_grad()
    def person_scores(self, frames: list[Image.Image], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(frames), batch):
            imgs = [self.det_tf(f).to(self.device) for f in frames[i : i + batch]]
            for pred in self.detector(imgs):
                s = pred["scores"][pred["labels"] == 1]  # COCO label 1 = person
                out.append(float(s.max()) if len(s) else 0.0)
        return np.array(out, dtype=np.float32)

    @torch.no_grad()
    def scene_scores(self, frames: list[Image.Image], batch: int = 32) -> np.ndarray:
        out = []
        n_active = len(ACTIVE_PROMPTS)
        for i in range(0, len(frames), batch):
            px = self.clip_proc(images=frames[i : i + batch], return_tensors="pt").to(
                self.device
            )
            emb = self.clip.get_image_features(**px)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (100.0 * emb @ self.text_emb.T).softmax(dim=-1)
            out.extend(probs[:, :n_active].sum(dim=-1).tolist())
        return np.array(out, dtype=np.float32)

    def speech_scores(self, audio: np.ndarray, n: int) -> np.ndarray:
        if len(audio) == 0:
            return np.zeros(n, dtype=np.float32)
        self.vad.reset_states()
        win = 512  # Silero's frame size at 16 kHz (32 ms)
        probs = []
        t = torch.from_numpy(audio)
        for i in range(0, len(t) - win + 1, win):
            probs.append(float(self.vad(t[i : i + win], AUDIO_SR)))
        probs_a = np.array(probs, dtype=np.float32)
        per_sample = AUDIO_SR / SAMPLE_FPS / win
        out = np.zeros(n, dtype=np.float32)
        for k in range(n):
            seg = probs_a[int(k * per_sample) : int((k + 1) * per_sample)]
            out[k] = float(np.percentile(seg, 75)) if len(seg) else 0.0
        return out


def motion_scores(frames: list[Image.Image]) -> np.ndarray:
    small = [
        np.asarray(f.convert("L").resize((160, 90)), dtype=np.float32) for f in frames
    ]
    diffs = [0.0] + [
        float(np.mean(np.abs(b - a))) for a, b in itertools.pairwise(small)
    ]
    return np.tanh(np.array(diffs, dtype=np.float32) / 6.0)


def rolling_mean(x: np.ndarray, seconds: float) -> np.ndarray:
    k = max(1, int(seconds * SAMPLE_FPS))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def fuse(signals: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    z = np.full(len(signals["person"]), weights["bias"], dtype=np.float32)
    for name in SIGNALS:
        z += weights[name] * signals[name]
    return 1.0 / (1.0 + np.exp(-z))


def smooth_and_segment(
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """EMA, then hysteresis: switch to active above ENTER_ACTIVE, to break below ENTER_BREAK."""
    alpha = 1.0 - math.exp(-1.0 / (EMA_SECONDS * SAMPLE_FPS))
    ema = np.empty_like(p)
    state = np.empty(len(p), dtype=np.int8)
    acc, cur = float(p[0]) if len(p) else 0.5, 1
    for i, v in enumerate(p):
        acc += alpha * (float(v) - acc)
        ema[i] = acc
        if cur == 0 and acc > ENTER_ACTIVE:
            cur = 1
        elif cur == 1 and acc < ENTER_BREAK:
            cur = 0
        state[i] = cur
    # Fold segments shorter than MIN_SEGMENT_SECONDS into their predecessor so blips do not show.
    segs: list[dict[str, Any]] = []
    i = 0
    while i < len(state):
        j = i
        while j < len(state) and state[j] == state[i]:
            j += 1
        label = "active" if state[i] == 1 else "break"
        if segs and (
            (j - i) / SAMPLE_FPS < MIN_SEGMENT_SECONDS or segs[-1]["label"] == label
        ):
            segs[-1]["end"] = j / SAMPLE_FPS
        else:
            segs.append(
                {"start": i / SAMPLE_FPS, "end": j / SAMPLE_FPS, "label": label}
            )
        i = j
    # A too-short first segment has no predecessor; it is warm-up before any window has filled.
    if len(segs) > 1 and segs[0]["end"] - segs[0]["start"] < MIN_SEGMENT_SECONDS:
        segs[1]["start"] = 0.0
        segs.pop(0)
    for s in segs:
        state[int(s["start"] * SAMPLE_FPS) : int(s["end"] * SAMPLE_FPS)] = (
            1 if s["label"] == "active" else 0
        )
    return ema, state, segs


def extract_raw(
    models: Models, decoded: Decoded
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    timings: dict[str, float] = {}
    n = len(decoded.frames)
    t0 = time.perf_counter()
    person = models.person_scores(decoded.frames)
    timings["person_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    scene = models.scene_scores(decoded.frames)
    timings["scene_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    speech = models.speech_scores(decoded.audio, n)
    timings["speech_ms"] = (time.perf_counter() - t0) * 1000
    motion = motion_scores(decoded.frames)
    return {
        "person": person,
        "scene": scene,
        "speech": speech,
        "motion": motion,
    }, timings


def rolling_max(x: np.ndarray, seconds: float) -> np.ndarray:
    k = max(1, int(seconds * SAMPLE_FPS))
    return np.array(
        [x[max(0, i - k + 1) : i + 1].max() for i in range(len(x))], dtype=np.float32
    )


def windowed(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in raw.items():
        how, seconds = WINDOWS[k]
        out[k] = rolling_max(v, seconds) if how == "max" else rolling_mean(v, seconds)
    return out


def extract(
    models: Models, decoded: Decoded
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    raw, timings = extract_raw(models, decoded)
    return windowed(raw), timings


def analyze(
    models: Models, decoded: Decoded, weights: dict[str, float]
) -> dict[str, Any]:
    t0 = time.perf_counter()
    signals, timings = extract(models, decoded)
    p = fuse(signals, weights)
    ema, state, segs = smooth_and_segment(p)
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    r = lambda a: [round(float(x), 3) for x in a]
    return {
        "sample_fps": SAMPLE_FPS,
        "duration_s": round(decoded.duration_s, 2),
        "signals": {k: r(v) for k, v in signals.items()},
        "p_active": r(p),
        "smoothed": r(ema),
        "state": [int(s) for s in state],
        "segments": [
            {**s, "start": round(s["start"], 2), "end": round(s["end"], 2)}
            for s in segs
        ],
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "weights": weights,
    }


def evaluate(result: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """Per-sample accuracy against ground-truth segments, ignoring the scripted transition shots."""
    state = result["state"]
    correct = total = 0
    for i, s in enumerate(state):
        t = i / SAMPLE_FPS
        seg = next((g for g in labels["segments"] if g["start"] <= t < g["end"]), None)
        if seg is None or seg["label"] == "transition":
            continue
        total += 1
        correct += int((s == 1) == (seg["label"] == "active"))
    return {"accuracy": round(correct / total, 4) if total else None, "samples": total}

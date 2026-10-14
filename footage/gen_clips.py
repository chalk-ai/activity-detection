#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Generate the fictional streamer's shots with an LTX-Video deployment, one job at a time.

footage/gen_clips.py clips/ https://<your text-to-video web_url>
"""

import json
import pathlib
import sys
import time

import httpx

# The web_url of your own deployment of github.com/chalk-ai/text-to-video.
BASE = sys.argv[2].rstrip("/")
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CHAR = (
    "Webcam view, static camera at eye level, of a young adult streamer with short curly dark hair, "
    "large white over-ear headphones and a mustard yellow hoodie, sitting in a black gaming chair at a desk "
    "in a small bedroom studio lit by purple and teal LED light strips, a black microphone on a boom arm in front of them."
)
ROOM = (
    "Webcam view, static camera at eye level, of an empty black gaming chair at a desk in a small bedroom studio "
    "lit by purple and teal LED light strips, a black microphone on a boom arm in front of the chair, nobody in the room."
)
SHOTS = {
    "talk_a": CHAR
    + " They talk cheerfully to the camera, smiling and gesturing with both hands.",
    "talk_b": CHAR
    + " They laugh and talk to the camera, leaning slightly toward the microphone.",
    "play": CHAR
    + " They concentrate on the monitor below the camera, playing a video game with one hand on the mouse, eyes moving quickly.",
    "leave": CHAR
    + " They take off the headphones, stand up from the chair and walk out of frame to the left, leaving the chair spinning slowly.",
    "empty": ROOM + " The chair is still, the LED lights glow softly, nothing moves.",
    "return": ROOM
    + " A young adult with short curly dark hair in a mustard yellow hoodie walks into frame from the left, sits down in the chair and puts on white headphones.",
    "phone": CHAR
    + " They lean back in the chair looking down at a phone held in both hands, quiet and not talking, barely moving.",
    "drink": CHAR
    + " They sip from a white mug, put it down and start talking to the camera with a smile.",
}
SEEDS = [1234, 98765]


def run(c: httpx.Client, name: str, prompt: str, seed: int) -> None:
    dst = OUT / f"{name}_{seed}.mp4"
    if dst.exists():
        return
    while True:
        r = c.post(f"{BASE}/api/generate", json={"prompt": prompt, "seed": seed})
        if r.status_code in (429, 503):
            time.sleep(15)
            continue
        r.raise_for_status()
        job = r.json()
        break
    t0 = time.time()
    while job["status"] not in ("done", "error", "failed"):
        time.sleep(5)
        job = c.get(f"{BASE}/api/jobs/{job['job_id']}").json()
    if job["status"] != "done":
        print(name, seed, "FAILED", job.get("error"), flush=True)
        return
    dst.write_bytes(c.get(f"{BASE}/{job['video_url']}").content)
    print(name, seed, f"{time.time() - t0:.0f}s", dst.stat().st_size, flush=True)


with httpx.Client(timeout=120) as c:
    for seed in SEEDS:
        for name, prompt in SHOTS.items():
            run(c, name, prompt, seed)
(OUT / "prompts.json").write_text(
    json.dumps(
        {
            "shots": SHOTS,
            "seeds": SEEDS,
            "model": "Lightricks/LTX-Video-0.9.8-13B-distilled",
        },
        indent=2,
    )
)
print("done", flush=True)

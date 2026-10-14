"""Compose a fictional livestream recording from generated shots, with ground-truth labels.

Usage: compose.py <clips_dir> <seed> <out_dir> <timeline>   (timeline: demo | calib)
Writes <out_dir>/<timeline>.mp4 (1280x720, 24 fps, H.264 + AAC) and <timeline>.labels.json.
"""

import json
import math
import pathlib
import subprocess
import sys

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FPS = 24
W, H = 1280, 720
VW, VH = 1080, 720  # stream video area; 768x512 shots scale exactly 1.40625x
SR = 24000
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CHANNEL = "mothlight"

clips_dir, seed, out_dir, timeline = (
    pathlib.Path(sys.argv[1]),
    sys.argv[2],
    pathlib.Path(sys.argv[3]),
    sys.argv[4],
)
out_dir.mkdir(parents=True, exist_ok=True)


def shot(name: str) -> list[np.ndarray]:
    frames = iio.imread(clips_dir / f"{name}_{seed}.mp4", index=None, plugin="pyav")
    return [f for f in frames]


def pingpong(frames: list[np.ndarray], seconds: float) -> list[np.ndarray]:
    """Loop a shot forward and backward until it fills `seconds` (talking/sitting shots read fine reversed)."""
    out: list[np.ndarray] = []
    seq = frames + frames[-2:0:-1]
    while len(out) < seconds * FPS:
        out.extend(seq)
    return out[: int(seconds * FPS)]


def blur_box(
    frames: list[np.ndarray], box: tuple[int, int, int, int]
) -> list[np.ndarray]:
    out = []
    for f in frames:
        im = Image.fromarray(f)
        region = im.crop(box).filter(ImageFilter.GaussianBlur(28))
        im.paste(region, box)
        out.append(np.asarray(im))
    return out


def brb_frames(seconds: float) -> list[np.ndarray]:
    n = int(seconds * FPS)
    big = ImageFont.truetype(FONT_B, 68)
    sub = ImageFont.truetype(FONT, 30)
    out = []
    for i in range(n):
        t = i / FPS
        im = Image.new("RGB", (768, 512))
        d = ImageDraw.Draw(im)
        for y in range(512):  # slow purple-to-teal gradient drift
            a = 0.5 + 0.5 * math.sin(y / 140 + t * 0.6)
            d.line(
                [(0, y), (768, y)],
                fill=(
                    int(70 + 60 * a),
                    int(30 + 90 * (1 - a)),
                    int(120 + 60 * (1 - a)),
                ),
            )
        d.text((384, 220), "BE RIGHT BACK", font=big, fill=(245, 240, 255), anchor="mm")
        dots = "." * (1 + int(t * 2) % 3)
        d.text(
            (384, 320),
            f"grabbing water{dots}",
            font=sub,
            fill=(220, 210, 240),
            anchor="mm",
        )
        out.append(np.asarray(im))
    return out


# --- timeline -------------------------------------------------------------------------
# Each entry: (frames, label, kind, speech lines [(offset_s, text)]).
if timeline == "demo":
    play = blur_box(shot("play"), (500, 380, 600, 470))  # the laptop lid's logo
    SEGMENTS = [
        (
            pingpong(shot("talk_a"), 14),
            "active",
            "talking",
            [
                (0.6, "Hey hey, welcome back to the stream, everybody!"),
                (5.5, "Today we're finally trying the no damage run."),
                (10.0, "Thanks for the follows, I see you in chat."),
            ],
        ),
        (
            pingpong(play, 12),
            "active",
            "gaming",
            [
                (1.5, "Okay, okay. Focus."),
                (5.5, "Dodge, dodge... nice!"),
                (9.0, "Let's go, that's phase two."),
            ],
        ),
        (
            pingpong(shot("talk_b"), 10),
            "active",
            "talking",
            [
                (0.5, "Chat, that was so close!"),
                (4.5, "Alright, I need some water. I'll be right back."),
            ],
        ),
        (shot("leave"), "transition", "leaving", []),
        (pingpong(shot("empty"), 16), "break", "away", []),
        (brb_frames(16), "break", "brb_screen", []),
        (shot("return"), "transition", "returning", []),
        (
            pingpong(shot("drink"), 14),
            "active",
            "talking",
            [
                (0.8, "Okay, I'm back. Did I miss anything?"),
                (6.0, "Someone said the boss has a third phase? No way."),
            ],
        ),
        (pingpong(shot("phone"), 14), "break", "idle", []),
        (
            pingpong(shot("talk_a"), 14),
            "active",
            "talking",
            [
                (0.5, "Alright, one more attempt and then we call it."),
                (6.5, "Seriously, thanks for hanging out tonight."),
            ],
        ),
    ]
else:  # calibration recording from the other seed, in a different order, used only to fit the fusion weights
    SEGMENTS = [
        (
            pingpong(shot("talk_b"), 12),
            "active",
            "talking",
            [
                (0.5, "What's up chat, good to see everyone."),
                (5.0, "Let me know what we should play next."),
            ],
        ),
        (pingpong(shot("empty"), 12), "break", "away", []),
        (
            pingpong(shot("play"), 12),
            "active",
            "gaming",
            [(2.0, "Come on, come on."), (7.0, "Yes! Got it.")],
        ),
        (pingpong(shot("leave"), 10), "break", "idle", []),
        (brb_frames(12), "break", "brb_screen", []),
        (
            pingpong(shot("drink"), 12),
            "active",
            "talking",
            [(1.0, "Sorry about that, I'm back."), (6.0, "Okay where were we.")],
        ),
        (pingpong(shot("phone"), 12), "break", "idle", []),
        (
            pingpong(shot("talk_a"), 12),
            "active",
            "talking",
            [
                (0.5, "Last question from chat."),
                (5.0, "That's it for today, see you tomorrow."),
            ],
        ),
    ]

frames: list[np.ndarray] = []
labels = []
speech: list[tuple[float, str]] = []
for seg_frames, label, kind, lines in SEGMENTS:
    start = len(frames) / FPS
    frames.extend(seg_frames)
    end = len(frames) / FPS
    labels.append(
        {"start": round(start, 3), "end": round(end, 3), "label": label, "kind": kind}
    )
    speech.extend((start + off, text) for off, text in lines)
duration = len(frames) / FPS

# --- audio ----------------------------------------------------------------------------
from kokoro import (
    KPipeline,
)

tts = KPipeline(lang_code="a")
rng = np.random.default_rng(7)
audio = (rng.standard_normal(int(duration * SR)) * 0.0015).astype(
    np.float32
)  # room tone
for at, text in speech:
    clip = np.concatenate(
        [np.asarray(a) for _, _, a in tts(text, voice="am_puck")]
    ).astype(np.float32)
    i = int(at * SR)
    audio[i : i + len(clip)] += 0.9 * clip[: len(audio) - i]
for seg in (
    labels
):  # soft synth loop under the BRB card, so music is present but is not speech
    if seg["kind"] != "brb_screen":
        continue
    i0, i1 = int(seg["start"] * SR), int(seg["end"] * SR)
    t = np.arange(i1 - i0) / SR
    chords = [
        (220.0, 277.2, 329.6),
        (196.0, 246.9, 293.7),
        (174.6, 220.0, 261.6),
        (196.0, 246.9, 329.6),
    ]
    pad = np.zeros_like(t)
    for k, chord in enumerate(chords * 8):
        a, b = k * 2.0, (k + 1) * 2.0
        m = (t >= a) & (t < b)
        env = np.sin(np.pi * (t[m] - a) / 2.0) ** 2
        pad[m] = sum(np.sin(2 * np.pi * f * t[m]) for f in chord) * env
    beat = ((t % 0.5) < 0.04) * rng.standard_normal(len(t)) * 0.3
    audio[i0:i1] += (0.05 * pad + 0.02 * beat).astype(np.float32)
audio = np.clip(audio, -1, 1)
wav = out_dir / f"{timeline}.wav"
sf.write(wav, audio, SR)

# --- video ----------------------------------------------------------------------------
title_f = ImageFont.truetype(FONT_B, 22)
small_f = ImageFont.truetype(FONT, 16)
mono_f = ImageFont.truetype(FONT_M, 15)
CHAT = [
    ("pixelfern", "hiii"),
    ("quietkoi", "lets gooo"),
    ("bramble_7", "no damage run??"),
    ("ottoloop", "you got this"),
    ("mintwave", "first time here, hi"),
    ("pixelfern", "that dodge was clean"),
    ("dustyowl", "phase 2 already"),
    ("quietkoi", "hydrate!!"),
    ("larkspur99", "brb gang"),
    ("mintwave", "lofi bangs"),
    ("ottoloop", "welcome back"),
    ("dustyowl", "third phase is real"),
    ("bramble_7", "one more!"),
    ("larkspur99", "gg"),
]
vtmp = out_dir / f"{timeline}.video.mp4"
writer = imageio_ffmpeg.write_frames(
    str(vtmp),
    (W, H),
    fps=FPS,
    codec="libx264",
    quality=None,
    bitrate=None,
    output_params=["-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p"],
    macro_block_size=8,
)
writer.send(None)
for i, f in enumerate(frames):
    t = i / FPS
    canvas = Image.new("RGB", (W, H), (18, 16, 26))
    canvas.paste(Image.fromarray(f).resize((VW, VH), Image.BILINEAR), (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((16, 16, 82, 44), 6, fill=(214, 44, 64))
    d.text((49, 30), "LIVE", font=small_f, fill="white", anchor="mm")
    d.rounded_rectangle((90, 16, 330, 44), 6, fill=(0, 0, 0, 160))
    viewers = 1180 + int(40 * math.sin(t / 9)) + int(t * 1.5)
    d.text(
        (100, 30),
        f"{CHANNEL}  ·  {viewers:,} watching",
        font=small_f,
        fill=(235, 235, 245),
        anchor="lm",
    )
    d.text(
        (VW - 16, 30),
        f"{int(t // 3600):01d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}",
        font=mono_f,
        fill=(235, 235, 245),
        anchor="rm",
    )
    # chat column
    d.text(
        (VW + 16, 28), "STREAM CHAT", font=small_f, fill=(170, 160, 200), anchor="lm"
    )
    shown = [m for k, m in enumerate(CHAT) if t >= 2 + k * (duration - 6) / len(CHAT)][
        -14:
    ]
    y = 60
    for user, msg in shown:
        d.text((VW + 16, y), user, font=small_f, fill=(150, 220, 190))
        d.text((VW + 16, y + 19), msg, font=small_f, fill=(225, 222, 235))
        y += 46
    writer.send(np.asarray(canvas).tobytes())
writer.close()

out = out_dir / f"{timeline}.mp4"
subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(vtmp),
        "-i",
        str(wav),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out),
    ],
    check=True,
)
vtmp.unlink()
(out_dir / f"{timeline}.labels.json").write_text(
    json.dumps(
        {
            "duration_s": round(duration, 3),
            "fps": FPS,
            "segments": labels,
            "speech": [{"at": round(a, 2), "text": s} for a, s in speech],
        },
        indent=2,
    )
)
print(timeline, f"{duration:.1f}s", out.stat().st_size // 1024, "KB")

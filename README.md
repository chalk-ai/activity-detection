# Livestream activity detection on Chalk

Is the streamer live, or on a break? This example scores a stream recording four times a second
and plots how confident it is that the streamer is at the desk and engaged. It runs as one GPU
replica deployed with [`chalkcompute`](https://pypi.org/project/chalkcompute/), which also serves
the page: the recording plays in sync with a scrolling confidence plot, and you can analyze a
short clip of your own.

```
frames (4/s) ──▶ object detector ─▶ person (mean 2 s) ┐
             ──▶ CLIP zero-shot   ─▶ scene  (mean 2 s) ├─▶ logistic fusion ─▶ EMA + hysteresis ─▶ active / break
audio  16kHz ──▶ Silero VAD       ─▶ speech (max 6 s)  │
frames       ──▶ frame difference ─▶ motion (mean 4 s) ┘
```

## Files

| Path | Role |
| --- | --- |
| `pipeline.py` | Decoding, the four signals, fusion, smoothing and segmentation. |
| `server.py` | FastAPI app: analyzes the sample recording at boot, serves the page, and takes uploads. |
| `assets/` | The page, the sample recording with its ground-truth script, and the fitted weights. |
| `calibrate.py` | Fits the fusion weights on `media/calib.mp4` and reports agreement on the sample recording. |
| `deploy.py` | Builds the image (weights baked in) and deploys the ScalingGroup on an L4. |
| `footage/` | How the fictional streamer recordings were generated (see below). |

## Run it

```bash
./deploy.py --gpu nvidia-l4
```

Credentials come from `chalk config`, or set `CHALK_API_SERVER`, `CHALK_CLIENT_ID`,
`CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID`. Open the printed `web_url`.

## API

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/api/recording` | The sample recording's analysis, computed on the replica at boot, with agreement against its script. |
| `POST` | `/api/analyze` | Multipart field `clip`, up to 15 MB and 30 seconds. Per-sample signals, `p_active`, `smoothed`, `state` and `segments`. |

Uploads are written to a temporary file, analyzed, and deleted before the response returns. Each
visitor gets one analysis at a time and ten per hour.

## How the signals are fused

Each raw signal is summarized over a trailing window: presence (person, scene) as a 2-second
mean, so breaks register quickly; speech as a 6-second max, meaning "spoke recently", which is
what separates a streamer scrolling a phone (present, silent) from a pause between sentences; and
motion as a 4-second mean. Then

    P(active) = sigmoid(bias + w_person·person + w_scene·scene + w_speech·speech + w_motion·motion)

with five weights fit by logistic regression on a separate labeled recording (`media/calib.mp4`),
never on the sample stream the page shows. An exponential moving average (1.5 s) and hysteresis
(switch to a break below 0.4, back to active above 0.6, ignore segments under 2 s) turn the
probability into segments. Scripted transition shots (standing up, sitting down) are left out of
both the fit and the agreement score.

## The footage

```bash
footage/gen_clips.py clips/ https://<your text-to-video web_url>   # 16 shots, two seeds
uv run --with kokoro --with "misaki[en]" --with soundfile --with imageio --with imageio-ffmpeg \
  --with av --with pillow --with numpy python footage/compose.py clips 1234 out demo
```

The streamer, the room and the chat are fictional. The video shots were generated with
[LTX-Video](https://github.com/Lightricks/LTX-Video) (`Lightricks/LTX-Video-0.9.8-13B-distilled`),
running on Chalk, from the prompts in `footage/prompts.json`. The narration was synthesized with
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), and the be-right-back card,
overlays, chat and music were rendered by `footage/compose.py`, which also writes the
ground-truth script for each recording.

## License

The code is Apache-2.0. Model weights keep their own licenses: torchvision's COCO detector,
OpenAI CLIP ViT-B/32 (MIT), Silero VAD (MIT).

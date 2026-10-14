"""Livestream activity detection served from one GPU.

GET  /              the demo page (?embed=1 hides its own title for iframing)
GET  /status        readiness: 200 once the models are loaded and the recording is analyzed
GET  /media/<name>  the bundled recording, with range requests for seeking
GET  /api/recording the recording's analysis, computed on this replica at boot
POST /api/analyze   multipart "clip": analyze a short upload; nothing is kept
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ASSETS = Path(os.environ.get("ASSETS_DIR", "/assets"))
sys.path.insert(0, str(Path(__file__).parent))
import pipeline

PORT = int(os.environ.get("PORT", "8080"))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_SECONDS = 30.0
UPLOADS_PER_HOUR = 10
CONCURRENT_ANALYSES = 2

state: dict[str, Any] = {}


def load() -> None:
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = pipeline.Models(device)
    weights = json.loads((ASSETS / "weights.json").read_text())
    labels = json.loads((ASSETS / "demo.labels.json").read_text())
    loaded = time.perf_counter()
    result = pipeline.analyze(
        models, pipeline.decode(ASSETS / "demo.mp4", 600), weights
    )
    result["evaluation"] = pipeline.evaluate(result, labels)
    result["labels"] = labels["segments"]
    result["device"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    result["model_load_s"] = round(loaded - t0, 1)
    state.update(models=models, weights=weights, recording=result)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(load)
    state["slots"] = asyncio.Semaphore(CONCURRENT_ANALYSES)
    state["recent"] = collections.defaultdict(collections.deque)
    state["busy"] = set()
    yield


app = FastAPI(lifespan=lifespan)


def client_address(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((ASSETS / "index.html").read_text())


@app.get("/status")
async def status() -> JSONResponse:
    rec = state["recording"]
    return JSONResponse(
        {"ok": True, "device": rec["device"], "accuracy": rec["evaluation"]["accuracy"]}
    )


@app.get("/media/{name}")
async def media(name: str) -> FileResponse:
    if name not in ("demo.mp4", "poster.jpg"):
        raise HTTPException(404)
    return FileResponse(
        ASSETS / name, headers={"Cache-Control": "public, max-age=3600"}
    )


@app.get("/api/recording")
async def recording() -> JSONResponse:
    return JSONResponse(state["recording"])


@app.post("/api/analyze")
async def analyze(clip: UploadFile, request: Request) -> JSONResponse:
    client = client_address(request)
    now = time.time()
    recent: collections.deque[float] = state["recent"][client]
    while recent and now - recent[0] > 3600:
        recent.popleft()
    if len(recent) >= UPLOADS_PER_HOUR:
        raise HTTPException(
            429, "That's the hourly limit for uploads. Try again later."
        )
    if client in state["busy"]:
        raise HTTPException(429, "Your previous clip is still being analyzed.")
    state["busy"].add(client)
    recent.append(now)
    # The clip only ever exists in this temp file, which is deleted before the response returns.
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        try:
            size = 0
            while chunk := await clip.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Clips are limited to 15 MB.")
                tmp.write(chunk)
            tmp.flush()
            async with state["slots"]:
                try:
                    decoded = await asyncio.to_thread(
                        pipeline.decode, Path(tmp.name), MAX_UPLOAD_SECONDS
                    )
                except Exception as e:
                    raise HTTPException(
                        400, "Couldn't read that file as a video."
                    ) from e
                if len(decoded.frames) < pipeline.SAMPLE_FPS * 2:
                    raise HTTPException(400, "Clips need at least 2 seconds of video.")
                result = await asyncio.to_thread(
                    pipeline.analyze, state["models"], decoded, state["weights"]
                )
        finally:
            state["busy"].discard(client)
    result["truncated"] = decoded.duration_s >= MAX_UPLOAD_SECONDS - 0.5
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

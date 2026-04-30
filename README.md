# neme-extractor

Pulls character-LoRA training crops out of a video. Give it a clip and one or more reference images of an anime character; it returns rectangular crops of just that character, sized for kohya-ss / OneTrainer / sd-scripts on SDXL-class anime models (Pony, Illustrious, NoobAI, vanilla SDXL).

## Pipeline

For each video:

1. PySceneDetect splits it into shots.
2. DeepGHS YOLO (via `imgutils`) detects characters per frame.
3. ByteTrack links detections into per-shot tracklets.
4. CCIP matches tracklets to your reference images.
5. 1–3 frames per kept tracklet are picked by sharpness, visibility, and aspect ratio.
6. Each pick is cropped at longest-side 1024 with the original background.
7. WD14 EVA02-Large v3 writes a kohya-style `.txt` next to each `.png`.

Detections and tracklets are cached so threshold re-runs skip the slow stages.

## Requirements

- Linux / WSL2 with CUDA 12.4+
- NVIDIA GPU, 16 GB VRAM minimum, 24 GB comfortable
- Python 3.11

## Install

```sh
uv sync --group gpu
```

First run downloads ~2.8 GB of weights (anime YOLOv8 person + face, CCIP, isnetis/anime-seg, WD14 with embeddings, CLIP base) to `~/.cache/huggingface/hub/`.

Override the cache location:

```sh
HF_HUB_CACHE=/mnt/c/Users/<you>/.cache/huggingface/hub uv run neme-extractor project extract ...
```

## CLI

```sh
uv run neme-extractor project create ~/neme-projects/megumin --name megumin
uv run neme-extractor project add-ref ~/neme-projects/megumin /path/to/portrait.png
uv run neme-extractor project add-video ~/neme-projects/megumin /path/to/ep01.mkv
uv run neme-extractor project add-video ~/neme-projects/megumin /path/to/ep02.mkv
uv run neme-extractor project extract ~/neme-projects/megumin
```

Project folder layout:

```
~/neme-projects/megumin/
  project.json
  refs/
  output/
    kept/             ep01__s003_t012_f000847.png + .txt
    rejected/
    metadata.jsonl
    cache/<stem>/     scenes.parquet, tracklets.parquet
```

Re-run with new thresholds (skips detection + tracking):

```sh
uv run neme-extractor project rerun ~/neme-projects/megumin --video ep01
```

## Web UI

```sh
uv run neme-extractor ui
```

Binds to `127.0.0.1:<random-port>` and opens the Svelte SPA. Tabs: Sources, Frames, Training, Settings.

- **Sources** — add MKV/MP4 videos and reference images, opt out of refs per video, run extraction.
- **Frames** — hover thumbnails for tag overlay; click pills to edit inline. Shift-click ranges, Ctrl-click multi-toggle, `A` select all, `D` / `Esc` deselect. Bulk regex replace with live preview.
- **Training** — LoRA training run with stop/resume and checkpoint retention.
- **Settings** — per-project threshold overrides (frame stride, identification distance, crop padding, etc.).

## REST API

- `/api/projects`, `/api/projects/<slug>/sources`, `/api/projects/<slug>/refs`, `/api/projects/<slug>/frames`, `/api/projects/<slug>/training`, `/api/queue`
- WebSocket at `/api/ws` streaming `queue.update` / `job.progress` / `job.frame` / `job.log` / `job.done`
- Health probe at `/api/health`

```sh
curl -s http://127.0.0.1:<port>/api/projects | jq
curl -s -X POST http://127.0.0.1:<port>/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"name": "megumin", "folder": "/home/me/projects/megumin"}'
curl -s -X POST http://127.0.0.1:<port>/api/projects/megumin/sources/0/extract
```

Project state lives in the project folder. The only server-side file is `~/.neme-extractor/db.sqlite` (project registry: names, paths, last-opened timestamps).

## Performance

A 20-min episode at 24 fps runs in 2–3 minutes on a 4090. Tuning knobs:

- `detect.frame_stride` (default 4) — every Nth frame.
- `detect.detect_faces` (default `false`) — face boxes; unused by the current matcher.
- `frame_select.candidate_cap` (default 20) — long tracklets get downsampled to this many evenly-spaced candidates before ranking.
- `frame_select.dedup_min_frame_gap` — minimum frames between picks within one tracklet.

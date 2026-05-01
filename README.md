# Neme-Anima

A three-step LoRA builder for anime characters:

1. Extract crops of a target character from a video using reference images.
2. Auto-tag each crop with WD14 danbooru tags and natural-language captions, then reorganize the dataset from the UI.
3. Train a LoRA on Anima with the parameters already wired in.

The extractor and tagger are model-agnostic and produce output sized for kohya-ss / OneTrainer / sd-scripts on SDXL-class anime models (Pony, Illustrious, NoobAI, vanilla SDXL). The trainer targets Anima.

<p align="center"><img src="docs/chie.png" alt="Result with a Chie LoRA applied to Anima" width="50%"></p>

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
HF_HUB_CACHE=/mnt/c/Users/<you>/.cache/huggingface/hub uv run neme-anima project extract ...
```

## CLI

```sh
uv run neme-anima project create ~/neme-projects/megumin --name megumin
uv run neme-anima project add-ref ~/neme-projects/megumin /path/to/portrait.png
uv run neme-anima project add-video ~/neme-projects/megumin /path/to/ep01.mkv
uv run neme-anima project add-video ~/neme-projects/megumin /path/to/ep02.mkv
uv run neme-anima project extract ~/neme-projects/megumin
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
uv run neme-anima project rerun ~/neme-projects/megumin --video ep01
```

## Web UI

```sh
uv run neme-anima ui
```

Binds to `127.0.0.1:<random-port>` and opens the Svelte SPA. Tabs: Sources, Frames, Training, Settings.

### Sources

Add MKV/MP4 videos and reference images, opt out of refs per video, run extraction.

![Sources tab](docs/neme-anima_extract.png)

### Frames

- Add or remove images from the dataset.
- Edit tags inline by clicking a pill; edit the natural-language description in the same panel.
- Search across the dataset by tag.
- Bulk-edit tags with regex replace, with live preview.
- Re-crop any image.

Selection: shift-click ranges, ctrl-click multi-toggle, `A` select all, `D` / `Esc` deselect. Hover a thumbnail for the tag overlay.

![Frames tab](docs/neme-anima_frames.png)

### Training

LoRA training with stop/resume and checkpoint retention. Targets Anima.

![Training tab](docs/neme-anima_train.png)

### Settings

Per-project threshold overrides (frame stride, identification distance, crop padding, etc.).

## REST API

- `/api/projects`, `/api/projects/<slug>/sources`, `/api/projects/<slug>/refs`, `/api/projects/<slug>/frames`, `/api/projects/<slug>/training`, `/api/queue`
- WebSocket at `/api/ws` streaming `queue.update` / `job.progress` / `job.frame` / `job.log` / `job.done`
- Health probe at `/api/health`

Project state lives in the project folder. The only server-side file is `~/.neme-anima/db.sqlite` (project registry).

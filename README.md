# neme-extractor

Pulls character-LoRA training crops out of a video. You give it a clip and one or more reference images of an anime character; it returns rectangular crops of just that character, framed for kohya-ss / OneTrainer / sd-scripts on SDXL-class anime models (Pony, Illustrious, NoobAI, vanilla SDXL).

## How it works

For each video:

- PySceneDetect splits it into shots.
- DeepGHS YOLO (via `imgutils`) finds anime characters per frame.
- ByteTrack links per-frame detections into tracklets, scoped to a single shot.
- **CCIP** (Contrastive Character Image Pretraining) decides which tracklets are the character you asked for. CCIP is trained for outfit and pose invariance, so a single portrait reference usually covers a whole video — even when the character changes outfits.
- For each kept tracklet, 1–3 frames are picked by sharpness, visibility, and aspect ratio.
- Each pick is cropped at longest-side 1024 with the **original background intact**. No mask cutouts — diffusion training works best on natural images, and good captioning handles background leakage better than any mask trick.
- WD14 EVA02-Large v3 tags every crop into a kohya-style `.txt` next to the `.png`.
- Detections and tracklets are cached so you can re-tune thresholds later without re-running the slow stages.

## Requirements

- Linux / WSL2 with CUDA 12.4+
- NVIDIA GPU, 16 GB VRAM minimum, 24 GB comfortable
- Python 3.11

## Install

```sh
uv sync --group gpu
```

First run pulls about 2.8 GB of weights from HuggingFace (anime YOLOv8 person + face, CCIP, isnetis/anime-seg, WD14 with embeddings, CLIP base). Cached after that.

### Where models live

By default `~/.cache/huggingface/hub/`. If you already have a HuggingFace cache elsewhere — say a Windows host running ComfyUI — point at it:

```sh
HF_HUB_CACHE=/mnt/c/Users/<you>/.cache/huggingface/hub uv run neme-extractor project extract ...
```

This tool doesn't use SAM 3 even if you have it cached. `isnetis` from `imgutils` produces cleaner masks on anime and runs lighter. ComfyUI's `face_yolov8*.pt` weights aren't drop-in replacements either; `imgutils` resolves models through its own `deepghs/anime_*_detection` registry.

## Usage

Create a project folder, add videos and a reference image, then run extraction:

```sh
uv run neme-extractor project create ~/neme-projects/megumin --name megumin
uv run neme-extractor project add-ref ~/neme-projects/megumin /path/to/portrait.png
uv run neme-extractor project add-video ~/neme-projects/megumin /path/to/ep01.mkv
uv run neme-extractor project add-video ~/neme-projects/megumin /path/to/ep02.mkv
uv run neme-extractor project extract ~/neme-projects/megumin
```

The project folder you'll get:

```
~/neme-projects/megumin/
  project.json
  refs/
  output/
    kept/             ep01__s003_t012_f000847.png + .txt   (all videos, prefixed)
    rejected/
    metadata.jsonl    one line per kept/rejected frame across all runs
    cache/<stem>/     scenes.parquet, tracklets.parquet (per video)
```

Re-tune without re-running detection: edit `project.json`'s `thresholds_overrides`, then:

```sh
uv run neme-extractor project rerun ~/neme-projects/megumin --video ep01
```

This skips detection + tracking and is roughly 10× faster than a fresh extract.

## Web UI (backend)

A local web server is available alongside the CLI:

```sh
uv run neme-extractor ui
```

That binds to `127.0.0.1:<random-port>` and opens your browser. The full Svelte SPA lands in Phase 2B; right now this exposes the backend that drives it:

- REST endpoints under `/api/projects`, `/api/projects/<slug>/sources`, `/api/projects/<slug>/refs`, `/api/projects/<slug>/frames`, `/api/queue`
- WebSocket at `/api/ws` streaming `queue.update` / `job.progress` / `job.frame` / `job.log` / `job.done` events as JSON
- Health probe at `/api/health`

You can drive the same flows from `curl`:

```sh
curl -s http://127.0.0.1:<port>/api/projects | jq
curl -s -X POST http://127.0.0.1:<port>/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"name": "megumin", "folder": "/home/me/projects/megumin"}'
curl -s -X POST http://127.0.0.1:<port>/api/projects/megumin/sources/0/extract
```

Project state lives entirely in the project folder; the only server-side file is `~/.neme-extractor/db.sqlite` (the project registry — names, paths, last-opened timestamps).

## Performance

A 20-min episode at 24 fps lands around 2–3 minutes on a 4090. Four knobs if you want it faster:

- `detect.frame_stride` (default 4) — every 4th frame, so 6 fps effective. Push to 6 or 8 for dialogue-heavy content; drop to 3 if ByteTrack starts losing fast-action characters between frames.
- `detect.detect_faces` (default `false`) — face boxes aren't used by the current matcher. Flip on only if you wire up a face stream.
- `frame_select.candidate_cap` (default 20) — long tracklets get downsampled to this many evenly-spaced candidates before ranking.
- `frame_select.dedup_min_frame_gap` — minimum frames between picks within one tracklet, so you don't end up with three nearly-identical neighbouring frames.

## Design notes

The full design rationale (why CCIP, why rectangles instead of masks, why tracklet + scene sampling) lives in the planning doc. Two things changed during build-out:

- **`isnetis` replaces SAM 3** for mask generation. isnetis is anime-trained, ships with `imgutils`, and produces cleaner masks on anime than a general-purpose SAM. Masks stay internal; saved crops are rectangles either way.
- **One CCIP stream** instead of a face + body two-stream matcher. No anime-specific face embedder exists in the standard libraries, and CCIP on the body crop already absorbs the face/hair/eye signature. Adding a face stream later is easy if precision turns out to be a problem in practice.

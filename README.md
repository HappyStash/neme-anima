# Neme-Anima

A three-step character LoRA builder:

1. Extract crops of one or more characters from a video using reference images.
2. Auto-tag each crop with WD14 danbooru tags and natural-language captions, then reorganize the dataset from the UI.
3. Train a LoRA per character on Anima with the parameters already wired in.

The extractor and tagger are model-agnostic and produce output sized for kohya-ss / OneTrainer / sd-scripts on SDXL-class anime models (Pony, Illustrious, NoobAI, vanilla SDXL). The trainer targets Anima.

<p align="center"><img src="docs/chie.png" alt="Result with a Chie LoRA applied to Anima" width="50%"></p>

## Pipeline

For each video:

1. PySceneDetect splits it into shots.
2. DeepGHS YOLO (via `imgutils`) detects characters per frame.
3. ByteTrack links detections into per-shot tracklets.
4. CCIP matches tracklets to each character's reference images and assigns tracklets to whichever character scores best (or rejects them if no character matches).
5. 1–3 frames per kept tracklet are picked by sharpness, visibility, and aspect ratio.
6. Each pick is cropped at longest-side 1024 with the original background.
7. CCIP runs over the kept crops a second time to drop near-duplicates inside a sliding window.
8. WD14 EVA02-Large v3 writes a kohya-style `.txt` next to each `.png`.

Detections and tracklets are cached so threshold re-runs skip the slow stages.

## Requirements

For the extractor/tagger:

- NVIDIA GPU, 4 GB VRAM minimum, 8 GB comfortable

For the trainer:

- Linux / WSL2 with CUDA 12.4+
- NVIDIA GPU, 6 GB VRAM minimum, 18 GB for full res LoRA

## One-click install and run

```sh
bash install_and_run.sh
```

The script installs `uv` and Node.js if they aren't already on the system, syncs the Python deps, builds the frontend, clones `tdrussell/diffusion-pipe` into `~/diffusion-pipe` and sets up its Python 3.12 venv, downloads the three Anima training weights (~14 GB) from HuggingFace, prefills the four trainer paths in the UI's Settings tab, and starts the server.

Re-running it is safe (will skip anything already in place).

Useful environment overrides:

| Variable | Default | What it does |
|---|---|---|
| `DIFFUSION_PIPE_DIR` | `~/diffusion-pipe` | Where to clone diffusion-pipe |
| `DIFFUSION_PIPE_PYTHON` | `3.12` | Python interpreter/version for diffusion-pipe's venv |
| `MODELS_DIR` | `~/.cache/neme-anima/models` | Where to put the downloaded weights |
| `SKIP_MODELS=1` | off | Skip the 14 GB weight download |
| `SKIP_LAUNCH=1` | off | Install everything, but don't start the UI at the end |

Linux / WSL2 only. On Linux, Node.js is installed through apt (with sudo) when available, and through nvm otherwise.

## Manual install

```sh
uv sync --group gpu
```

First run downloads ~2.8 GB of weights (anime YOLOv8 person + face, CCIP, isnetis/anime-seg, WD14 with embeddings, CLIP base) to `~/.cache/huggingface/hub/`.

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

After cloning the repository:

```sh
cd frontend && npm install && npm run build && cd ..
```

Then start the server:

```sh
uv run neme-anima ui
```

Binds to `127.0.0.1:<random-port>` and opens the SPA. Tabs: Sources, Frames, Training, Settings.

### Sources

Add MKV/MP4 videos and reference images.

![Sources tab](docs/neme-anima_extract.png)

### Frames

- Add or remove images from the dataset (using drag&drop).
- Edit tags inline by clicking a pill; edit the natural-language description in the same panel.
- Search across the dataset by tag.
- Bulk-edit tags with regex replace, with live preview.
- Re-crop any image.
- Filter by character with the chips at the top of the tab. Move a frame to a different character or also-assign it to a second character.

Selection: shift-click ranges, ctrl-click multi-toggle, `A` select all, `D` / `Esc` deselect. Hover a thumbnail for the tag overlay.

![Frames tab](docs/neme-anima_frames.png)

### Training

LoRA training with stop/resume and checkpoint retention. Targets Anima. One LoRA per character, queued sequentially.

Two per-character knobs sit alongside the standard rank/alpha/lr settings:

- Core-tag pruning. Compute the tags that show up in more than X% of a character's frames (default 35%) and drop them from the captions at staging time. Turns "long hair, blue eyes, school uniform, smile, ..." into just "smile, ..." for a character whose hair, eyes, and outfit are constants. The LoRA learns those from the visuals; the caption only adds noise. Off by default; opt in once you've reviewed the suggested list.
- Repeat multiplier. Over- or under-sample a character's frames in the dataset. `0.0` is auto, computed from relative frame counts so a 50-frame character isn't drowned by a 500-frame one. A positive value pins it manually.

![Training tab](docs/neme-anima_train.png)

Training is run through [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe), which has to be set up separately:

```sh
git clone https://github.com/tdrussell/diffusion-pipe ~/diffusion-pipe
cd ~/diffusion-pipe
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Then in the Settings tab, point `diffusion_pipe_dir` at that clone and set the Anima DiT, Qwen VAE, and Qwen 3 0.6B text encoder paths (separate download on Huggingface).

### Settings

Per-project threshold overrides (frame stride, identification distance, crop padding, etc.).

Project state lives in the project folder. The only server-side file is `~/.neme-anima/db.sqlite` (project registry).

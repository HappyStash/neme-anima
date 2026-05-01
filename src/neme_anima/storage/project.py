"""Project: a folder of input videos + reference images + extracted output.

Layout under the project root:

    project.json
    refs/                    (link targets; thumbnails cached under .thumbnails/)
    output/
      kept/                  (all kept frames, prefixed with <video_stem>__)
      rejected/
      metadata.jsonl
      cache/<video_stem>/    (per-video detection cache, parquet)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTENSIONS = frozenset({
    ".mkv", ".mp4", ".webm", ".mov", ".avi", ".m4v", ".ts", ".wmv",
})

# Suffix appended to a kept frame's filename to mark its crop derivative
# (`<filename>_crop.png`). Defined here because the layout is shared by the
# API (which writes/deletes derivatives) and the trainer (which pairs each
# derivative with the original's `.txt` at staging time). There's only ever
# one derivative per original; re-cropping overwrites it.
CROP_SUFFIX = "_crop"


def refs_dir_contains(project_root: Path, candidate: Path) -> bool:
    """True iff ``candidate`` resolves to a file under ``project_root/refs/``."""
    try:
        candidate.resolve().relative_to((project_root / "refs").resolve())
        return True
    except (ValueError, OSError):
        return False


def list_videos(folder: Path) -> list[Path]:
    """Return a sorted list of video files directly under ``folder`` (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


@dataclass
class Source:
    """An input video tracked by the project."""
    path: str                         # absolute path to the video file
    added_at: str                     # ISO-8601 UTC
    excluded_refs: list[str] = field(default_factory=list)
    extraction_runs: list[dict] = field(default_factory=list)


@dataclass
class RefImage:
    """A reference image used for character matching."""
    path: str
    added_at: str


@dataclass
class LLMConfig:
    """Optional second-pass image-description tagger using an OpenAI-compatible
    chat-completions endpoint (LMStudio by default).

    ``enabled`` only flips on once the user has both pointed at a reachable
    server and picked a model from its discovery response — see
    :func:`neme_anima.llm.discover_models`. The pipeline treats
    ``enabled=False`` *or* ``model==""`` as off, so the disabled-by-default
    behaviour falls out without an extra check.

    ``api_key`` is empty by default — LMStudio doesn't require auth. Set it
    when targeting providers that gate ``/v1/models`` and ``/v1/chat/completions``
    behind a bearer token (OpenAI, OpenRouter, hosted vLLM, etc.).
    """
    enabled: bool = False
    endpoint: str = "http://localhost:1234"
    model: str = ""
    prompt: str = ""  # empty = use llm.DEFAULT_PROMPT
    api_key: str = ""  # empty = no Authorization header (LMStudio default)


@dataclass
class TrainingConfig:
    """Anima LoRA-training settings, persisted alongside the project.

    Defaults match the official tdrussell style-LoRA recipe (see
    docs/anima-lora-training-notes.md). Three groups: trainer paths
    (validated to actually exist on disk before a run is allowed to start),
    hyperparameters (faithful to the reference TOML), and captioning +
    checkpoint retention. ``keep_last_n_checkpoints == 0`` means
    "keep all" — the user-requested default.
    """

    preset: str = "style"  # "style" | "character"

    # Trainer paths — none of these are auto-downloaded.
    diffusion_pipe_dir: str = ""
    dit_path: str = ""        # anima-preview3-base.safetensors
    vae_path: str = ""        # qwen_image_vae.safetensors
    llm_path: str = ""        # qwen_3_06b_base.safetensors
    launcher_override: str = ""  # empty -> built-in deepspeed command

    # Adapter
    rank: int = 32
    alpha: int = 16  # kohya-only; ignored by canonical tdrussell schema

    # Optimizer / schedule
    learning_rate: float = 2e-5
    optimizer_betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    weight_decay: float = 0.01
    eps: float = 1e-8
    warmup_steps: int = 100
    gradient_clipping: float = 1.0

    # Batching
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # VRAM levers — usually only set by the "Fit in 8 GB" preset, hidden
    # from the regular form. Defaults match the recipe so the 24 GB-class
    # path stays byte-identical to its pre-preset output.
    #
    # ``transformer_dtype`` controls the storage dtype of the diffusion
    # transformer ("bfloat16" = recipe default, "float8" = ~half the VRAM
    # at a small quality cost — note: structurally broken for Anima,
    # rejected by validate_for_run, see comment there).
    # ``blocks_to_swap`` asks diffusion-pipe to keep N transformer blocks
    # on CPU and stream them in/out per step — heavy on CPU↔GPU traffic
    # but the difference between OOM and a usable run on small cards.
    # ``optimizer_type`` is the diffusion-pipe ``[optimizer] type`` value
    # — defaults to the recipe's "adamw_optimi"; "AdamW8bitKahan" is the
    # 8-bit-state variant that saves ~75% of optimizer-state VRAM (used
    # by the canonical wan_14b_min_vram example). The optimizer kwargs
    # (lr / betas / weight_decay / eps) flow through unchanged — train.py
    # builds the optimizer kwargs by stripping ``type`` and forwarding the
    # rest, so a swap is purely additive.
    # ``activation_checkpointing_mode`` chooses the recompute strategy:
    # "default" → ``activation_checkpointing = true`` (PyTorch native
    # checkpoint), "unsloth" → unsloth's more aggressive variant (less
    # GPU memory at a small CPU cost).
    transformer_dtype: str = "bfloat16"
    blocks_to_swap: int = 0
    optimizer_type: str = "adamw_optimi"
    activation_checkpointing_mode: str = "default"

    # Resolution / bucketing
    resolutions: list[int] = field(default_factory=lambda: [512, 1024])
    enable_ar_bucket: bool = True
    min_ar: float = 0.5
    max_ar: float = 2.0
    num_ar_buckets: int = 9

    # Duration
    epochs: int = 40
    eval_every_n_epochs: int = 5
    save_every_n_epochs: int = 10

    # Anima specifics — llm_adapter_lr=0 prevents style dilution per the
    # reference recipe; we expose it but the UI warns against changing it.
    sigmoid_scale: float = 1.3
    llm_adapter_lr: float = 0.0

    # Captioning
    caption_mode: str = "mixed"  # "tags" | "nl" | "mixed"
    tag_dropout_pct: int = 10
    trigger_token: str = ""

    # Retention: 0 = keep every checkpoint (the user-requested default).
    keep_last_n_checkpoints: int = 0


@dataclass
class Project:
    name: str
    slug: str
    root: Path
    created_at: datetime
    sources: list[Source] = field(default_factory=list)
    refs: list[RefImage] = field(default_factory=list)
    thresholds_overrides: dict = field(default_factory=dict)
    source_root: str | None = None
    # When True, extract/rerun pipelines pause after writing kept frames to
    # disk and wait for an explicit resume signal before tagging — giving the
    # user a chance to delete unwanted frames so they don't pay the tagging
    # cost on them. False = tag inline like the original pipeline.
    pause_before_tag: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ---------------- factory methods ----------------

    @classmethod
    def create(cls, root: Path, *, name: str) -> "Project":
        root = Path(root)
        if root.exists():
            raise FileExistsError(f"refusing to overwrite existing folder {root}")
        slug = root.name
        now = datetime.now(timezone.utc)
        project = cls(
            name=name,
            slug=slug,
            root=root,
            created_at=now,
        )
        # Folder skeleton.
        (root / "refs" / ".thumbnails").mkdir(parents=True)
        (root / "output" / "kept").mkdir(parents=True)
        (root / "output" / "rejected").mkdir(parents=True)
        (root / "output" / "cache").mkdir(parents=True)
        project.save()
        return project

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = Path(root)
        with open(root / "project.json") as f:
            data = json.load(f)
        llm_raw = data.get("llm") or {}
        training_raw = data.get("training") or {}
        # Build TrainingConfig from disk; tolerate missing keys so older
        # project.json files keep loading after this field was added.
        defaults = TrainingConfig()
        training = TrainingConfig(**{
            f.name: training_raw[f.name]
            for f in fields(defaults)
            if f.name in training_raw
        })
        return cls(
            name=data["name"],
            slug=data["slug"],
            root=root,
            created_at=datetime.fromisoformat(data["created_at"]),
            sources=[Source(**s) for s in data.get("sources", [])],
            refs=[RefImage(**r) for r in data.get("refs", [])],
            thresholds_overrides=data.get("thresholds_overrides", {}),
            source_root=data.get("source_root"),
            pause_before_tag=bool(data.get("pause_before_tag", True)),
            llm=LLMConfig(
                enabled=bool(llm_raw.get("enabled", False)),
                endpoint=str(llm_raw.get("endpoint") or "http://localhost:1234"),
                model=str(llm_raw.get("model") or ""),
                prompt=str(llm_raw.get("prompt") or ""),
                api_key=str(llm_raw.get("api_key") or ""),
            ),
            training=training,
        )

    def save(self) -> None:
        out = {
            "name": self.name,
            "slug": self.slug,
            "created_at": self.created_at.isoformat(),
            "sources": [asdict(s) for s in self.sources],
            "refs": [asdict(r) for r in self.refs],
            "thresholds_overrides": self.thresholds_overrides,
            "source_root": self.source_root,
            "pause_before_tag": self.pause_before_tag,
            "llm": asdict(self.llm),
            "training": asdict(self.training),
        }
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(self.root / "project.json")

    # ---------------- mutations ----------------

    def add_source(self, video_path: Path) -> Source:
        video_path = Path(video_path).resolve()
        if any(Path(s.path) == video_path for s in self.sources):
            raise ValueError(f"video already in project: {video_path}")
        s = Source(
            path=str(video_path),
            added_at=datetime.now(timezone.utc).isoformat(),
        )
        self.sources.append(s)
        self.save()
        return s

    def add_ref(self, ref_path: Path) -> RefImage:
        """Copy an external image into the project's refs/ folder and track it."""
        ref_path = Path(ref_path)
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)
        return self._ingest_ref(ref_path.name, ref_path.read_bytes())

    def add_ref_bytes(self, filename: str, data: bytes) -> RefImage:
        """Save uploaded image bytes into the project's refs/ folder and track it."""
        return self._ingest_ref(filename, data)

    def _ingest_ref(self, filename: str, data: bytes) -> RefImage:
        refs_dir = self.root / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        dest = self._unique_ref_path(filename)
        dest.write_bytes(data)
        r = RefImage(
            path=str(dest.resolve()),
            added_at=datetime.now(timezone.utc).isoformat(),
        )
        self.refs.append(r)
        self.save()
        return r

    def _unique_ref_path(self, filename: str) -> Path:
        """Return a refs/ destination path that doesn't collide with an existing ref."""
        # Sanitize: drop any path components, keep only basename.
        name = Path(filename).name or "ref"
        dest = self.root / "refs" / name
        if not dest.exists():
            return dest
        stem, suffix = dest.stem, dest.suffix
        for n in range(2, 10_000):
            candidate = self.root / "refs" / f"{stem}-{n}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"too many copies of ref named {name!r}")

    def remove_source(self, source_idx: int) -> None:
        del self.sources[source_idx]
        self.save()

    def remove_ref(self, ref_path: str) -> None:
        ref_path = str(Path(ref_path).resolve())
        kept: list[RefImage] = []
        deleted: list[Path] = []
        for r in self.refs:
            if r.path == ref_path:
                deleted.append(Path(r.path))
            else:
                kept.append(r)
        self.refs = kept
        # Also strip from any source's excluded_refs so dangling references don't accumulate.
        for s in self.sources:
            s.excluded_refs = [p for p in s.excluded_refs if p != ref_path]
        self.save()
        # Delete the on-disk file only if it's inside our refs/ folder — never touch
        # external files that may be referenced by older project formats.
        for d in deleted:
            try:
                if d.is_file() and refs_dir_contains(self.root, d):
                    d.unlink()
            except OSError:
                pass

    # ---------------- folder-based source import ----------------

    def import_videos_from_folder(
        self, folder: Path, *, set_root: bool = True
    ) -> tuple[list[Source], list[str]]:
        """Add every video file in ``folder`` as a source.

        Returns ``(added, skipped)`` where ``skipped`` contains the resolved paths
        that were already in the project.
        """
        folder = Path(folder)
        added: list[Source] = []
        skipped: list[str] = []
        for vid in list_videos(folder):
            try:
                added.append(self.add_source(vid))
            except ValueError:
                skipped.append(str(vid.resolve()))
        if set_root:
            self.source_root = str(folder.resolve())
            self.save()
        return added, skipped

    def set_excluded_refs(self, source_idx: int, excluded: list[str]) -> None:
        excluded = [str(Path(p).resolve()) for p in excluded]
        self.sources[source_idx].excluded_refs = excluded
        self.save()

    # ---------------- ref-set + path helpers ----------------

    def effective_refs_for(self, source_idx: int) -> list[str]:
        """All project ref paths minus the per-video opt-outs."""
        excluded = set(self.sources[source_idx].excluded_refs)
        return [r.path for r in self.refs if r.path not in excluded]

    def video_stem(self, source_idx: int) -> str:
        return Path(self.sources[source_idx].path).stem

    @property
    def kept_dir(self) -> Path:
        return self.root / "output" / "kept"

    @property
    def rejected_dir(self) -> Path:
        return self.root / "output" / "rejected"

    @property
    def metadata_path(self) -> Path:
        return self.root / "output" / "metadata.jsonl"

    def cache_dir_for(self, video_stem: str) -> Path:
        return self.root / "output" / "cache" / video_stem

    # ---------------- training ----------------

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def training_runs_dir(self) -> Path:
        return self.training_dir / "runs"

    @property
    def training_state_path(self) -> Path:
        """Where the live runner persists its state across server restarts."""
        return self.training_dir / "state.json"

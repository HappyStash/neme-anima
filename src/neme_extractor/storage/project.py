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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

VIDEO_EXTENSIONS = frozenset({
    ".mkv", ".mp4", ".webm", ".mov", ".avi", ".m4v", ".ts", ".wmv",
})


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

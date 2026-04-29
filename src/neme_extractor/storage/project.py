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
        )

    def save(self) -> None:
        out = {
            "name": self.name,
            "slug": self.slug,
            "created_at": self.created_at.isoformat(),
            "sources": [asdict(s) for s in self.sources],
            "refs": [asdict(r) for r in self.refs],
            "thresholds_overrides": self.thresholds_overrides,
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
        ref_path = Path(ref_path).resolve()
        if any(Path(r.path) == ref_path for r in self.refs):
            raise ValueError(f"ref already in project: {ref_path}")
        r = RefImage(
            path=str(ref_path),
            added_at=datetime.now(timezone.utc).isoformat(),
        )
        self.refs.append(r)
        self.save()
        return r

    def remove_source(self, source_idx: int) -> None:
        del self.sources[source_idx]
        self.save()

    def remove_ref(self, ref_path: str) -> None:
        ref_path = str(Path(ref_path).resolve())
        self.refs = [r for r in self.refs if r.path != ref_path]
        # Also strip from any source's excluded_refs so dangling references don't accumulate.
        for s in self.sources:
            s.excluded_refs = [p for p in s.excluded_refs if p != ref_path]
        self.save()

    def set_excluded_refs(self, source_idx: int, excluded: list[str]) -> None:
        excluded = [str(Path(p).resolve()) for p in excluded]
        self.sources[source_idx].excluded_refs = excluded
        self.save()

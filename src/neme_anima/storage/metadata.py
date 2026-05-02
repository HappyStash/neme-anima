"""Append-only jsonl metadata log for kept/rejected frames.

One record per saved frame. The file lives at <project>/output/metadata.jsonl
and is the single source of truth for per-image traceability across all
extraction runs in a project.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_CHARACTER_SLUG = "default"


@dataclass
class FrameRecord:
    """One row in metadata.jsonl, for traceability of every kept / rejected image.

    ``character_slug`` identifies which character this frame was routed to
    during identification (single-character extractions all use
    ``"default"``). Frames produced before the multi-character migration
    have no slug in their JSON row — :meth:`MetadataLog.iter_records`
    backfills :data:`DEFAULT_CHARACTER_SLUG` so callers always see a slug.
    """
    filename: str
    kept: bool
    scene_idx: int
    tracklet_id: int
    frame_idx: int
    timestamp_seconds: float
    bbox: tuple[int, int, int, int]
    ccip_distance: float
    sharpness: float
    visibility: float
    aspect: float
    score: float
    video_stem: str
    character_slug: str = DEFAULT_CHARACTER_SLUG


class MetadataLog:
    """Wrapper around <project>/output/metadata.jsonl."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: FrameRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def iter_records(
        self, *, video_stem: str | None = None, character_slug: str | None = None,
    ) -> Iterator[FrameRecord]:
        """Stream records, optionally filtered by ``video_stem`` and/or
        ``character_slug``. Older rows that pre-date the multi-character
        migration are backfilled with ``character_slug='default'`` so a
        character-aware caller never has to special-case missing keys.
        """
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["bbox"] = tuple(d["bbox"])
                # Back-compat: pre-migration rows omit character_slug.
                d.setdefault("character_slug", DEFAULT_CHARACTER_SLUG)
                rec = FrameRecord(**d)
                if video_stem is not None and rec.video_stem != video_stem:
                    continue
                if character_slug is not None and rec.character_slug != character_slug:
                    continue
                yield rec

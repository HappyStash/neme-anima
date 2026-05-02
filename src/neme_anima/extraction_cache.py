"""Per-video extraction cache metadata + freshness comparison.

The detection pipeline (scenes → detect → track) is the expensive phase of
``run_extract``: it scales with video length and is dominated by ONNX/YOLO
inference. The output (scenes.parquet + tracklets.parquet) lives in
``<project>/output/cache/<video_stem>/`` and lets ``run_rerun`` skip those
stages and re-evaluate identification + selection + tagging only.

The catch: the cache is only valid as long as the scene / detect / track
thresholds that produced it match the *current* thresholds. If the user
tightens ``track_thresh`` and reruns, they'll get the old tracks. The UI
hides that footgun by disabling Re-process when the cache is stale and
disabling Extract when there's nothing scan-affecting to redo.

This module is the bookkeeping for that:

  * :class:`ExtractionCacheMeta` is the snapshot stamped at extract time.
  * :func:`stamp_meta` writes it to ``cache/<video_stem>/extraction_meta.json``.
  * :func:`cache_state` returns one of "none" / "current" / "stale" given
    the current thresholds — what the UI consumes to decide button states.

Only the scene / detect / track sections are tracked. Identification,
frame selection, cropping, dedup, and tagging are all re-applied by
Re-process, so changes to those don't invalidate the cache.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from neme_anima.config import DetectConfig, SceneConfig, Thresholds, TrackConfig
from neme_anima.storage.project import Project

# Bumped whenever the meta schema changes incompatibly. Old metas without
# this field (or with a different number) read as "stale" so the user is
# guided into a fresh extract that re-stamps the file.
_META_VERSION = 1

CacheState = Literal["none", "current", "stale"]


@dataclass(frozen=True)
class ExtractionCacheMeta:
    """Threshold snapshot taken at the end of a successful Extract run.

    Stamped in ``cache/<video_stem>/extraction_meta.json``. Re-read at UI
    refresh time to compute :func:`cache_state`. ``stamped_at`` is purely
    informational — the freshness comparison is content-based, not time-
    based, so a quick edit-and-revert cycle correctly reads as "current"
    even if minutes pass.
    """
    version: int
    scene: dict
    detect: dict
    track: dict
    stamped_at: str  # ISO-8601 UTC

    @classmethod
    def from_thresholds(cls, t: Thresholds) -> ExtractionCacheMeta:
        return cls(
            version=_META_VERSION,
            scene=asdict(t.scene),
            detect=asdict(t.detect),
            track=asdict(t.track),
            stamped_at=datetime.now(UTC).isoformat(),
        )

    def matches(self, t: Thresholds) -> bool:
        """True iff every scan-affecting threshold matches what's in ``t``.

        Equality is by-field rather than full-dataclass so a future
        non-cache-invalidating field added to one of the dataclasses
        won't be required to round-trip through old meta files. Old
        fields removed from the dataclass will quietly drop here too —
        that's fine; they couldn't have affected the current run.
        """
        if self.version != _META_VERSION:
            return False
        return (
            _section_matches(self.scene, asdict(t.scene), SceneConfig)
            and _section_matches(self.detect, asdict(t.detect), DetectConfig)
            and _section_matches(self.track, asdict(t.track), TrackConfig)
        )


def _section_matches(stamped: dict, current: dict, dc_cls: type) -> bool:
    """Compare ``stamped`` and ``current`` only for fields the dataclass
    currently declares — older snapshot keys outside the schema are
    ignored, missing keys count as mismatches.
    """
    for f in fields(dc_cls()):
        if f.name not in current:
            # Defensive — current Thresholds always carries every field,
            # but a future schema change could drop one. Tolerate.
            continue
        if stamped.get(f.name) != current[f.name]:
            return False
    return True


def _meta_path_for(project: Project, video_stem: str) -> Path:
    return project.cache_dir_for(video_stem) / "extraction_meta.json"


def stamp_meta(project: Project, video_stem: str, thresholds: Thresholds) -> Path:
    """Write the freshness snapshot for ``video_stem``.

    Creates the cache dir if needed (it should already exist after
    write_scenes / write_tracklets, but defensive). Atomic via tmp+rename
    so a partial write never poisons cache_state into reading garbage.
    """
    cache_dir = project.cache_dir_for(video_stem)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = ExtractionCacheMeta.from_thresholds(thresholds)
    target = _meta_path_for(project, video_stem)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def _read_meta(project: Project, video_stem: str) -> ExtractionCacheMeta | None:
    path = _meta_path_for(project, video_stem)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Field-set load so an older meta missing one of the keys reads as
    # "no version" → handled as stale, and a newer meta with extra keys
    # silently drops them.
    return ExtractionCacheMeta(
        version=int(raw.get("version", 0)),
        scene=dict(raw.get("scene") or {}),
        detect=dict(raw.get("detect") or {}),
        track=dict(raw.get("track") or {}),
        stamped_at=str(raw.get("stamped_at") or ""),
    )


def cache_state(
    *, project: Project, video_stem: str, current_thresholds: Thresholds,
) -> CacheState:
    """Three-way state used by the UI to enable/disable Extract vs Re-process.

      * ``"none"``: no usable cache yet. Extract is the only option.
      * ``"current"``: cache exists and the scan-affecting thresholds
        haven't changed since it was stamped — Re-process is sufficient
        and Extract becomes a no-op the UI hides.
      * ``"stale"``: cache exists but at least one scan-affecting
        threshold differs. Re-process would silently use the old
        detections; the UI surfaces a warning and steers the user to
        Extract.

    A ``tracklets.parquet`` without a paired ``extraction_meta.json`` is
    treated as stale — it's almost certainly from a pre-stamping version
    of the pipeline, and we'd rather prompt a fresh extract that produces
    the snapshot than reuse a cache we can't validate.
    """
    cache_dir = project.cache_dir_for(video_stem)
    parquet = cache_dir / "tracklets.parquet"
    if not parquet.is_file():
        return "none"
    meta = _read_meta(project, video_stem)
    if meta is None:
        return "stale"
    return "current" if meta.matches(current_thresholds) else "stale"


def cache_state_for_source(
    project: Project, source_idx: int, current_thresholds: Thresholds,
) -> CacheState:
    """Same as :func:`cache_state` keyed by source index — the shape the
    project view uses when rendering one row per source."""
    if source_idx < 0 or source_idx >= len(project.sources):
        return "none"
    video_stem = project.video_stem(source_idx)
    return cache_state(
        project=project,
        video_stem=video_stem,
        current_thresholds=current_thresholds,
    )


# Public for tests that want to inspect the snapshot without re-reading
# the file.
__all__ = [
    "CacheState",
    "ExtractionCacheMeta",
    "cache_state",
    "cache_state_for_source",
    "stamp_meta",
]

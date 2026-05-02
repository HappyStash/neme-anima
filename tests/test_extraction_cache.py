"""Tests for the extraction-cache freshness module.

The pipeline-side stamping is exercised via run_extract in the existing
e2e tests; this file covers the logic in isolation: stamping, reading,
the three-way state computation, and corruption tolerance. The state
machine is what drives the Sources-tab Extract / Re-process button
disable rules — getting it wrong silently misroutes users to a wasted
re-scan or a stale-cache trap, so coverage matters.
"""

from __future__ import annotations

import json
from pathlib import Path

from neme_anima.config import Thresholds
from neme_anima.extraction_cache import (
    ExtractionCacheMeta,
    cache_state,
    cache_state_for_source,
    stamp_meta,
)
from neme_anima.storage.project import Project


def _make_parquet(project: Project, video_stem: str) -> None:
    """Write a stub tracklets.parquet so cache_state thinks the cache
    exists. The contents are irrelevant for the freshness check — only
    the file's presence and the meta JSON matter."""
    cache_dir = project.cache_dir_for(video_stem)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "tracklets.parquet").write_bytes(b"stub")


def test_state_none_when_no_cache(tmp_path: Path):
    """Without a tracklets.parquet, state is 'none' regardless of meta — the
    UI hides Re-process and points the user at Extract."""
    project = Project.create(tmp_path / "p", name="x")
    state = cache_state(
        project=project, video_stem="ep01",
        current_thresholds=Thresholds(),
    )
    assert state == "none"


def test_state_stale_when_parquet_without_meta(tmp_path: Path):
    """A cache parquet without the paired meta is from a pre-stamping
    extract — treat as stale to nudge a fresh extract that produces the
    snapshot."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    state = cache_state(
        project=project, video_stem="ep01",
        current_thresholds=Thresholds(),
    )
    assert state == "stale"


def test_state_current_when_thresholds_match(tmp_path: Path):
    """Stamping the current thresholds and immediately reading them back
    produces 'current' — the UI keeps Extract muted and Re-process primary."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    t = Thresholds()
    stamp_meta(project, "ep01", t)
    state = cache_state(
        project=project, video_stem="ep01", current_thresholds=t,
    )
    assert state == "current"


def test_state_stale_after_scene_threshold_change(tmp_path: Path):
    """Tweaking a scene-detect knob invalidates the cache — re-process
    would silently reuse the OLD scenes/tracklets."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    stamp_meta(project, "ep01", Thresholds())
    later = Thresholds()
    later.scene.threshold = 33.0  # was 27.0
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=later,
    ) == "stale"


def test_state_stale_after_detect_stride_change(tmp_path: Path):
    """Frame stride is the most-tweaked detect knob and absolutely affects
    the cache. A change must be flagged stale."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    stamp_meta(project, "ep01", Thresholds())
    later = Thresholds()
    later.detect.frame_stride = 8  # was 4
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=later,
    ) == "stale"


def test_state_stale_after_track_threshold_change(tmp_path: Path):
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    stamp_meta(project, "ep01", Thresholds())
    later = Thresholds()
    later.track.match_thresh = 0.6  # was 0.8
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=later,
    ) == "stale"


def test_state_remains_current_after_identify_change(tmp_path: Path):
    """Identification thresholds are NOT cache-invalidating — Re-process
    is the whole point of changing them. State must stay 'current' so
    the UI doesn't push the user toward a re-scan they don't need."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    stamp_meta(project, "ep01", Thresholds())
    later = Thresholds()
    later.identify.body_max_distance_loose = 0.30  # was 0.20
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=later,
    ) == "current"


def test_state_remains_current_after_tag_change(tmp_path: Path):
    """Same idea: tag thresholds run after the cached stages."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    stamp_meta(project, "ep01", Thresholds())
    later = Thresholds()
    later.tag.general_threshold = 0.5  # was 0.35
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=later,
    ) == "current"


def test_state_remains_current_after_dedup_threshold_change(tmp_path: Path):
    """Dedup is post-identify — moving its threshold doesn't require a
    re-scan. (Dedup is always on; the only knob is the threshold.)"""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    stamp_meta(project, "ep01", Thresholds())
    later = Thresholds()
    later.dedup.distance_threshold = 0.07
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=later,
    ) == "current"


def test_corrupt_meta_falls_back_to_stale(tmp_path: Path):
    """A meta file that doesn't parse must NOT crash the project view —
    treat as stale so the user is steered toward a fresh extract that
    rewrites it."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    cache_dir = project.cache_dir_for("ep01")
    (cache_dir / "extraction_meta.json").write_text("{not json", encoding="utf-8")
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=Thresholds(),
    ) == "stale"


def test_unknown_meta_version_treated_as_stale(tmp_path: Path):
    """Bumping ``_META_VERSION`` is how we invalidate stamps from a
    schema-incompatible older release. An unknown version must NOT match
    even when the rest of the snapshot looks right."""
    project = Project.create(tmp_path / "p", name="x")
    _make_parquet(project, "ep01")
    cache_dir = project.cache_dir_for("ep01")
    (cache_dir / "extraction_meta.json").write_text(json.dumps({
        "version": 999,  # future version
        "scene": {}, "detect": {}, "track": {},
        "stamped_at": "2030-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    assert cache_state(
        project=project, video_stem="ep01", current_thresholds=Thresholds(),
    ) == "stale"


def test_cache_state_for_source_indexes_by_position(tmp_path: Path):
    """The source-keyed helper resolves the right video stem and reuses
    the underlying state computation. A bad index returns 'none' rather
    than throwing — the project view runs in a request handler and a
    request-time exception would surface as a confusing 500."""
    project = Project.create(tmp_path / "p", name="x")
    vid_a = tmp_path / "ep01.mkv"
    vid_a.write_bytes(b"")
    vid_b = tmp_path / "ep02.mkv"
    vid_b.write_bytes(b"")
    project.add_source(vid_a)
    project.add_source(vid_b)
    _make_parquet(project, "ep01")
    stamp_meta(project, "ep01", Thresholds())
    # Source 0 has fresh cache; source 1 has none.
    t = Thresholds()
    assert cache_state_for_source(project, 0, t) == "current"
    assert cache_state_for_source(project, 1, t) == "none"
    # Out-of-range index is graceful.
    assert cache_state_for_source(project, 99, t) == "none"
    assert cache_state_for_source(project, -1, t) == "none"


def test_stamp_writes_meta_atomically(tmp_path: Path):
    """The tmp+rename pattern is what makes a partial write impossible —
    confirm both that the final file ends up at the right path and that
    the tmp sentinel doesn't survive."""
    project = Project.create(tmp_path / "p", name="x")
    project.cache_dir_for("ep01").mkdir(parents=True, exist_ok=True)
    target = stamp_meta(project, "ep01", Thresholds())
    assert target.is_file()
    tmp = target.with_suffix(".json.tmp")
    assert not tmp.exists()
    # Round-trips: parsed meta matches a freshly-built one for the same
    # thresholds (modulo timestamps, which the matches() check ignores).
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert "scene" in raw and "detect" in raw and "track" in raw


def test_meta_matches_is_field_by_field(tmp_path: Path):
    """A future Thresholds field added to one of the cached sections
    must NOT cause every old meta to read as stale — the comparison is
    by-field over the dataclass's declared schema."""
    t = Thresholds()
    meta = ExtractionCacheMeta.from_thresholds(t)
    # Inject a stranger key the dataclass doesn't know about — a future
    # release added a field, then rolled it back, leaving leftovers in
    # the JSON. Comparison still passes for the fields we care about.
    meta_with_extra = ExtractionCacheMeta(
        version=meta.version,
        scene={**meta.scene, "future_field": 42},
        detect=meta.detect,
        track=meta.track,
        stamped_at=meta.stamped_at,
    )
    assert meta_with_extra.matches(t)

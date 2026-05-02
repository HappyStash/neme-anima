"""Cross-tracklet perceptual deduplication of kept crops.

The in-tracklet ``frame_select`` already enforces a minimum frame gap so picks
inside a single tracklet aren't on top of each other. Near-duplicates that
leak past that pass come from elsewhere: OP/ED scenes repeated across
episodes, near-identical poses across separate shots in the same scene, two
tracklets that happen to land on the same pose. This module catches that
class of duplicate by embedding every kept crop with CCIP — already loaded
for identification, no new model — and moving the lower-scoring members of
each near-duplicate group to ``rejected/`` so the user can recover them if
the threshold was wrong.

Off by default. Default distance threshold (0.05) is well below the
identification floor (0.15 strict / 0.20 loose), so it only collapses crops
that are essentially the same image — not different poses of the same
character.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neme_anima.config import DedupConfig
from neme_anima.pipeline_progress import NULL_PROGRESS, PipelineProgress
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import Project


@dataclass(frozen=True)
class DedupReport:
    inspected: int
    groups_found: int  # connected components with size > 1
    removed: int       # number of crops moved/deleted
    threshold: float


def _kept_pngs_for_video(project: Project, video_stem: str) -> list[Path]:
    """List PNGs in the project's kept_dir whose filenames start with the stem.

    The trailing double-underscore separator is what the rest of the pipeline
    uses to scope by video — keep that contract here so a video named
    ``ep01ext`` never picks up crops from ``ep01``.
    """
    if not project.kept_dir.exists():
        return []
    prefix = f"{video_stem}__"
    return sorted(
        p for p in project.kept_dir.iterdir()
        if p.is_file() and p.suffix == ".png" and p.name.startswith(prefix)
    )


def _scores_by_filename(project: Project, video_stem: str) -> dict[str, float]:
    """Return ``{filename_stem: score}`` for kept frames in this video.

    The metadata log is append-only, so a re-run leaves stale rows. Take the
    *last* row per filename — the freshest. ``filename`` in the record has no
    extension; the keys here match ``Path.stem`` of the PNGs.
    """
    log = MetadataLog(project.metadata_path)
    out: dict[str, float] = {}
    for rec in log.iter_records(video_stem=video_stem):
        if rec.kept:
            out[rec.filename] = rec.score
    return out


def find_duplicate_groups(
    pairwise_distances: np.ndarray, threshold: float
) -> list[list[int]]:
    """Return connected components in the threshold graph.

    Edge between i and j if ``distances[i,j] < threshold``. Each returned
    group is the list of indices in one component (singletons included). The
    threshold is strict-less-than so an exact identical pair (distance 0)
    always merges; equal-to-threshold does not, which makes 0.0 a safe sentinel.

    Pure function — no I/O — so the caller can unit-test it without loading
    CCIP.
    """
    n = pairwise_distances.shape[0]
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Upper triangle only — distance matrix is symmetric.
    iu, ju = np.triu_indices(n, k=1)
    edges = pairwise_distances[iu, ju] < threshold
    for i, j, e in zip(iu.tolist(), ju.tolist(), edges.tolist(), strict=True):
        if e:
            union(i, j)

    by_root: dict[int, list[int]] = {}
    for idx in range(n):
        by_root.setdefault(find(idx), []).append(idx)
    return list(by_root.values())


def select_keepers(
    groups: list[list[int]], scores: list[float]
) -> tuple[set[int], set[int]]:
    """For each multi-member group, keep the highest-scoring index, drop the rest.

    Singleton groups always keep. Returns ``(keep_indices, drop_indices)``.
    Score ties resolve to the lowest index — deterministic and easy to test.
    """
    keep: set[int] = set()
    drop: set[int] = set()
    for group in groups:
        if len(group) == 1:
            keep.add(group[0])
            continue
        winner = max(group, key=lambda i: (scores[i], -i))
        keep.add(winner)
        for i in group:
            if i != winner:
                drop.add(i)
    return keep, drop


def _move_or_delete(
    png: Path, project: Project, *, move_to_rejected: bool
) -> None:
    """Move ``png`` (and its sidecar ``.txt`` if present) to rejected/, or delete.

    Frame-derivative files (``<stem>_crop.png``, ``<stem>.crop.json``) live
    next to the original under ``kept_dir``; tear them down too, otherwise
    re-crop derivatives outlive their parent in the dataset and the trainer
    picks up a crop with no caption.
    """
    txt = png.with_suffix(".txt")
    crop_png = png.with_name(f"{png.stem}_crop.png")
    crop_json = png.with_suffix(".crop.json")

    if move_to_rejected:
        rej = project.rejected_dir
        rej.mkdir(parents=True, exist_ok=True)
        png.replace(rej / png.name)
        if txt.exists():
            txt.replace(rej / txt.name)
        # Crop derivatives are pure outputs of the kept frame and have no
        # value once the parent is rejected — drop them rather than ferrying
        # them across, since the rejected/ folder is for review, not training.
        for p in (crop_png, crop_json):
            if p.exists():
                p.unlink()
    else:
        for p in (png, txt, crop_png, crop_json):
            if p.exists():
                p.unlink()


def dedup_kept_for_video(
    *,
    project: Project,
    video_stem: str,
    cfg: DedupConfig,
    progress: PipelineProgress | None = None,
) -> DedupReport:
    """Embed kept crops, group near-duplicates, demote the losers.

    Returns the report regardless of whether dedup was enabled — a disabled
    pass is a zero-removed run, which the orchestrator can use to keep the
    progress UI honest. If CCIP isn't installed (CPU-only dev box), the
    function still imports lazily so the rest of the pipeline doesn't break.
    """
    progress = progress or NULL_PROGRESS

    if not cfg.enabled:
        progress.stage_start("dedup", "Dedup", message="disabled")
        progress.stage_done("dedup", message="disabled")
        return DedupReport(inspected=0, groups_found=0, removed=0, threshold=cfg.distance_threshold)

    pngs = _kept_pngs_for_video(project, video_stem)
    if not pngs:
        progress.stage_start("dedup", "Dedup", message="no kept frames")
        progress.stage_done("dedup", message="0 kept frames")
        return DedupReport(inspected=0, groups_found=0, removed=0, threshold=cfg.distance_threshold)

    progress.stage_start(
        "dedup", "Dedup",
        total=len(pngs),
        message=f"embedding {len(pngs)} crops",
    )

    # Lazy-import CCIP so test environments without the GPU group still load
    # the rest of the package.
    from imgutils.metrics import ccip_batch_differences, ccip_batch_extract_features
    from PIL import Image

    pil_imgs = [Image.open(p).convert("RGB") for p in pngs]
    features = ccip_batch_extract_features(pil_imgs)
    progress.stage_advance("dedup", n=len(pngs))
    progress.stage_message("dedup", "computing pairwise distances")

    distances = np.asarray(ccip_batch_differences(features), dtype=np.float64)

    score_map = _scores_by_filename(project, video_stem)
    scores = [score_map.get(p.stem, 0.0) for p in pngs]

    groups = find_duplicate_groups(distances, threshold=cfg.distance_threshold)
    multi_groups = [g for g in groups if len(g) > 1]
    _, drop_indices = select_keepers(groups, scores)

    for idx in sorted(drop_indices):
        _move_or_delete(pngs[idx], project, move_to_rejected=cfg.move_to_rejected)

    if drop_indices:
        _append_dedup_metadata(project, video_stem, pngs, drop_indices)

    progress.stage_done(
        "dedup",
        message=(
            f"{len(drop_indices)} duplicate{'s' if len(drop_indices) != 1 else ''} "
            f"in {len(multi_groups)} group{'s' if len(multi_groups) != 1 else ''}"
        ),
    )
    return DedupReport(
        inspected=len(pngs),
        groups_found=len(multi_groups),
        removed=len(drop_indices),
        threshold=cfg.distance_threshold,
    )


def _append_dedup_metadata(
    project: Project, video_stem: str, pngs: list[Path], drop_indices: set[int]
) -> None:
    """Append a ``kept=False`` record per demoted frame so the metadata log
    reflects the dedup pass.

    The log is append-only and ``list_frames`` (server/api/frames.py) uses
    last-write-wins per filename, so appending a fresh record with
    ``kept=False`` flips the frame's view in the UI from kept to rejected
    without any retroactive editing.
    """
    log = MetadataLog(project.metadata_path)
    score_map = _scores_by_filename(project, video_stem)
    # Rebuild the most-recent metadata snapshot for every dropped frame so we
    # preserve bbox/timestamps without re-deriving them. Walk the log once.
    last_by_name: dict[str, FrameRecord] = {}
    for rec in log.iter_records(video_stem=video_stem):
        last_by_name[rec.filename] = rec
    for idx in drop_indices:
        png = pngs[idx]
        prev = last_by_name.get(png.stem)
        if prev is None:
            # No prior metadata to preserve — synthesize a minimal record.
            log.append(FrameRecord(
                filename=png.stem, kept=False, scene_idx=-1, tracklet_id=-1,
                frame_idx=-1, timestamp_seconds=0.0, bbox=(0, 0, 0, 0),
                ccip_distance=0.0, sharpness=0.0, visibility=0.0, aspect=0.0,
                score=score_map.get(png.stem, 0.0), video_stem=video_stem,
            ))
            continue
        log.append(FrameRecord(
            filename=prev.filename, kept=False,
            scene_idx=prev.scene_idx, tracklet_id=prev.tracklet_id,
            frame_idx=prev.frame_idx, timestamp_seconds=prev.timestamp_seconds,
            bbox=prev.bbox, ccip_distance=prev.ccip_distance,
            sharpness=prev.sharpness, visibility=prev.visibility,
            aspect=prev.aspect, score=prev.score, video_stem=prev.video_stem,
        ))

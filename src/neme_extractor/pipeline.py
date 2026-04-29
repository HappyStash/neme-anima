"""End-to-end orchestration for project-centric extraction + rerun."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from neme_extractor.config import Thresholds
from neme_extractor.crop import crop_frame
from neme_extractor.detect import Detector, FrameDetections
from neme_extractor.frame_select import select_frames
from neme_extractor.identify import Identifier, Verdict
from neme_extractor.output import OutputWriter
from neme_extractor.storage.metadata import FrameRecord
from neme_extractor.storage.project import Project
from neme_extractor.tag import Tagger
from neme_extractor.track import Tracklet, track_scene
from neme_extractor.video import Video, detect_scenes

console = Console()


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console, transient=False,
    )


def _resolve_thresholds(project: Project) -> Thresholds:
    """Merge project.thresholds_overrides over the dataclass defaults."""
    base = Thresholds()
    overrides = project.thresholds_overrides or {}
    for section_name, section_overrides in overrides.items():
        section = getattr(base, section_name, None)
        if section is None:
            continue
        for k, v in section_overrides.items():
            if hasattr(section, k):
                setattr(section, k, v)
    return base


def run_extract(*, project: Project, source_idx: int) -> None:
    thresholds = _resolve_thresholds(project)
    source = project.sources[source_idx]
    video_path = Path(source.path)
    video_stem = project.video_stem(source_idx)
    eff_refs = project.effective_refs_for(source_idx)
    if not eff_refs:
        raise ValueError(f"source has no effective references (all opted out): {video_path.name}")

    writer = OutputWriter(project=project, video_stem=video_stem)
    console.rule(f"[bold]neme-extractor[/bold] :: {video_path.name}")
    console.print(f"refs: {len(eff_refs)} effective ({len(project.refs)} project total)")

    vid = Video(video_path)
    console.print(f"video: {vid.num_frames} frames @ {vid.fps:.2f} fps "
                  f"({vid.duration_seconds:.1f} s)")

    scenes = detect_scenes(
        video_path,
        content_threshold=thresholds.scene.threshold,
        min_scene_len_frames=thresholds.scene.min_scene_len_frames,
    )
    console.print(f"scenes: {len(scenes)}")
    writer.write_scenes(scenes)

    identifier = Identifier(ref_paths=[Path(p) for p in eff_refs], cfg=thresholds.identify)
    detector = Detector(
        person_score_min=thresholds.detect.person_score_min,
        face_score_min=thresholds.detect.face_score_min,
    )

    per_scene: dict[int, list[FrameDetections]] = defaultdict(list)
    stride = max(1, thresholds.detect.frame_stride)
    total_frames = sum(len(range(s.start_frame, s.end_frame, stride)) for s in scenes)

    with _make_progress() as p:
        task = p.add_task("detect", total=total_frames)
        for scene in scenes:
            for fi, frame in vid.iter_frames(
                start=scene.start_frame, end=scene.end_frame, stride=stride
            ):
                fd = detector.detect_frame(fi, frame, with_faces=thresholds.detect.detect_faces)
                per_scene[scene.index].append(fd)
                p.advance(task)

    tracklets: list[Tracklet] = []
    track_cfg = thresholds.track
    track_cfg = type(track_cfg)(
        track_thresh=track_cfg.track_thresh, match_thresh=track_cfg.match_thresh,
        frame_rate=int(round(vid.fps)) or 30,
        track_buffer=track_cfg.track_buffer, min_tracklet_len=track_cfg.min_tracklet_len,
    )
    for scene in scenes:
        scene_dets = per_scene.get(scene.index, [])
        if scene_dets:
            tracklets.extend(track_scene(scene.index, scene_dets, track_cfg))
    console.print(f"tracklets: {len(tracklets)}")
    writer.write_tracklets(tracklets)

    tagger = Tagger(thresholds.tag)
    ref_features = identifier.reference_features()

    with _make_progress() as p:
        task = p.add_task("identify+save", total=len(tracklets))
        kept, rejected = 0, 0
        for tracklet in tracklets:
            score = identifier.score_tracklet(tracklet, vid)
            if score.verdict == Verdict.REJECT:
                _save_one_rejected_sample(writer, vid, tracklet, score.median_distance,
                                          thresholds, video_stem)
                rejected += 1
                p.advance(task)
                continue
            picks = select_frames(tracklet, vid, ref_features, thresholds.frame_select)
            for pick in picks:
                frame = vid.get(pick.frame_idx)
                cropped = crop_frame(frame, pick.detection_bbox, thresholds.crop, compute_mask=False)
                tag_res = tagger.tag(cropped.image_rgb)
                rec = FrameRecord(
                    filename=OutputWriter.filename_for(
                        video_stem=video_stem, scene_idx=pick.scene_idx,
                        tracklet_id=pick.tracklet_id, frame_idx=pick.frame_idx,
                    ),
                    kept=True,
                    scene_idx=pick.scene_idx, tracklet_id=pick.tracklet_id,
                    frame_idx=pick.frame_idx,
                    timestamp_seconds=pick.frame_idx / vid.fps if vid.fps else 0.0,
                    bbox=pick.detection_bbox,
                    ccip_distance=pick.ccip_distance,
                    sharpness=pick.sharpness, visibility=pick.visibility, aspect=pick.aspect,
                    score=pick.score, video_stem=video_stem,
                )
                writer.write_kept(rec, cropped.image_rgb, tag_res.text)
                kept += 1
            p.advance(task)

    console.rule(f"[bold green]done[/bold green]")
    console.print(f"kept: {kept}  rejected: {rejected}  output: {project.kept_dir}")


def _save_one_rejected_sample(
    writer: OutputWriter, vid: Video, tracklet: Tracklet,
    distance: float, thresholds: Thresholds, video_stem: str,
) -> None:
    mid = tracklet.items[len(tracklet.items) // 2]
    bbox = (mid.detection.x1, mid.detection.y1, mid.detection.x2, mid.detection.y2)
    frame = vid.get(mid.frame_idx)
    cropped = crop_frame(frame, bbox, thresholds.crop, compute_mask=False)
    rec = FrameRecord(
        filename=OutputWriter.filename_for(
            video_stem=video_stem, scene_idx=tracklet.scene_idx,
            tracklet_id=tracklet.tracklet_id, frame_idx=mid.frame_idx,
        ),
        kept=False,
        scene_idx=tracklet.scene_idx, tracklet_id=tracklet.tracklet_id,
        frame_idx=mid.frame_idx,
        timestamp_seconds=mid.frame_idx / vid.fps if vid.fps else 0.0,
        bbox=bbox, ccip_distance=distance,
        sharpness=0.0, visibility=0.0, aspect=0.0, score=0.0,
        video_stem=video_stem,
    )
    writer.write_rejected(rec, cropped.image_rgb)


def run_rerun(*, project: Project, video_stem: str) -> None:
    thresholds = _resolve_thresholds(project)
    # Find the source matching this video_stem.
    source_idx = next(
        (i for i, s in enumerate(project.sources) if Path(s.path).stem == video_stem),
        None,
    )
    if source_idx is None:
        raise ValueError(f"no source matches video_stem={video_stem!r}")
    eff_refs = project.effective_refs_for(source_idx)
    if not eff_refs:
        raise ValueError("source has no effective references (all opted out)")

    writer = OutputWriter(project=project, video_stem=video_stem)
    tracklets = writer.read_tracklets()
    console.print(f"cached tracklets: {len(tracklets)}")

    vid = Video(Path(project.sources[source_idx].path))
    identifier = Identifier(ref_paths=[Path(p) for p in eff_refs], cfg=thresholds.identify)
    tagger = Tagger(thresholds.tag)
    ref_features = identifier.reference_features()

    # Wipe previous outputs for THIS video only (filename prefix scopes the delete).
    prefix = f"{video_stem}__"
    for d in (project.kept_dir, project.rejected_dir):
        for f in d.iterdir():
            if f.name.startswith(prefix):
                f.unlink()

    with _make_progress() as p:
        task = p.add_task("rerun", total=len(tracklets))
        for tracklet in tracklets:
            score = identifier.score_tracklet(tracklet, vid)
            if score.verdict == Verdict.REJECT:
                _save_one_rejected_sample(writer, vid, tracklet, score.median_distance,
                                          thresholds, video_stem)
                p.advance(task)
                continue
            picks = select_frames(tracklet, vid, ref_features, thresholds.frame_select)
            for pick in picks:
                frame = vid.get(pick.frame_idx)
                cropped = crop_frame(frame, pick.detection_bbox, thresholds.crop, compute_mask=False)
                tag_res = tagger.tag(cropped.image_rgb)
                rec = FrameRecord(
                    filename=OutputWriter.filename_for(
                        video_stem=video_stem, scene_idx=pick.scene_idx,
                        tracklet_id=pick.tracklet_id, frame_idx=pick.frame_idx,
                    ),
                    kept=True,
                    scene_idx=pick.scene_idx, tracklet_id=pick.tracklet_id,
                    frame_idx=pick.frame_idx,
                    timestamp_seconds=pick.frame_idx / vid.fps if vid.fps else 0.0,
                    bbox=pick.detection_bbox,
                    ccip_distance=pick.ccip_distance,
                    sharpness=pick.sharpness, visibility=pick.visibility, aspect=pick.aspect,
                    score=pick.score, video_stem=video_stem,
                )
                writer.write_kept(rec, cropped.image_rgb, tag_res.text)
            p.advance(task)
    console.rule(f"[bold green]rerun done[/bold green]")

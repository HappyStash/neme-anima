"""End-to-end orchestration for the ``extract`` and ``rerun`` commands."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from neme_extractor.config import Thresholds
from neme_extractor.crop import crop_frame
from neme_extractor.detect import Detector, FrameDetections
from neme_extractor.frame_select import select_frames
from neme_extractor.identify import Identifier, Verdict
from neme_extractor.output import FrameRecord, OutputWriter
from neme_extractor.tag import Tagger
from neme_extractor.track import Tracklet, track_scene
from neme_extractor.video import Video, detect_scenes

console = Console()


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def run_extract(
    video: Path,
    refs_dir: Path,
    out_root: Path,
    config_path: Path | None = None,
) -> None:
    thresholds = Thresholds.from_json(config_path) if config_path else Thresholds()

    writer = OutputWriter(root=out_root, video_stem=video.stem)
    writer.write_thresholds(thresholds)

    console.rule(f"[bold]neme-extractor[/bold] :: {video.name}")

    # 1) Video + scenes
    vid = Video(video)
    console.print(f"video: {vid.num_frames} frames @ {vid.fps:.2f} fps "
                  f"({vid.duration_seconds:.1f} s)")
    scenes = detect_scenes(
        video,
        content_threshold=thresholds.scene.threshold,
        min_scene_len_frames=thresholds.scene.min_scene_len_frames,
    )
    console.print(f"scenes: {len(scenes)}")
    writer.write_scenes(scenes)

    # 2) Identifier (loads refs into CCIP feature cache)
    identifier = Identifier(refs_dir, thresholds.identify)
    console.print(f"references: {identifier.num_references} loaded from {refs_dir}")
    writer.write_run_header(
        video_path=video,
        fps=vid.fps,
        num_frames=vid.num_frames,
        refs=list(identifier.reference_paths()),
    )

    # 3) Detection per scene + per frame (stride applies)
    detector = Detector(
        person_score_min=thresholds.detect.person_score_min,
        face_score_min=thresholds.detect.face_score_min,
    )

    per_scene_detections: dict[int, list[FrameDetections]] = defaultdict(list)
    stride = max(1, thresholds.detect.frame_stride)

    total_frames_to_process = sum(
        len(range(s.start_frame, s.end_frame, stride)) for s in scenes
    )

    with _make_progress() as p:
        task = p.add_task("detect", total=total_frames_to_process)
        for scene in scenes:
            for fi, frame in vid.iter_frames(
                start=scene.start_frame, end=scene.end_frame, stride=stride
            ):
                fd = detector.detect_frame(
                    fi, frame, with_faces=thresholds.detect.detect_faces
                )
                per_scene_detections[scene.index].append(fd)
                p.advance(task)

    # 4) Tracking per scene
    all_tracklets: list[Tracklet] = []
    track_cfg = thresholds.track
    track_cfg = type(track_cfg)(
        track_thresh=track_cfg.track_thresh,
        match_thresh=track_cfg.match_thresh,
        frame_rate=int(round(vid.fps)) or 30,
        track_buffer=track_cfg.track_buffer,
        min_tracklet_len=track_cfg.min_tracklet_len,
    )
    for scene in scenes:
        scene_dets = per_scene_detections.get(scene.index, [])
        if not scene_dets:
            continue
        all_tracklets.extend(track_scene(scene.index, scene_dets, track_cfg))
    console.print(f"tracklets: {len(all_tracklets)}")
    writer.write_tracklets(all_tracklets)

    # 5) Identification + 6) Frame selection + 7) Crop + tag + save
    tagger = Tagger(thresholds.tag)
    ref_features = identifier.reference_features()

    with _make_progress() as p:
        task = p.add_task("identify+frame_select+save", total=len(all_tracklets))
        kept_count, rejected_count = 0, 0
        for tracklet in all_tracklets:
            score = identifier.score_tracklet(tracklet, vid)

            if score.verdict == Verdict.REJECT:
                # Save 1 representative crop to rejected/ for inspection.
                _save_one_rejected_sample(
                    writer, vid, tracklet, score.median_distance, thresholds
                )
                rejected_count += 1
                p.advance(task)
                continue

            picks = select_frames(tracklet, vid, ref_features, thresholds.frame_select)
            for pick in picks:
                frame = vid.get(pick.frame_idx)
                cropped = crop_frame(
                    frame,
                    detection_bbox=pick.detection_bbox,
                    cfg=thresholds.crop,
                    compute_mask=False,  # mask not used for output; saves time
                )
                tag_res = tagger.tag(cropped.image_rgb)
                rec = FrameRecord(
                    filename=OutputWriter.filename_for(
                        pick.scene_idx, pick.tracklet_id, pick.frame_idx
                    ),
                    kept=True,
                    scene_idx=pick.scene_idx,
                    tracklet_id=pick.tracklet_id,
                    frame_idx=pick.frame_idx,
                    timestamp_seconds=pick.frame_idx / vid.fps if vid.fps else 0.0,
                    bbox=pick.detection_bbox,
                    ccip_distance=pick.ccip_distance,
                    sharpness=pick.sharpness,
                    visibility=pick.visibility,
                    aspect=pick.aspect,
                    score=pick.score,
                )
                writer.write_kept(rec, cropped.image_rgb, tag_res.text)
                kept_count += 1
            p.advance(task)

    writer.flush_metadata()
    console.rule(f"[bold green]done[/bold green]")
    console.print(f"kept:     {kept_count}")
    console.print(f"rejected: {rejected_count}")
    console.print(f"output:   {writer.dir}")


def _save_one_rejected_sample(
    writer: OutputWriter,
    vid: Video,
    tracklet: Tracklet,
    distance: float,
    thresholds: Thresholds,
) -> None:
    mid = tracklet.items[len(tracklet.items) // 2]
    frame = vid.get(mid.frame_idx)
    cropped = crop_frame(
        frame,
        detection_bbox=(mid.detection.x1, mid.detection.y1,
                        mid.detection.x2, mid.detection.y2),
        cfg=thresholds.crop,
        compute_mask=False,
    )
    rec = FrameRecord(
        filename=OutputWriter.filename_for(
            tracklet.scene_idx, tracklet.tracklet_id, mid.frame_idx
        ),
        kept=False,
        scene_idx=tracklet.scene_idx,
        tracklet_id=tracklet.tracklet_id,
        frame_idx=mid.frame_idx,
        timestamp_seconds=mid.frame_idx / vid.fps if vid.fps else 0.0,
        bbox=(mid.detection.x1, mid.detection.y1, mid.detection.x2, mid.detection.y2),
        ccip_distance=distance,
        sharpness=0.0, visibility=0.0, aspect=0.0, score=0.0,
    )
    writer.write_rejected(rec, cropped.image_rgb)


def run_rerun(out_dir: Path) -> None:
    """Re-run selection / framing / tagging with edited thresholds, reading the cache."""
    writer = OutputWriter(root=out_dir.parent, video_stem=out_dir.name)
    thresholds = writer.read_thresholds()
    header = writer.read_run_header()
    video_path = Path(header["video_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Cached video path not found: {video_path}")

    refs_dir = Path(header["refs"][0]).parent if header["refs"] else None
    if refs_dir is None or not refs_dir.exists():
        raise FileNotFoundError(
            "Cached refs directory missing — cannot rerun without references."
        )

    console.rule(f"[bold]neme-extractor[/bold] :: rerun {video_path.name}")
    vid = Video(video_path)
    tracklets = writer.read_tracklets()
    console.print(f"cached tracklets: {len(tracklets)}")

    identifier = Identifier(refs_dir, thresholds.identify)
    tagger = Tagger(thresholds.tag)
    ref_features = list(identifier._ref_features)

    # Wipe previous kept/rejected so this rerun produces a clean output set.
    for d in (writer.kept_dir, writer.rejected_dir):
        for f in d.iterdir():
            f.unlink()

    writer._records.clear()

    with _make_progress() as p:
        task = p.add_task("rerun", total=len(tracklets))
        kept, rejected = 0, 0
        for tracklet in tracklets:
            score = identifier.score_tracklet(tracklet, vid)
            if score.verdict == Verdict.REJECT:
                _save_one_rejected_sample(writer, vid, tracklet,
                                          score.median_distance, thresholds)
                rejected += 1
                p.advance(task)
                continue
            picks = select_frames(tracklet, vid, ref_features, thresholds.frame_select)
            for pick in picks:
                frame = vid.get(pick.frame_idx)
                cropped = crop_frame(
                    frame,
                    detection_bbox=pick.detection_bbox,
                    cfg=thresholds.crop,
                    compute_mask=False,
                )
                tag_res = tagger.tag(cropped.image_rgb)
                rec = FrameRecord(
                    filename=OutputWriter.filename_for(
                        pick.scene_idx, pick.tracklet_id, pick.frame_idx
                    ),
                    kept=True,
                    scene_idx=pick.scene_idx,
                    tracklet_id=pick.tracklet_id,
                    frame_idx=pick.frame_idx,
                    timestamp_seconds=pick.frame_idx / vid.fps if vid.fps else 0.0,
                    bbox=pick.detection_bbox,
                    ccip_distance=pick.ccip_distance,
                    sharpness=pick.sharpness,
                    visibility=pick.visibility,
                    aspect=pick.aspect,
                    score=pick.score,
                )
                writer.write_kept(rec, cropped.image_rgb, tag_res.text)
                kept += 1
            p.advance(task)

    writer.flush_metadata()
    console.rule(f"[bold green]rerun done[/bold green]")
    console.print(f"kept:     {kept}")
    console.print(f"rejected: {rejected}")

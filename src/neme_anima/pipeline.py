"""End-to-end orchestration for project-centric extraction + rerun."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from neme_anima.config import Thresholds
from neme_anima.crop import crop_frame
from neme_anima.dedup import dedup_kept_for_video
from neme_anima.detect import Detector, FrameDetections
from neme_anima.frame_select import select_frames
from neme_anima.identify import MultiCharacterRouter
from neme_anima.output import OutputWriter
from neme_anima.pipeline_progress import NULL_PROGRESS, PipelineProgress
from neme_anima.storage.metadata import FrameRecord
from neme_anima.storage.project import Project
from neme_anima.tag import Tagger, join_sidecar, split_sidecar
from neme_anima.track import Tracklet, track_scene
from neme_anima.video import Video, detect_scenes

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


def run_extract(
    *, project: Project, source_idx: int,
    progress: PipelineProgress | None = None,
) -> None:
    progress = progress or NULL_PROGRESS
    try:
        _run_extract_inner(project=project, source_idx=source_idx, progress=progress)
    except Exception as exc:
        progress.stage_fail("setup", f"{type(exc).__name__}: {exc}")
        raise


def _run_extract_inner(
    *, project: Project, source_idx: int, progress: PipelineProgress
) -> None:
    progress.stage_start("setup", "Setup", message="Loading video and references")
    thresholds = _resolve_thresholds(project)
    source = project.sources[source_idx]
    video_path = Path(source.path)
    video_stem = project.video_stem(source_idx)
    refs_by_slug = _refs_by_character(project, source_idx)
    if not any(refs_by_slug.values()):
        raise ValueError(
            f"no character has effective references for {video_path.name}: "
            "every character is either empty or fully opted-out"
        )

    writer = OutputWriter(project=project, video_stem=video_stem)
    console.rule(f"[bold]neme-anima[/bold] :: {video_path.name}")
    total_refs = sum(len(v) for v in refs_by_slug.values())
    active_chars = sum(1 for v in refs_by_slug.values() if v)
    console.print(
        f"refs: {total_refs} effective across {active_chars} "
        f"character{'s' if active_chars != 1 else ''}"
    )

    vid = Video(video_path)
    console.print(f"video: {vid.num_frames} frames @ {vid.fps:.2f} fps "
                  f"({vid.duration_seconds:.1f} s)")
    progress.stage_done(
        "setup",
        message=(
            f"{vid.num_frames:,} frames @ {vid.fps:.1f} fps · "
            f"{total_refs} ref{'s' if total_refs != 1 else ''} "
            f"× {active_chars} char{'s' if active_chars != 1 else ''}"
        ),
    )

    progress.stage_start("scenes", "Scene detection", message="Analysing shots")
    scenes = detect_scenes(
        video_path,
        content_threshold=thresholds.scene.threshold,
        min_scene_len_frames=thresholds.scene.min_scene_len_frames,
    )
    console.print(f"scenes: {len(scenes)}")
    writer.write_scenes(scenes)
    progress.stage_done("scenes", message=f"{len(scenes)} scene{'s' if len(scenes)!=1 else ''}")

    router = MultiCharacterRouter(refs_by_slug=refs_by_slug, cfg=thresholds.identify)
    detector = Detector(
        person_score_min=thresholds.detect.person_score_min,
        face_score_min=thresholds.detect.face_score_min,
    )

    per_scene: dict[int, list[FrameDetections]] = defaultdict(list)
    stride = max(1, thresholds.detect.frame_stride)
    total_frames = sum(len(range(s.start_frame, s.end_frame, stride)) for s in scenes)

    progress.stage_start(
        "detect", "Person detection",
        total=total_frames,
        message=f"0 / {total_frames:,} frames",
    )
    with _make_progress() as p:
        task = p.add_task("detect", total=total_frames)
        seen = 0
        for scene in scenes:
            for fi, frame in vid.iter_frames(
                start=scene.start_frame, end=scene.end_frame, stride=stride
            ):
                fd = detector.detect_frame(fi, frame, with_faces=thresholds.detect.detect_faces)
                per_scene[scene.index].append(fd)
                p.advance(task)
                seen += 1
                progress.stage_advance("detect")
    progress.stage_done("detect", message=f"{total_frames:,} frames scanned")

    progress.stage_start("track", "Tracking", message="Building tracklets")
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
    progress.stage_done("track", message=f"{len(tracklets)} tracklet{'s' if len(tracklets)!=1 else ''}")

    progress.stage_start(
        "identify", "Identify · select · save",
        total=len(tracklets),
        message=f"0 / {len(tracklets)} tracklets",
    )
    with _make_progress() as p:
        task = p.add_task("identify+save", total=len(tracklets))
        kept, rejected = 0, 0
        for tracklet in tracklets:
            routed = router.route_tracklet(tracklet, vid)
            if routed.character_slug is None:
                _save_one_rejected_sample(
                    writer, vid, tracklet, routed.score.median_distance,
                    thresholds, video_stem,
                )
                rejected += 1
                p.advance(task)
                progress.stage_advance("identify")
                progress.stage_message(
                    "identify",
                    f"{kept + rejected} / {len(tracklets)} · kept {kept} · rejected {rejected}",
                )
                continue
            ref_features = router.reference_features(routed.character_slug)
            picks = select_frames(tracklet, vid, ref_features, thresholds.frame_select)
            for pick in picks:
                frame = vid.get(pick.frame_idx)
                cropped = crop_frame(frame, pick.detection_bbox, thresholds.crop, compute_mask=False)
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
                    character_slug=routed.character_slug,
                )
                # Defer tagging — write the image with an empty .txt so the
                # user can review/delete kept frames before paying the
                # tagger cost on them.
                writer.write_kept_image(rec, cropped.image_rgb)
                kept += 1
            p.advance(task)
            progress.stage_advance("identify")
            progress.stage_message(
                "identify",
                f"{kept + rejected} / {len(tracklets)} · kept {kept} · rejected {rejected}",
            )

    progress.stage_done(
        "identify",
        message=f"kept {kept} · rejected {rejected}",
    )

    dedup_report = dedup_kept_for_video(
        project=project, video_stem=video_stem,
        cfg=thresholds.dedup, progress=progress,
    )

    _run_tag_stage(
        project=project, video_stem=video_stem, thresholds=thresholds,
        progress=progress, pause=project.pause_before_tag,
    )

    progress.finish({
        "kept": kept - dedup_report.removed,
        "rejected": rejected + dedup_report.removed,
        "deduped": dedup_report.removed,
    })

    console.rule("[bold green]done[/bold green]")
    console.print(
        f"kept: {kept - dedup_report.removed}  rejected: {rejected + dedup_report.removed}"
        f"  (dedup removed {dedup_report.removed})  output: {project.kept_dir}"
    )


def _run_tag_stage(
    *, project: Project, video_stem: str, thresholds: Thresholds,
    progress: PipelineProgress, pause: bool,
) -> None:
    """Tag every kept frame currently on disk for ``video_stem``.

    Splitting tagging out of the identify loop lets the UI pause here so the
    user can delete unwanted frames before they get tagged. Files the user
    deleted between identify and resume simply aren't picked up by this scan.
    """
    if pause:
        progress.wait_for_resume(
            message="Review kept frames, then resume to tag remaining",
        )

    prefix = f"{video_stem}__"
    if not project.kept_dir.exists():
        progress.stage_start("tag", "Tagging", total=0, message="no kept frames")
        progress.stage_done("tag", message="0 frames")
        return

    pending = sorted(
        p for p in project.kept_dir.iterdir()
        if p.is_file() and p.suffix == ".png" and p.name.startswith(prefix)
    )
    progress.stage_start(
        "tag", "Tagging", total=len(pending),
        message=f"0 / {len(pending)} frames",
    )
    if not pending:
        progress.stage_done("tag", message="0 frames")
        return

    tagger = Tagger(thresholds.tag)
    llm_active = bool(project.llm.enabled and project.llm.model)
    with _make_progress() as p:
        task = p.add_task("tag", total=len(pending))
        tagged = 0
        for png in pending:
            with Image.open(png) as im:
                arr = np.array(im.convert("RGB"))
            tag_res = tagger.tag(arr)
            description = ""
            if llm_active:
                description = _safe_describe(png, project, tag_res.text)
            png.with_suffix(".txt").write_text(
                join_sidecar(tag_res.text, description), encoding="utf-8",
            )
            tagged += 1
            p.advance(task)
            progress.stage_advance("tag")
            progress.stage_message("tag", f"{tagged} / {len(pending)} frames")
    progress.stage_done("tag", message=f"{tagged} frame{'s' if tagged != 1 else ''} tagged")


def _safe_describe(png: Path, project, danbooru_tags: str) -> str:
    """Run the LLM description without taking down the whole pipeline on a
    transient endpoint hiccup — log and skip instead. The user can re-trigger
    LLM tagging from the frames toolbar after fixing the endpoint.
    """
    from neme_anima.llm import DEFAULT_PROMPT, LLMUnavailable, describe_image

    try:
        return describe_image(
            endpoint=project.llm.endpoint,
            model=project.llm.model,
            image_path=png,
            prompt=project.llm.prompt or DEFAULT_PROMPT,
            danbooru_tags=danbooru_tags,
            api_key=project.llm.api_key or None,
        )
    except LLMUnavailable as exc:
        console.print(f"[yellow]llm describe failed for {png.name}: {exc}[/yellow]")
        return ""
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]llm describe error for {png.name}: {exc}[/yellow]")
        return ""


def _save_one_rejected_sample(
    writer: OutputWriter, vid: Video, tracklet: Tracklet,
    distance: float, thresholds: Thresholds, video_stem: str,
) -> None:
    """Write a midpoint sample of a rejected tracklet to ``rejected/``.

    Rejected frames belong to no character — they're surfaced to the user
    purely so they can audit "why was this rejected?". We tag them with
    the default character slug so per-character listings still see them
    when filtering by the default character (which is where the only
    surviving filter, in mono-character projects, lives).
    """
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


def _refs_by_character(project: Project, source_idx: int) -> dict[str, list[Path]]:
    """Build the ``{character_slug: [ref Path, ...]}`` map the router needs.

    Each character's per-source opt-outs are honoured. Characters with zero
    refs (after opt-outs) are still present in the returned map with an
    empty list — the router skips empty lists internally, but keeping them
    makes the diagnostic table in :class:`RoutedTrackletScore.per_character`
    easier to render in the UI.
    """
    out: dict[str, list[Path]] = {}
    for c in project.characters:
        eff = project.effective_refs_for(source_idx, character_slug=c.slug)
        out[c.slug] = [Path(p) for p in eff]
    return out


def _wipe_outputs_for_stem(project: Project, video_stem: str) -> None:
    """Delete only the kept/rejected files belonging to one video, identified by
    the ``<video_stem>__`` filename prefix. The trailing double-underscore
    separator is what makes this safe — a video named ``ep01ext`` produces the
    prefix ``ep01ext__`` which never collides with ``ep01__``.
    """
    prefix = f"{video_stem}__"
    for d in (project.kept_dir, project.rejected_dir):
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.name.startswith(prefix):
                f.unlink()


def run_rerun(
    *, project: Project, video_stem: str,
    progress: PipelineProgress | None = None,
) -> None:
    progress = progress or NULL_PROGRESS
    try:
        _run_rerun_inner(project=project, video_stem=video_stem, progress=progress)
    except Exception as exc:
        progress.stage_fail("setup", f"{type(exc).__name__}: {exc}")
        raise


def _run_rerun_inner(
    *, project: Project, video_stem: str, progress: PipelineProgress
) -> None:
    progress.stage_start("setup", "Setup", message="Loading cached tracklets")
    thresholds = _resolve_thresholds(project)
    # Find the source matching this video_stem.
    source_idx = next(
        (i for i, s in enumerate(project.sources) if Path(s.path).stem == video_stem),
        None,
    )
    if source_idx is None:
        raise ValueError(f"no source matches video_stem={video_stem!r}")
    refs_by_slug = _refs_by_character(project, source_idx)
    if not any(refs_by_slug.values()):
        raise ValueError(
            "no character has effective references — every character is "
            "either empty or fully opted-out for this source"
        )

    writer = OutputWriter(project=project, video_stem=video_stem)
    tracklets = writer.read_tracklets()
    console.print(f"cached tracklets: {len(tracklets)}")

    vid = Video(Path(project.sources[source_idx].path))
    router = MultiCharacterRouter(refs_by_slug=refs_by_slug, cfg=thresholds.identify)

    _wipe_outputs_for_stem(project, video_stem)
    progress.stage_done(
        "setup",
        message=f"{len(tracklets)} cached tracklet{'s' if len(tracklets)!=1 else ''}",
    )

    progress.stage_start(
        "identify", "Identify · select · save",
        total=len(tracklets),
        message=f"0 / {len(tracklets)} tracklets",
    )
    with _make_progress() as p:
        task = p.add_task("rerun", total=len(tracklets))
        kept, rejected = 0, 0
        for tracklet in tracklets:
            routed = router.route_tracklet(tracklet, vid)
            if routed.character_slug is None:
                _save_one_rejected_sample(
                    writer, vid, tracklet, routed.score.median_distance,
                    thresholds, video_stem,
                )
                rejected += 1
                p.advance(task)
                progress.stage_advance("identify")
                progress.stage_message(
                    "identify",
                    f"{kept + rejected} / {len(tracklets)} · kept {kept} · rejected {rejected}",
                )
                continue
            ref_features = router.reference_features(routed.character_slug)
            picks = select_frames(tracklet, vid, ref_features, thresholds.frame_select)
            for pick in picks:
                frame = vid.get(pick.frame_idx)
                cropped = crop_frame(frame, pick.detection_bbox, thresholds.crop, compute_mask=False)
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
                    character_slug=routed.character_slug,
                )
                writer.write_kept_image(rec, cropped.image_rgb)
                kept += 1
            p.advance(task)
            progress.stage_advance("identify")
            progress.stage_message(
                "identify",
                f"{kept + rejected} / {len(tracklets)} · kept {kept} · rejected {rejected}",
            )
    progress.stage_done("identify", message=f"kept {kept} · rejected {rejected}")

    dedup_report = dedup_kept_for_video(
        project=project, video_stem=video_stem,
        cfg=thresholds.dedup, progress=progress,
    )

    _run_tag_stage(
        project=project, video_stem=video_stem, thresholds=thresholds,
        progress=progress, pause=project.pause_before_tag,
    )

    progress.finish({
        "kept": kept - dedup_report.removed,
        "rejected": rejected + dedup_report.removed,
        "deduped": dedup_report.removed,
    })
    console.rule("[bold green]rerun done[/bold green]")

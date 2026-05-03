"""Cross-project character copy.

Recreates a source character inside a destination project as a brand-new
character with the same slug + name, and copies over every artifact
"related to the character only" (refs, source videos that produced kept
frames, those frames + sidecars + crops, plus identity-scoped settings
like ``trigger_token`` / ``core_tags`` / ``multiply``).

Conflict semantics — per-object "drop the imported object":
  * Character slug already in dst → :class:`ValueError`.
  * Source video (same resolved abs path) already in dst → drop the source
    record only; still try to import its frames (each frame collides
    individually).
  * Ref filename already in dst's ``refs/`` → auto-rename via
    ``Project._unique_ref_path``.
  * Frame filename already in dst's ``kept/`` → drop that frame and its
    sidecar / crop / metadata row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import (
    CROP_SUFFIX, Character, Project,
)


@dataclass
class CopyReport:
    character_slug: str
    sources_added: list[str] = field(default_factory=list)
    sources_skipped: list[str] = field(default_factory=list)
    refs_added: list[str] = field(default_factory=list)
    refs_renamed: dict[str, str] = field(default_factory=dict)
    frames_added: list[str] = field(default_factory=list)
    frames_skipped: list[str] = field(default_factory=list)
    custom_uploads_added: int = 0
    crops_copied: int = 0
    metadata_rows_appended: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def copy_character_to_project(
    *,
    src: Project,
    src_character_slug: str,
    dst: Project,
    dry_run: bool = False,
) -> CopyReport:
    """Copy ``src.character_by_slug(src_character_slug)`` into ``dst``.

    Raises ``KeyError`` if the source character doesn't exist; raises
    ``ValueError`` if ``dst`` already has a character with the same slug
    (no partial writes happen before this check).
    """
    src_char = src.character_by_slug(src_character_slug)
    if src_char is None:
        raise KeyError(
            f"unknown character {src_character_slug!r} in source project",
        )
    if dst.character_by_slug(src_char.slug) is not None:
        raise ValueError(
            f"character {src_char.slug!r} already exists in destination — refuse copy",
        )

    report = CopyReport(character_slug=src_char.slug, dry_run=dry_run)

    # 1) Resolve which videos travel with this character: every distinct
    #    video_stem in the metadata log for this character.
    src_log = MetadataLog(src.metadata_path)
    related_stems: set[str] = set()
    for rec in src_log.iter_records(character_slug=src_character_slug):
        related_stems.add(rec.video_stem)
    custom_stems = {s for s in related_stems if s == "custom_uploads"}
    real_stems = related_stems - custom_stems

    # 2) Pre-build collision sets in dst. Resolved abs paths for sources
    #    and a directory listing for kept/.
    dst_source_paths = {
        str(Path(s.path).resolve()) for s in dst.sources
    }
    dst_kept_files = (
        {p.name for p in dst.kept_dir.iterdir()}
        if dst.kept_dir.is_dir() else set()
    )

    # 3) Add the new character record (or simulate in dry-run).
    if not dry_run:
        new_char = dst.add_character(name=src_char.name, slug=src_char.slug)
        new_char.trigger_token = src_char.trigger_token
        new_char.core_tags = list(src_char.core_tags)
        new_char.core_tags_freq_threshold = src_char.core_tags_freq_threshold
        new_char.core_tags_enabled = src_char.core_tags_enabled
        new_char.multiply = src_char.multiply
        dst.save()

    # 4) Resolve src.sources keyed by stem so we can cross-reference each
    #    related stem to a Source record.
    src_source_by_stem: dict[str, object] = {
        Path(s.path).stem: s for s in src.sources
    }

    # 5) Import sources.
    for stem in sorted(real_stems):
        src_source = src_source_by_stem.get(stem)
        if src_source is None:
            # Orphan stem (no Source record in src). We'll still try to
            # import frames; nothing source-side to add.
            continue
        src_path = Path(src_source.path).resolve()
        if str(src_path) in dst_source_paths:
            report.sources_skipped.append(str(src_path))
            continue
        if not dry_run:
            new_src = dst.add_source(src_path)
            # Carry over the source's per-character excluded_refs for this
            # character only.
            excluded = src_source.excluded_refs.get(src_character_slug, [])
            if excluded:
                idx = len(dst.sources) - 1
                dst.set_excluded_refs(
                    idx, list(excluded), character_slug=src_char.slug,
                )
        report.sources_added.append(str(src_path))

    # 6) Import refs.
    for ref in src_char.refs:
        ref_path = Path(ref.path)
        if not ref_path.is_file():
            continue
        original_name = ref_path.name
        if dry_run:
            # Simulate: would the dst's refs/ folder produce a renamed file?
            dst_refs = dst.root / "refs"
            target = dst_refs / original_name
            n = 2
            stem, suffix = target.stem, target.suffix
            while target.exists():
                target = dst_refs / f"{stem}-{n}{suffix}"
                n += 1
            if target.name != original_name:
                report.refs_renamed[original_name] = target.name
            report.refs_added.append(target.name)
        else:
            saved = dst.add_ref_bytes(
                original_name, ref_path.read_bytes(),
                character_slug=src_char.slug,
            )
            saved_name = Path(saved.path).name
            if saved_name != original_name:
                report.refs_renamed[original_name] = saved_name
            report.refs_added.append(saved_name)

    # 7) Import frames + sidecars + crops.
    seen_filenames: set[str] = set()
    for rec in src_log.iter_records(character_slug=src_character_slug):
        # Append-only log → multiple rows per filename can exist after
        # moves. Last-write-wins semantics: keep only the most recent row
        # we see for a given filename. Since iter_records is ordered, we
        # fold by filename and process in a second pass.
        seen_filenames.add(rec.filename)

    # Reload to get last-write-wins per filename.
    last_rec_by_filename: dict[str, FrameRecord] = {}
    for rec in src_log.iter_records():
        if rec.filename in seen_filenames and rec.character_slug == src_character_slug:
            last_rec_by_filename[rec.filename] = rec
        elif rec.filename in seen_filenames and rec.character_slug != src_character_slug:
            # Frame was reassigned to another character; drop it.
            last_rec_by_filename.pop(rec.filename, None)

    dst_log = MetadataLog(dst.metadata_path)
    for filename, rec in last_rec_by_filename.items():
        src_png = src.kept_dir / f"{filename}.png"
        if not src_png.is_file():
            continue
        if f"{filename}.png" in dst_kept_files:
            report.frames_skipped.append(filename)
            continue
        # Copy png + optional sidecar + optional crop derivative + crop spec.
        src_txt = src.kept_dir / f"{filename}.txt"
        src_crop_png = src.kept_dir / f"{filename}{CROP_SUFFIX}.png"
        src_crop_spec = src.kept_dir / f"{filename}.crop.json"
        if not dry_run:
            dst.kept_dir.mkdir(parents=True, exist_ok=True)
            (dst.kept_dir / f"{filename}.png").write_bytes(src_png.read_bytes())
            if src_txt.is_file():
                (dst.kept_dir / f"{filename}.txt").write_text(
                    src_txt.read_text(encoding="utf-8"), encoding="utf-8",
                )
            if src_crop_png.is_file():
                (dst.kept_dir / f"{filename}{CROP_SUFFIX}.png").write_bytes(
                    src_crop_png.read_bytes(),
                )
                report.crops_copied += 1
            if src_crop_spec.is_file():
                (dst.kept_dir / f"{filename}.crop.json").write_text(
                    src_crop_spec.read_text(encoding="utf-8"), encoding="utf-8",
                )
            dst_log.append(FrameRecord(
                filename=rec.filename, kept=True,
                scene_idx=rec.scene_idx, tracklet_id=rec.tracklet_id,
                frame_idx=rec.frame_idx,
                timestamp_seconds=rec.timestamp_seconds, bbox=rec.bbox,
                ccip_distance=rec.ccip_distance, sharpness=rec.sharpness,
                visibility=rec.visibility, aspect=rec.aspect, score=rec.score,
                video_stem=rec.video_stem, character_slug=src_char.slug,
            ))
            report.metadata_rows_appended += 1
        report.frames_added.append(filename)
        if rec.video_stem == "custom_uploads":
            report.custom_uploads_added += 1

    return report

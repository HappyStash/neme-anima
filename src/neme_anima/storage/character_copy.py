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
    return report

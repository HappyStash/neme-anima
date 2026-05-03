"""Per-character core-tag analysis and pruning.

Idea (ported from anime_screenshot_pipeline's ``CoreTagProcessor``): tags
that appear in a high fraction of a character's frames are *implied* by the
character's trigger word and shouldn't be in per-image captions. Pruning
them reduces over-fitting on incidental traits and produces cleaner
captions.

This module is purely analytical and pure-functional where possible — it
reads tag sidecars and project metadata, returns a report, and the rest of
the system decides whether/when to apply pruning. Pruning is applied at
training-dataset-staging time, not by mutating the kept-dir sidecars, so
the user's manual edits and full-fidelity captions stay on disk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from neme_anima.storage.metadata import MetadataLog
from neme_anima.storage.project import Project
from neme_anima.tag import split_sidecar

# Tags that describe pose/setting/composition rather than character identity.
# They commonly clear the per-character frequency bar (most shots have a
# subject in some pose with some background) but pruning them would damage
# training because those traits are exactly what we want the LoRA to learn
# to vary. Mirrors anime2sd's blacklist; keep it tight — adding too much
# silently swallows real character traits.
DEFAULT_BLACKLIST: tuple[str, ...] = (
    "solo",
    "1girl",
    "1boy",
    "looking_at_viewer",
    "simple_background",
    "white_background",
    "smile",
    "open_mouth",
    "closed_mouth",
    "blush",
    "standing",
    "sitting",
    "looking_back",
    "from_side",
    "from_behind",
    "from_above",
    "from_below",
    "indoors",
    "outdoors",
    "day",
    "night",
)


@dataclass(frozen=True)
class CoreTagsReport:
    """Result of scanning one character's frames for core tags.

    ``tags`` is sorted by descending frequency so the UI can render the
    table in priority order. ``corpus_size`` is the number of single-
    character frames considered — surfaced so the user can see whether the
    threshold is meaningful (a 35 % bar over 5 frames is noisier than over
    500). ``blacklisted`` lists tags that *would* have been core-tags but
    were filtered out by the blacklist — useful for diagnostics.
    """
    character_slug: str
    corpus_size: int
    threshold: float
    tags: tuple[tuple[str, float], ...]  # (tag, frequency_in_corpus) — sorted desc
    blacklisted: tuple[str, ...]


def _parse_tags(tag_text: str) -> list[str]:
    """Split a danbooru-line into individual tags.

    The kohya convention is comma-separated tags with single spaces after
    each comma; we strip whitespace defensively and drop empty tokens.
    """
    return [t.strip() for t in tag_text.split(",") if t.strip()]


def _filenames_for_character(
    project: Project, character_slug: str,
) -> list[str]:
    """Return the kept filenames currently routed to ``character_slug``
    AND still present on disk.

    The metadata log is append-only; ``delete_frame`` removes the .png/.txt
    on disk but leaves the historical row. Without the on-disk filter the
    corpus_size denominator would include phantom rows for deleted frames,
    crushing every tag's frequency below the threshold and producing an
    empty suggestions list. Mirrors ``list_frames``' "metadata + on-disk"
    intersection so the user sees the same set the UI shows.

    Last-write-wins per filename so a frame moved to a different character
    via the bulk-move endpoint is counted under its new owner only.
    """
    log = MetadataLog(project.metadata_path)
    by_name: dict[str, str] = {}
    for rec in log.iter_records():
        if not rec.kept:
            continue
        by_name[rec.filename] = rec.character_slug
    return [
        fn for fn, slug in by_name.items()
        if slug == character_slug
        and (project.kept_dir / f"{fn}.png").is_file()
    ]


def compute_core_tags(
    *,
    project: Project,
    character_slug: str,
    threshold: float = 0.35,
    blacklist: tuple[str, ...] = DEFAULT_BLACKLIST,
) -> CoreTagsReport:
    """Scan ``character_slug``'s kept frames and return tags above ``threshold``.

    For each frame, we read the danbooru line from the on-disk sidecar
    (line 1 of ``<filename>.txt``) and count tag occurrences. A tag's
    "frequency" is ``count / corpus_size``; tags with frequency >= threshold
    AND not in blacklist are returned as core tags. Below-threshold tags are
    dropped silently — the report is for what *would* be pruned.
    """
    filenames = _filenames_for_character(project, character_slug)
    corpus_size = len(filenames)
    if corpus_size == 0:
        return CoreTagsReport(
            character_slug=character_slug, corpus_size=0,
            threshold=threshold, tags=(), blacklisted=(),
        )

    counter: Counter[str] = Counter()
    for fn in filenames:
        sidecar = project.kept_dir / f"{fn}.txt"
        if not sidecar.is_file():
            continue
        try:
            text = sidecar.read_text(encoding="utf-8")
        except OSError:
            continue
        danbooru, _ = split_sidecar(text)
        # Tags appear at most once per frame in our normalized sidecars
        # (join_sidecar dedupes), so set() is defensive but cheap.
        for tag in set(_parse_tags(danbooru)):
            counter[tag] += 1

    blacklisted_set = set(blacklist)
    above_threshold = []
    blacklisted_hits = []
    for tag, count in counter.most_common():
        freq = count / corpus_size
        if freq < threshold:
            break  # most_common is sorted desc, so no later tag can pass
        if tag in blacklisted_set:
            blacklisted_hits.append(tag)
            continue
        above_threshold.append((tag, freq))

    return CoreTagsReport(
        character_slug=character_slug,
        corpus_size=corpus_size,
        threshold=threshold,
        tags=tuple(above_threshold),
        blacklisted=tuple(blacklisted_hits),
    )


def prune_tags(tag_text: str, core_tags: list[str] | tuple[str, ...]) -> str:
    """Drop ``core_tags`` from a comma-separated danbooru tag line.

    Order of remaining tags is preserved so the trainer's tag-ordering
    heuristics (kohya conventionally weighs earlier tags more) aren't
    disturbed. Returns the line in the same comma-separated form ready to
    be written back to a sidecar.
    """
    if not core_tags:
        return tag_text
    drop = set(core_tags)
    kept = [t for t in _parse_tags(tag_text) if t not in drop]
    return ", ".join(kept)


def prune_sidecar_text(sidecar_text: str, core_tags: list[str] | tuple[str, ...]) -> str:
    """Apply :func:`prune_tags` to the danbooru line of a two-line sidecar.

    The description line (line 2) is preserved intact — core-tag pruning
    only affects the danbooru tag set. Returns a string identical in shape
    to what :func:`neme_anima.tag.join_sidecar` would produce.
    """
    from neme_anima.tag import join_sidecar
    danbooru, description = split_sidecar(sidecar_text)
    pruned = prune_tags(danbooru, core_tags)
    return join_sidecar(pruned, description)


def core_tags_for_filename(
    project: Project, filename: str,
) -> tuple[str, list[str]] | None:
    """Resolve ``(character_slug, core_tags)`` for a kept frame.

    Reads metadata last-write-wins to find the character that currently
    owns the frame, then looks up that character's persisted core_tags.
    Returns ``None`` if the frame has no metadata row — the caller should
    treat that as "no pruning" rather than failing.
    """
    log = MetadataLog(project.metadata_path)
    owner: str | None = None
    for rec in log.iter_records():
        if rec.filename == filename:
            owner = rec.character_slug
    if owner is None:
        return None
    character = project.character_by_slug(owner)
    if character is None or not character.core_tags_enabled:
        return owner, []
    return owner, list(character.core_tags)


def _read_sidecar_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""

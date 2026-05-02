"""Per-character dataset balancing — inverse-frequency repeat multipliers.

Ported from anime_screenshot_pipeline's ``balancing.py`` and adapted to
neme-extractor's flat-kept-dir, metadata-driven character grouping.

Why this matters: a multi-character project usually has very uneven frame
counts (the protagonist gets 800 keepers; a side character gets 30). Naive
training over-fits the protagonist and under-trains the side character.
The fix is to weight each character's frames by ``1/frequency`` so the
trainer sees roughly equal per-step exposure to each character.

The output is a dict ``{character_slug: multiply}`` where ``multiply`` is
the repeat factor diffusion-pipe should apply to that character's frames
(the ``num_repeats`` field in the per-character ``[[directory]]`` block in
``dataset.toml``). Single-character projects degenerate to ``{slug: 1.0}``;
nothing to balance.

Design notes:
  - "Auto" mode uses inverse frequency clamped to ``[min, max]``. Default
    range is conservative (1–10) so the largest character isn't crushed
    when one tiny character would push multipliers to absurd values.
  - "Manual" mode honours each character's ``Character.multiply`` field
    when it's > 0; auto-computed otherwise. This is the unit the UI
    surfaces — you can let it auto-balance globally and override one
    character's multiplier without leaving the auto path for the others.
  - Multipliers are rounded to one decimal so dataset.toml stays diffable.
    diffusion-pipe accepts floats here; an integer-only trainer would
    require a bigger denominator step which we'd add as a separate config.
"""

from __future__ import annotations

from dataclasses import dataclass

from neme_anima.storage.metadata import MetadataLog
from neme_anima.storage.project import Project


@dataclass(frozen=True)
class CharacterBalanceRow:
    """One row in the balancing-preview table.

    ``frame_count`` is what the metadata log says for this character (kept
    frames only, last-write-wins). ``auto_multiply`` is the value the
    inverse-frequency formula would pick. ``effective_multiply`` is what
    the trainer will actually use — equal to the manual override when one
    is set, else the auto value.
    """
    character_slug: str
    name: str
    frame_count: int
    auto_multiply: float
    manual_multiply: float  # 0.0 = "auto"
    effective_multiply: float


def _frame_counts_per_character(project: Project) -> dict[str, int]:
    """Return ``{character_slug: kept_frame_count}`` derived from the metadata log.

    Last-write-wins per filename so a frame that's been moved to a
    different character via the bulk-move endpoint is counted once, under
    its new owner. Filenames whose latest record is ``kept=False`` (rejected
    or removed) are excluded — they don't contribute to training corpus.
    """
    log = MetadataLog(project.metadata_path)
    latest_owner: dict[str, tuple[bool, str]] = {}
    for rec in log.iter_records():
        latest_owner[rec.filename] = (rec.kept, rec.character_slug)
    counts: dict[str, int] = {}
    for kept, slug in latest_owner.values():
        if not kept:
            continue
        counts[slug] = counts.get(slug, 0) + 1
    # Ensure every project character is represented even with zero frames
    # so the UI table never silently drops a row that the user just
    # created but hasn't extracted into yet.
    for c in project.characters:
        counts.setdefault(c.slug, 0)
    return counts


def _auto_multiplier(
    n: int, total: int, character_count: int,
    *, min_multiply: float, max_multiply: float,
) -> float:
    """Inverse-frequency multiplier for a character with ``n`` frames.

    Mirrors anime2sd's normalization: each character's "share" of training
    steps is proportional to ``multiply * n``; we want every character to
    get the same share, which means ``multiply ∝ 1/n``. Anchored so the
    smallest character gets ``min_multiply`` (typically 1.0) and clamped
    upward at ``max_multiply`` to avoid blowups when one character has
    only a handful of frames.

    Edge cases:
      - n == 0: the character has no frames; return min_multiply (it'll
        never be applied since there are no frames to repeat).
      - total == 0 or character_count == 0: defensive — no balancing
        possible, return min_multiply.
    """
    if n <= 0 or total <= 0 or character_count <= 0:
        return min_multiply
    # Mean frames per character; each character's "deficit" relative to
    # this mean is what the multiplier compensates for.
    mean = total / character_count
    raw = mean / n
    if raw < 1.0:
        # The character has more frames than the mean — no need to repeat
        # them. Floor at min_multiply so the trainer always sees them at
        # least once per epoch.
        return min_multiply
    return min(round(raw, 1), max_multiply)


def compute_character_balancing(
    *,
    project: Project,
    min_multiply: float = 1.0,
    max_multiply: float = 10.0,
) -> list[CharacterBalanceRow]:
    """Return one row per project character with auto + effective multipliers.

    Rows are returned in project-character order (the same order the UI
    renders them) so callers can render the table without re-sorting.
    Characters with zero frames still get a row — they show up in the UI
    so the user can see the empty state.
    """
    counts = _frame_counts_per_character(project)
    # Total + denominator for the auto formula come from characters that
    # actually have frames; an empty character would otherwise drag the
    # mean to zero and make every multiplier the floor.
    contributing = [c for c in project.characters if counts.get(c.slug, 0) > 0]
    total = sum(counts[c.slug] for c in contributing)
    denom = len(contributing)

    rows: list[CharacterBalanceRow] = []
    for c in project.characters:
        n = counts.get(c.slug, 0)
        auto = _auto_multiplier(
            n, total, denom,
            min_multiply=min_multiply, max_multiply=max_multiply,
        )
        manual = c.multiply
        effective = manual if manual > 0 else auto
        rows.append(CharacterBalanceRow(
            character_slug=c.slug,
            name=c.name,
            frame_count=n,
            auto_multiply=auto,
            manual_multiply=manual,
            effective_multiply=effective,
        ))
    return rows


def effective_multiplier_for(project: Project, character_slug: str) -> float:
    """Convenience: just the trainer-facing multiplier for one character.

    Used by the dataset-staging step that needs ``num_repeats`` to write
    into ``dataset.toml`` per-directory. Returns 1.0 for unknown slugs so a
    metadata row pointing at a deleted character degrades gracefully.
    """
    if project.character_by_slug(character_slug) is None:
        return 1.0
    rows = compute_character_balancing(project=project)
    for row in rows:
        if row.character_slug == character_slug:
            return row.effective_multiply
    return 1.0

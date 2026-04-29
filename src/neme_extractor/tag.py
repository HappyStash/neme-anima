"""WD14 (SmilingWolf v3) auto-tagging via imgutils.

Produces kohya-style comma-separated tag strings. Output is written as a sibling
``.txt`` file alongside each ``.png`` (the standard convention for kohya-ss /
OneTrainer / sd-scripts).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from imgutils.tagging import get_wd14_tags

from neme_extractor.config import TagConfig


@dataclass(frozen=True)
class TagResult:
    rating: dict[str, float]
    general: dict[str, float]
    character: dict[str, float]
    text: str  # comma-separated, ready to write to disk


class Tagger:
    """Holds tagging settings; calling ``tag(image)`` returns a TagResult."""

    def __init__(self, cfg: TagConfig | None = None) -> None:
        self.cfg = cfg or TagConfig()

    def tag(self, image_rgb: np.ndarray) -> TagResult:
        pil = Image.fromarray(image_rgb) if not isinstance(image_rgb, Image.Image) else image_rgb
        rating, general, character = get_wd14_tags(
            pil,
            model_name=self.cfg.model_name,
            general_threshold=self.cfg.general_threshold,
            character_threshold=self.cfg.character_threshold,
            no_underline=self.cfg.no_underline,
            drop_overlap=self.cfg.drop_overlap,
        )
        text = self._compose_text(general, character)
        return TagResult(rating=rating, general=general, character=character, text=text)

    def _compose_text(
        self,
        general: dict[str, float],
        character: dict[str, float],
    ) -> str:
        # Sort by descending confidence; character tags first (kohya convention is
        # often "char_name, general_tag, general_tag, ..." for character LoRAs).
        char_tags = [t for t, _ in sorted(character.items(), key=lambda kv: -kv[1])]
        general_tags = [t for t, _ in sorted(general.items(), key=lambda kv: -kv[1])]
        excluded = set(self.cfg.exclude_tags)
        all_tags = [t for t in (char_tags + general_tags) if t not in excluded]
        return ", ".join(all_tags)


def write_tags_sidecar(image_path: Path, tag_text: str) -> Path:
    """Write tag text to a ``.txt`` file next to the image. Returns the txt path."""
    txt = image_path.with_suffix(".txt")
    txt.write_text(tag_text + "\n", encoding="utf-8")
    return txt

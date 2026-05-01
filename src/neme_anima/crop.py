"""Final crop framing.

Aspect-preserved padded framing around the character, longest-side 1024 px,
**preserving original pixels** (no mask cutout, no flat-background letterbox).

Mask is optional: when produced via `isnetis` (anime-tuned segmenter from
imgutils), it is used to verify the framing quality (mask connectedness, mask
extent inside the bbox). The mask is NOT applied to the saved image — diffusion
LoRA training works best on natural backgrounds.

Note: the original design plan named SAM 3 for masking. Switched to imgutils'
`isnetis` here because it is anime-trained (much cleaner masks on anime),
already in our dependency tree, and significantly faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from neme_anima.config import CropConfig


@dataclass(frozen=True)
class CroppedFrame:
    """Result of framing: the saved-as-is image, plus optional mask aligned to it."""
    image_rgb: np.ndarray            # HxWx3 uint8, original pixels, longest side = cfg.longest_side
    mask: np.ndarray | None          # HxW uint8 binary aligned with image_rgb, or None
    source_bbox_in_frame: tuple[int, int, int, int]  # the (clipped) box used as the source crop, in original frame coords
    longest_side_px: int


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    pad_ratio: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px = int(round(w * pad_ratio))
    py = int(round(h * pad_ratio))
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(frame_w, x2 + px)
    ny2 = min(frame_h, y2 + py)
    return nx1, ny1, nx2, ny2


def _resize_longest(image: np.ndarray, target: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest == target:
        return image.copy()
    scale = target / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def crop_frame(
    frame_rgb: np.ndarray,
    detection_bbox: tuple[int, int, int, int],
    cfg: CropConfig,
    *,
    compute_mask: bool = True,
) -> CroppedFrame:
    """Apply aspect-preserved padded framing and resize. Background preserved.

    The detection bbox is padded by `cfg.pad_ratio`, clipped to the frame, then
    rescaled so the longest side equals `cfg.longest_side`. Pixels are sampled
    directly from the original frame — no mask is applied to the image.

    When `compute_mask=True`, the imgutils isnetis segmenter is run on the
    output crop and its binary character mask is returned for downstream
    quality use. The mask is NOT used to alter the image pixels.
    """
    h, w = frame_rgb.shape[:2]
    box = _pad_bbox(detection_bbox, cfg.pad_ratio, w, h)
    x1, y1, x2, y2 = box
    raw_crop = frame_rgb[y1:y2, x1:x2]
    if raw_crop.size == 0 or min(raw_crop.shape[:2]) < 8:
        # Degenerate. Return as-is; downstream will likely drop it.
        return CroppedFrame(
            image_rgb=raw_crop.copy() if raw_crop.size else np.zeros((1, 1, 3), dtype=np.uint8),
            mask=None,
            source_bbox_in_frame=box,
            longest_side_px=max(1, max(*raw_crop.shape[:2]) if raw_crop.size else 1),
        )

    resized = _resize_longest(raw_crop, cfg.longest_side)

    mask: np.ndarray | None = None
    if compute_mask:
        mask = _isnetis_binary_mask(resized)

    return CroppedFrame(
        image_rgb=resized,
        mask=mask,
        source_bbox_in_frame=box,
        longest_side_px=max(resized.shape[:2]),
    )


def _isnetis_binary_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Run isnetis on an RGB ndarray, return a uint8 binary mask aligned to it."""
    from imgutils.segment import get_isnetis_mask

    pil = Image.fromarray(image_rgb)
    raw = get_isnetis_mask(pil)
    arr = np.asarray(raw)
    # isnetis returns either a float [0,1] mask or HxW uint8 — normalize.
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    binary = (arr > 127).astype(np.uint8)
    # Resize mask to match image_rgb if shapes differ (isnetis may scale internally).
    h, w = image_rgb.shape[:2]
    if binary.shape != (h, w):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    return binary

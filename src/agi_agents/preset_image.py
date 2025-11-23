"""
image_preset.py

Utility functions for:
- Scoring a batch of images with an Image Quality Assessment (IQA) model
  using the `pyiqa` library.
- Automatically selecting the "best" image in a set (e.g., best frame in a burst).
- Applying a LUT-based preset to that best image using `lutlib`.

This is meant to be used by your PhotoWorkflowAgent for tasks like:
- Ingestion & culling (pick the hero shot from a sequence).
- Editing & enhancements (apply a style preset automatically).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import os

import numpy as np
from PIL import Image

import torch
import pyiqa
from lutlib import apply_lut

from agisdk.REAL.logging import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScoredImage:
    """Simple container: an image path and its IQA score."""

    path: Path
    score: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_paths(image_paths: Iterable[str | Path]) -> List[Path]:
    """
    Normalize and filter image paths.

    - Converts strings to Path objects.
    - Filters out non-existing files, logging a warning.
    """
    resolved: List[Path] = []

    for p in image_paths:
        path = Path(p)
        if path.is_file():
            resolved.append(path)
        else:
            logger.warning(f"[image_preset] Skipping non-existent file: {path}")

    if not resolved:
        raise ValueError("No valid image files found in image_paths.")

    return resolved


def _choose_device(device: str | None) -> str:
    """
    Pick an appropriate device.

    If device is provided, use it.
    Otherwise, use 'cuda' when available, else 'cpu'.
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_image_as_tensor(path: Path, device: str) -> torch.Tensor:
    """
    Load an image from disk and convert it to a normalized NCHW tensor on `device`.

    - Reads RGB image via Pillow.
    - Converts to float32 in [0, 1].
    - Shape: (1, 3, H, W)
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0  # H x W x 3
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_images(
    image_paths: Iterable[str | Path],
    metric_name: str = "paq2piq",
    device: str | None = None,
) -> Tuple[List[ScoredImage], bool]:
    """
    Score a collection of images using a pyiqa metric.

    Args:
        image_paths:
            List or iterable of paths to images.
        metric_name:
            Name of the pyiqa metric, e.g. 'paq2piq', 'niqe', 'brisque',
            'topiq_nr', etc.
        device:
            'cuda' or 'cpu'. If None, it will auto-select.

    Returns:
        (scores, lower_better)
        - scores: list of ScoredImage(path, score)
        - lower_better: bool indicating whether lower scores mean higher quality
          for this metric.
    """
    paths = _resolve_paths(image_paths)
    device = _choose_device(device)

    logger.info(
        f"[image_preset] Scoring {len(paths)} images with metric='{metric_name}' on device='{device}'"
    )

    # Create the IQA metric model
    metric = pyiqa.create_metric(metric_name, device=device)
    lower_better: bool = getattr(metric, "lower_better", False)

    results: List[ScoredImage] = []

    for path in paths:
        tensor = _load_image_as_tensor(path, device)
        with torch.no_grad():
            score_tensor = metric(tensor)
        score = float(score_tensor.item())
        results.append(ScoredImage(path=path, score=score))
        logger.debug(f"[image_preset] Scored {path.name}: {score:.4f}")

    return results, lower_better


def select_best_image(scores: List[ScoredImage], lower_better: bool) -> ScoredImage:
    """
    Select the best image based on scores and metric direction.

    Args:
        scores:
            List of ScoredImage objects.
        lower_better:
            If True, the image with the lowest score is best.
            If False, the image with the highest score is best.

    Returns:
        The chosen ScoredImage.
    """
    if not scores:
        raise ValueError("scores must not be empty")

    if lower_better:
        best = min(scores, key=lambda s: s.score)
    else:
        best = max(scores, key=lambda s: s.score)

    logger.info(
        f"[image_preset] Best image: {best.path.name} (score={best.score:.4f}, "
        f"{'lower' if lower_better else 'higher'} is better)"
    )

    return best


def apply_lut_preset(
    image_path: Path | str,
    lut_path: Path | str,
    output_dir: Path | str,
    suffix: str = "_preset",
    output_ext: str = ".jpg",
) -> Path:
    """
    Apply a LUT-based preset to a single image using `lutlib`.

    Args:
        image_path:
            Path to the source image.
        lut_path:
            Path to a .cube LUT file that encodes the preset.
        output_dir:
            Directory where the processed image will be stored.
        suffix:
            String appended to the original filename stem.
        output_ext:
            Output file extension (e.g. '.jpg', '.png').

    Returns:
        Path to the processed output image.
    """
    image_path = Path(image_path)
    lut_path = Path(lut_path)
    output_dir = Path(output_dir)

    if not lut_path.is_file():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{image_path.stem}{suffix}{output_ext}"
    output_path = output_dir / output_name

    logger.info(
        f"[image_preset] Applying LUT '{lut_path.name}' to '{image_path.name}' -> '{output_path.name}'"
    )

    # lutlib applies the LUT and writes directly to output_path
    apply_lut(str(lut_path), str(image_path), str(output_path))

    return output_path


def select_best_and_apply_preset(
    image_paths: Iterable[str | Path],
    lut_path: str | Path,
    output_dir: str | Path,
    metric_name: str = "paq2piq",
    device: str | None = None,
) -> Dict[str, Any]:
    """
    High-level helper that:
    1) Scores all candidate images using `pyiqa`.
    2) Chooses the best image based on the metric.
    3) Applies a LUT preset to that best image.
    4) Returns a summary dictionary.

    This is the main "callable tool" your agent will use for:
    - Automated hero-frame selection from a burst.
    - One-click preset application for that hero frame.

    Args:
        image_paths:
            Iterable of paths to candidate images.
        lut_path:
            Path to the LUT (.cube) file representing the preset.
        output_dir:
            Directory where the processed image will be stored.
        metric_name:
            pyiqa metric name (default: 'paq2piq' for perceptual quality).
        device:
            'cuda' or 'cpu'. If None, it will auto-detect.

    Returns:
        A dict with:
        {
            "best_image": Path,
            "best_score": float,
            "score_direction": "lower_better" or "higher_better",
            "output_image": Path,
            "scores": List[{"path": str, "score": float}],
        }
    """
    paths = _resolve_paths(image_paths)
    scores, lower_better = score_images(paths, metric_name=metric_name, device=device)
    best = select_best_image(scores, lower_better=lower_better)

    output_path = apply_lut_preset(
        image_path=best.path,
        lut_path=lut_path,
        output_dir=output_dir,
    )

    result = {
        "best_image": best.path,
        "best_score": best.score,
        "score_direction": "lower_better" if lower_better else "higher_better",
        "output_image": output_path,
        "scores": [
            {"path": s.path, "score": s.score}
            for s in scores
        ],
    }

    logger.info(
        f"[image_preset] select_best_and_apply_preset complete. "
        f"Best='{best.path.name}', Output='{output_path.name}'"
    )

    return result


# ---------------------------------------------------------------------------
# Optional: simple CLI for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick manual test:

    python -m agi_agents.image_preset \
        /path/to/img1.jpg /path/to/img2.jpg \
        --lut /path/to/preset.cube \
        --out /tmp/output_dir
    """
    import argparse

    parser = argparse.ArgumentParser("image_preset quick test")
    parser.add_argument("images", nargs="+", help="Image file paths")
    parser.add_argument("--lut", required=True, help="Path to .cube LUT file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--metric", default="paq2piq", help="pyiqa metric name")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'")

    args = parser.parse_args()

    summary = select_best_and_apply_preset(
        image_paths=args.images,
        lut_path=args.lut,
        output_dir=args.out,
        metric_name=args.metric,
        device=args.device,
    )

    print("Best image:", summary["best_image"])
    print("Best score:", summary["best_score"])
    print("Output image:", summary["output_image"])

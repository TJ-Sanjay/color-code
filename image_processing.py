"""
image_processing.py
-------------------
Classical image-processing core for ColorPrint.

No Streamlit code lives here. Everything is plain NumPy / OpenCV / Pillow so it
can be tested on its own (see test_core.py).

Pipeline
    bytes -> RGB array -> (optional HSV) -> quantized 3-D histogram
          -> normalized (sum = 1) -> flattened "color fingerprint"
          -> histogram intersection + Bhattacharyya distance
"""

from __future__ import annotations

import colorsys
import hashlib
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SUPPORTED_RESOLUTIONS = (64, 192, 512)

RESOLUTION_LABELS = {
    64: "Low (64 bins)",
    192: "Medium (192 bins)",
    512: "High (512 bins)",
}

# (bins_ch0, bins_ch1, bins_ch2) for every color space and resolution.
# The product of the three numbers MUST equal the resolution (checked below).
BIN_CONFIGS: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    "HSV": {64: (4, 4, 4), 192: (12, 4, 4), 512: (8, 8, 8)},
    "RGB": {64: (4, 4, 4), 192: (8, 6, 4), 512: (8, 8, 8)},
}

CHANNEL_NAMES = {
    "RGB": ("Red", "Green", "Blue"),
    "HSV": ("Hue", "Saturation", "Value"),
}

# Exclusive upper limit of each channel for cv2.calcHist.
# OpenCV 8-bit HSV: H in [0, 179], S in [0, 255], V in [0, 255].
CHANNEL_MAX = {
    "RGB": (256, 256, 256),
    "HSV": (180, 256, 256),
}

# Heuristic interpretation thresholds (documented, NOT universal).
INTERSECTION_HIGH = 70.0      # percent
INTERSECTION_MODERATE = 40.0  # percent
BHATTACHARYYA_HIGH = 0.30     # distance (lower = more similar)
BHATTACHARYYA_MODERATE = 0.55

INTERPRETATION_RULE = (
    "Heuristic rule: histogram intersection >= 70% is HIGH, 40-70% is MODERATE, "
    "below 40% is LOW. Bhattacharyya distance <= 0.30 is HIGH, 0.30-0.55 is "
    "MODERATE, above 0.55 is LOW. These cut-offs are rules of thumb chosen for "
    "this lab, not universally valid thresholds."
)

SEMANTIC_WARNING = (
    "Color-distribution similarity does not establish that two images contain "
    "the same objects or depict the same scene."
)


def _validate_all_configs() -> None:
    """Fail loudly at import time if a bin configuration is inconsistent."""
    for space, table in BIN_CONFIGS.items():
        for resolution, bins in table.items():
            if int(np.prod(bins)) != resolution:
                raise RuntimeError(
                    f"Invalid bin configuration {space} {bins}: "
                    f"product {int(np.prod(bins))} != {resolution}"
                )
        if set(table) != set(SUPPORTED_RESOLUTIONS):
            raise RuntimeError(f"{space} is missing a supported resolution.")


_validate_all_configs()


class ImageLoadError(Exception):
    """Raised when an uploaded file cannot be turned into a usable image."""


@dataclass
class LoadedImage:
    name: str
    rgb: np.ndarray      # H x W x 3, uint8, RGB order
    width: int
    height: int
    pixels: int
    fmt: str             # original file format, e.g. "JPEG"
    mode: str            # original Pillow mode, e.g. "RGBA", "L"
    had_alpha: bool
    digest: str = ""      # md5 of the file bytes (used for caching)


# --------------------------------------------------------------------------- #
# Step 1: load and normalize
# --------------------------------------------------------------------------- #

def load_image(data: bytes, filename: str) -> LoadedImage:
    """Decode an uploaded file into a consistent 8-bit RGB array."""
    if not data:
        raise ImageLoadError("The file is empty.")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode so corrupt files fail here
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError) as exc:
        raise ImageLoadError("It could not be read as a valid JPG/PNG image.") from exc

    fmt = img.format or "Unknown"
    original_mode = img.mode

    # Respect camera orientation stored in EXIF (phone photos).
    img = ImageOps.exif_transpose(img)

    has_alpha = (
        img.mode in ("RGBA", "LA", "PA", "La", "RGBa")
        or (img.mode == "P" and "transparency" in img.info)
    )

    if has_alpha:
        # Composite transparent pixels against a white background.
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        if img.mode in ("I", "I;16", "F"):
            # 16-bit / float grayscale: rescale to 8 bit before converting.
            arr = np.asarray(img).astype(np.float64)
            peak = arr.max()
            if peak > 255:
                arr = arr / peak * 255.0
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.convert("RGB")  # handles L (grayscale), P, CMYK, ...

    rgb = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ImageLoadError("The decoded image has an invalid shape.")

    height, width = rgb.shape[:2]
    return LoadedImage(
        name=filename, rgb=rgb, width=width, height=height,
        pixels=width * height, fmt=fmt, mode=original_mode, had_alpha=has_alpha,
        digest=hashlib.md5(data).hexdigest(),
    )


# --------------------------------------------------------------------------- #
# Steps 2 and 3: color space, quantized histogram, fingerprint
# --------------------------------------------------------------------------- #

def get_bins(color_space: str, resolution: int) -> Tuple[int, int, int]:
    """Return the validated bin configuration for a color space/resolution."""
    if color_space not in BIN_CONFIGS:
        raise ValueError(f"Unsupported color space: {color_space!r}")
    if resolution not in BIN_CONFIGS[color_space]:
        raise ValueError(
            f"Unsupported histogram resolution: {resolution!r}. "
            f"Choose one of {SUPPORTED_RESOLUTIONS}."
        )
    bins = BIN_CONFIGS[color_space][resolution]
    if int(np.prod(bins)) != resolution:
        raise ValueError("Invalid histogram configuration (bin product mismatch).")
    return bins


def compute_histogram(rgb: np.ndarray, color_space: str, resolution: int) -> np.ndarray:
    """
    Quantized 3-D color histogram, normalized so that all bins sum to 1.

    Returns an array of shape `bins` (e.g. 12 x 4 x 4 for HSV-192).
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Empty or invalid image array.")

    bins = get_bins(color_space, resolution)
    img = np.ascontiguousarray(rgb, dtype=np.uint8)

    if color_space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # H:0-179, S:0-255, V:0-255

    ranges = []
    for upper in CHANNEL_MAX[color_space]:
        ranges.extend([0, upper])  # calcHist ranges are [low, high) per channel

    hist = cv2.calcHist([img], [0, 1, 2], None, list(bins), ranges)
    hist = hist.astype(np.float64)

    total = hist.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Histogram is empty (sum is zero).")
    return hist / total


def fingerprint(hist: np.ndarray) -> np.ndarray:
    """Flatten a normalized 3-D histogram into a 1-D color fingerprint."""
    fp = np.asarray(hist, dtype=np.float64).reshape(-1)
    validate_fingerprint(fp)
    return fp


def validate_fingerprint(fp: np.ndarray) -> None:
    if fp.size == 0:
        raise ValueError("Fingerprint is empty.")
    if not np.all(np.isfinite(fp)):
        raise ValueError("Fingerprint contains NaN or infinite values.")
    if np.any(fp < 0):
        raise ValueError("Fingerprint contains negative values.")
    if not np.isclose(fp.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Fingerprint is not normalized (sum = {fp.sum():.6f}).")


def channel_marginal(hist: np.ndarray, axis: int) -> np.ndarray:
    """1-D distribution of a single channel (sum over the other two axes)."""
    other = tuple(i for i in range(3) if i != axis)
    return hist.sum(axis=other)


# --------------------------------------------------------------------------- #
# Step 4: dominant colors
# --------------------------------------------------------------------------- #

def _bin_center_rgb(index: Tuple[int, int, int], color_space: str,
                    bins: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Representative RGB color for a histogram cell (the cell's center)."""
    center = [
        (index[k] + 0.5) * CHANNEL_MAX[color_space][k] / bins[k] for k in range(3)
    ]
    if color_space == "RGB":
        r, g, b = (int(np.clip(round(c), 0, 255)) for c in center)
        return r, g, b

    h = int(np.clip(round(center[0]), 0, 179))
    s = int(np.clip(round(center[1]), 0, 255))
    v = int(np.clip(round(center[2]), 0, 255))
    pixel = np.array([[[h, s, v]]], dtype=np.uint8)
    r, g, b = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0]
    return int(r), int(g), int(b)


def bin_stats(rgb: np.ndarray, color_space: str, resolution: int):
    """
    Per-bin pixel counts and per-bin sums of the ORIGINAL RGB values.

    Used to show the real average color of each bin instead of its geometric
    center, so swatches look like the photo. Bin edges follow cv2.calcHist.
    Very large images are sub-sampled (means barely change).
    """
    bins = get_bins(color_space, resolution)
    h, w = rgb.shape[:2]
    if h * w > 4_000_000:
        step = int(np.ceil(np.sqrt(h * w / 4_000_000)))
        rgb = rgb[::step, ::step]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if color_space == "HSV" else rgb

    flat = np.zeros(img.shape[:2], dtype=np.int64)
    for k in range(3):
        scale = bins[k] / float(CHANNEL_MAX[color_space][k])
        idx = np.floor(img[..., k].astype(np.float64) * scale).astype(np.int64)
        flat = flat * bins[k] + np.clip(idx, 0, bins[k] - 1)
    flat = flat.ravel()

    cells = int(np.prod(bins))
    counts = np.bincount(flat, minlength=cells).astype(np.float64)
    pixels = rgb.reshape(-1, 3)
    sums = np.stack(
        [np.bincount(flat, weights=pixels[:, c].astype(np.float64), minlength=cells)
         for c in range(3)], axis=1)
    return counts, sums


def _cell_color(flat_index, idx, color_space, bins, stats):
    """Real mean color of a cell if known, otherwise the cell center."""
    if stats is not None:
        counts, sums = stats
        if counts[flat_index] > 0:
            r, g, b = (int(round(v)) for v in sums[flat_index] / counts[flat_index])
            return r, g, b
    return _bin_center_rgb(idx, color_space, bins)


def dominant_colors(hist: np.ndarray, color_space: str, resolution: int,
                    n: int, stats=None) -> List[dict]:
    """Top-n most populated bins. With `stats`, swatches are real mean colors."""
    bins = get_bins(color_space, resolution)
    flat = hist.reshape(-1)
    n = int(max(1, min(n, flat.size)))
    order = np.argsort(-flat, kind="stable")[:n]

    colors = []
    for flat_index in order:
        idx = tuple(int(i) for i in np.unravel_index(flat_index, bins))
        r, g, b = _cell_color(flat_index, idx, color_space, bins, stats)
        colors.append({
            "bin": idx,
            "name": color_name((r, g, b)),
            "rgb": (r, g, b),
            "hex": f"#{r:02X}{g:02X}{b:02X}",
            "percent": float(flat[flat_index] * 100.0),
        })
    return colors


def average_color(rgb: np.ndarray) -> Tuple[int, int, int]:
    """Mean RGB of the whole image (sub-sampled when very large)."""
    h, w = rgb.shape[:2]
    if h * w > 4_000_000:
        step = int(np.ceil(np.sqrt(h * w / 4_000_000)))
        rgb = rgb[::step, ::step]
    r, g, b = rgb.reshape(-1, 3).mean(axis=0)
    return int(round(r)), int(round(g)), int(round(b))

# --------------------------------------------------------------------------- #
# Human-friendly helpers (names, bin colors, shared bins)
# --------------------------------------------------------------------------- #

def color_name(rgb: Tuple[int, int, int]) -> str:
    """Rough everyday name for an RGB color (for labels only)."""
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:
        return "Black"
    if s < 0.12:
        if v > 0.88:
            return "White"
        if v > 0.6:
            return "Light gray"
        if v > 0.35:
            return "Gray"
        return "Dark gray"
    deg = h * 360.0
    for limit, name in ((12, "Red"), (40, "Orange"), (70, "Yellow"), (165, "Green"),
                        (200, "Cyan"), (255, "Blue"), (290, "Purple"), (335, "Pink"),
                        (361, "Red")):
        if deg < limit:
            base = name
            break
    if base in ("Orange", "Yellow") and v < 0.6:
        return "Brown" if base == "Orange" else "Olive"
    if v < 0.45:
        return "Dark " + base.lower()
    if s < 0.40 and v > 0.7:
        return "Pale " + base.lower()
    return base


def bin_color(index: Tuple[int, int, int], color_space: str,
              resolution: int) -> Tuple[int, int, int]:
    """Public wrapper: representative RGB color of one histogram cell."""
    return _bin_center_rgb(index, color_space, get_bins(color_space, resolution))


def top_union_bins(fp_a: np.ndarray, fp_b: np.ndarray, color_space: str,
                   resolution: int, k: int = 8, stats_a=None, stats_b=None) -> List[dict]:
    """The k bins that matter most to either image, with both percentages."""
    _check_pair(fp_a, fp_b)
    bins = get_bins(color_space, resolution)
    pooled = None
    if stats_a is not None and stats_b is not None:
        pooled = (stats_a[0] + stats_b[0], stats_a[1] + stats_b[1])
    order = np.argsort(-np.maximum(fp_a, fp_b), kind="stable")[:k]
    rows = []
    for flat_index in order:
        idx = tuple(int(i) for i in np.unravel_index(flat_index, bins))
        rgb = _cell_color(flat_index, idx, color_space, bins, pooled)
        rows.append({
            "bin": idx,
            "rgb": rgb,
            "hex": "#%02X%02X%02X" % rgb,
            "name": color_name(rgb),
            "a": float(fp_a[flat_index] * 100.0),
            "b": float(fp_b[flat_index] * 100.0),
        })
    return rows


def _pct(v: float) -> str:
    return f"{v:.0f}%" if v >= 1 else f"{v:.1f}%"


def _luma(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def describe_pair(rows: List[dict], avg_a, avg_b) -> List[str]:
    """Plain-language observations about the two images (markdown strings)."""
    out = []
    shared = sorted(rows, key=lambda r: min(r["a"], r["b"]), reverse=True)
    for r in shared[:2]:
        if min(r["a"], r["b"]) >= 3:
            out.append(f"Both images contain **{r['name'].lower()}** "
                       f"(Image A {_pct(r['a'])}, Image B {_pct(r['b'])}).")
    diff = sorted(rows, key=lambda r: r["a"] - r["b"])
    more_b, more_a = diff[0], diff[-1]
    if more_a["a"] - more_a["b"] >= 5:
        out.append(f"**{more_a['name']}** is much more common in Image A "
                   f"({_pct(more_a['a'])} vs {_pct(more_a['b'])}).")
    if more_b["b"] - more_b["a"] >= 5:
        out.append(f"**{more_b['name']}** is much more common in Image B "
                   f"({_pct(more_b['b'])} vs {_pct(more_b['a'])}).")
    la, lb = _luma(avg_a), _luma(avg_b)
    if abs(la - lb) >= 25:
        out.append(f"Image {'A' if la < lb else 'B'} is noticeably darker overall.")
    if not out:
        out.append("No single color stands out as very different between the two images.")
    return out

# --------------------------------------------------------------------------- #
# Step 5: similarity metrics
# --------------------------------------------------------------------------- #

def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Incompatible fingerprints: {a.shape} vs {b.shape}. "
            "Both images must use the same color space and bin configuration."
        )
    validate_fingerprint(a)
    validate_fingerprint(b)


def histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """S = sum(min(A_i, B_i)), in [0, 1]. Multiply by 100 for a percentage."""
    _check_pair(a, b)
    return float(np.clip(np.minimum(a, b).sum(), 0.0, 1.0))


def bhattacharyya_distance(a: np.ndarray, b: np.ndarray) -> float:
    """D = sqrt(1 - sum(sqrt(A_i * B_i))), in [0, 1]. Lower = more similar."""
    _check_pair(a, b)
    coefficient = np.sqrt(a * b).sum()          # Bhattacharyya coefficient
    inside = float(np.clip(1.0 - coefficient, 0.0, 1.0))  # kill float error
    return float(np.sqrt(inside))


def compare_images(rgb_a: np.ndarray, rgb_b: np.ndarray, color_space: str,
                   resolution: int) -> dict:
    """Run the whole pipeline on a pair using one shared configuration."""
    bins = get_bins(color_space, resolution)
    hist_a = compute_histogram(rgb_a, color_space, resolution)
    hist_b = compute_histogram(rgb_b, color_space, resolution)
    fp_a, fp_b = fingerprint(hist_a), fingerprint(hist_b)
    return {
        "bins": bins,
        "hist_a": hist_a,
        "hist_b": hist_b,
        "fp_a": fp_a,
        "fp_b": fp_b,
        "intersection": histogram_intersection(fp_a, fp_b),
        "bhattacharyya": bhattacharyya_distance(fp_a, fp_b),
    }


def experiment_table(rgb_a: np.ndarray, rgb_b: np.ndarray) -> List[dict]:
    """Metrics for every color space x resolution (Experiment Lab)."""
    rows = []
    for space in ("HSV", "RGB"):
        for resolution in SUPPORTED_RESOLUTIONS:
            result = compare_images(rgb_a, rgb_b, space, resolution)
            bins = result["bins"]
            rows.append({
                "Color space": space,
                "Resolution": RESOLUTION_LABELS[resolution],
                "Bin configuration": f"{bins[0]} x {bins[1]} x {bins[2]} = {resolution}",
                "Histogram intersection (%)": round(100.0 * result["intersection"], 2),
                "Bhattacharyya distance": round(result["bhattacharyya"], 4),
            })
    return rows


# --------------------------------------------------------------------------- #
# Step 6: interpretation
# --------------------------------------------------------------------------- #

def interpret_intersection(percent: float) -> str:
    if percent >= INTERSECTION_HIGH:
        return "high"
    if percent >= INTERSECTION_MODERATE:
        return "moderate"
    return "low"


def interpret_distance(distance: float) -> str:
    if distance <= BHATTACHARYYA_HIGH:
        return "high"
    if distance <= BHATTACHARYYA_MODERATE:
        return "moderate"
    return "low"

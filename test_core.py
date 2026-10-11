"""
Quick automated checks for the core maths. Run:  python test_core.py
(The manual UI checks are described in README.md.)
"""

import io

import numpy as np
from PIL import Image

import image_processing as ip


def png_bytes(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return buf.getvalue()


rng = np.random.default_rng(0)
img1 = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
img2 = np.zeros((90, 70, 3), dtype=np.uint8)
img2[..., 0] = 255  # pure red, different size

# 1. identical images
for space in ("HSV", "RGB"):
    for res in ip.SUPPORTED_RESOLUTIONS:
        r = ip.compare_images(img1, img1, space, res)
        assert abs(r["intersection"] - 1.0) < 1e-9, (space, res)
        assert r["bhattacharyya"] < 1e-6, (space, res)
        assert r["fp_a"].size == res and abs(r["fp_a"].sum() - 1) < 1e-9
print("1. identical images -> 100% / 0.0  OK")

# 3 + 4. very different colors and different resolutions
r = ip.compare_images(img1, img2, "HSV", 192)
assert r["intersection"] < 0.2 and r["bhattacharyya"] > 0.7
print(f"3/4. different images (different sizes): I={100*r['intersection']:.1f}%  "
      f"D={r['bhattacharyya']:.3f}  OK")

# resizing the same image should barely change the fingerprint
small = np.ascontiguousarray(img1[::2, ::2])
r = ip.compare_images(img1, small, "RGB", 64)
assert r["intersection"] > 0.85
print(f"4. same image at half size: I={100*r['intersection']:.1f}%  OK")

# worked example from the About section
a, b = np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.25, 0.25])
assert abs(ip.histogram_intersection(a, b) - 0.75) < 1e-12
assert abs(ip.bhattacharyya_distance(a, b) - 0.3827) < 1e-3
print("Worked example 0.75 / 0.383  OK")

# incompatible histograms must raise, never silently compare
try:
    ip.histogram_intersection(np.ones(64) / 64, np.ones(192) / 192)
    raise SystemExit("incompatible shapes were accepted!")
except ValueError:
    print("Incompatible fingerprints rejected  OK")

# loader: grayscale, RGBA, corrupt
gray = png_bytes(rng.integers(0, 256, (30, 40), dtype=np.uint8))
loaded = ip.load_image(gray, "gray.png")
assert loaded.rgb.shape == (30, 40, 3)

rgba = np.zeros((20, 20, 4), dtype=np.uint8)  # fully transparent
loaded = ip.load_image(png_bytes(rgba), "rgba.png")
assert loaded.had_alpha and (loaded.rgb == 255).all()  # white background

try:
    ip.load_image(b"not an image", "bad.png")
    raise SystemExit("corrupt file was accepted!")
except ip.ImageLoadError:
    pass
print("Grayscale / RGBA / corrupt-file handling  OK")

# dominant colors: swatches must be the REAL mean color of the pixels
stats = ip.bin_stats(img2, "HSV", 192)
cols = ip.dominant_colors(ip.compute_histogram(img2, "HSV", 192), "HSV", 192, 3, stats)
assert cols[0]["percent"] > 99 and cols[0]["hex"] == "#FF0000" and cols[0]["name"] == "Red"
print("Dominant color of a pure-red image is exactly", cols[0]["hex"], " OK")

# bin_stats must use the same bins as cv2.calcHist
for space in ("HSV", "RGB"):
    for res in ip.SUPPORTED_RESOLUTIONS:
        counts, _ = ip.bin_stats(img1, space, res)
        hist = ip.compute_histogram(img1, space, res).reshape(-1)
        assert np.allclose(counts / counts.sum(), hist, atol=1e-12), (space, res)
print("bin_stats agrees with calcHist for all 6 configurations  OK")

# plain-language insights run and mention the colors
rows = ip.top_union_bins(ip.compare_images(img1, img2, "HSV", 192)["fp_a"],
                         ip.compare_images(img1, img2, "HSV", 192)["fp_b"], "HSV", 192, 10)
assert len(ip.describe_pair(rows, ip.average_color(img1), ip.average_color(img2))) >= 1
print("describe_pair  OK")

print("\nAll checks passed.")

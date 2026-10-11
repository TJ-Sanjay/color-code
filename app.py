"""
app.py - ColorPrint: compare the colors of two images.

Run with:  streamlit run app.py
All image-processing logic lives in image_processing.py.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

import image_processing as ip

COLOR_A = "#2f8f8a"      # Image A (teal)
COLOR_B = "#d9772f"      # Image B (orange)
SHARED = "#2e9e6b"       # shared / overlapping colors (green)
LEVEL_COLORS = {"high": "#2e9e6b", "moderate": "#d99a1e", "low": "#d4553f"}

st.set_page_config(page_title="ColorPrint", page_icon="🎨", layout="wide")


# --------------------------------------------------------------------------- #
# Theme handling (works in light and dark mode)
# --------------------------------------------------------------------------- #
def apply_chart_theme():
    try:
        is_dark = st.context.theme.type == "dark"
    except Exception:
        is_dark = False
    fg = "#ece9e4" if is_dark else "#1c1917"
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none", "savefig.transparent": True,
        "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": fg,
        "xtick.color": fg, "ytick.color": fg, "font.size": 10,
    })


apply_chart_theme()

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; max-width: 1180px; }
    .hero-title { font-size: 2.7rem; font-weight: 800; margin: 0; line-height: 1.1; }
    .hero-sub { opacity: .75; font-size: 1.15rem; margin: .3rem 0 1.1rem; }
    .steps { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 22px; }
    .step { flex: 1; min-width: 210px; border: 1px solid rgba(128,128,128,.30);
            border-radius: 14px; padding: 12px 14px; background: rgba(128,128,128,.07); }
    .step-n { display: inline-block; width: 24px; height: 24px; border-radius: 50%;
              background: #2f8f8a; color: #fff; text-align: center; line-height: 24px;
              font-weight: 700; margin-right: 8px; font-size: 13px; }
    .step small { display: block; opacity: .75; margin-top: 4px; }
    .verdict { border: 1px solid rgba(128,128,128,.30); border-left-width: 8px;
               border-radius: 14px; padding: 16px 20px; background: rgba(128,128,128,.07);
               margin-bottom: 10px; }
    .v-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 4px; }
    .v-text { opacity: .85; }
    .tile { border: 1px solid rgba(128,128,128,.30); border-radius: 14px;
            padding: 12px 16px; background: rgba(128,128,128,.07); margin-bottom: 10px; }
    .tile-label { font-size: .85rem; opacity: .7; }
    .tile-big { font-size: 2rem; font-weight: 800; line-height: 1.25; }
    .meter { height: 10px; border-radius: 6px; background: rgba(128,128,128,.25);
             margin: 6px 0; overflow: hidden; }
    .meter > div { height: 100%; border-radius: 6px; }
    .tile-note { font-size: .8rem; opacity: .7; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
    .chip { font-size: .78rem; border: 1px solid rgba(128,128,128,.35);
            border-radius: 999px; padding: 2px 10px; background: rgba(128,128,128,.08); }
    .sw-row { display: flex; flex-wrap: wrap; gap: 10px; }
    .sw { width: 138px; border: 1px solid rgba(128,128,128,.35); border-radius: 12px;
          overflow: hidden; background: rgba(128,128,128,.07); }
    .sw-c { height: 66px; }
    .sw-t { padding: 8px 10px; line-height: 1.5; font-size: 12.5px; }
    .sw-name { font-weight: 700; font-size: 13.5px; }
    .sw-pct { font-weight: 700; }
    .avg { display: flex; align-items: center; gap: 10px; margin: 4px 0 12px; }
    .avg-box { width: 34px; height: 34px; border-radius: 8px;
               border: 1px solid rgba(128,128,128,.5); }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached heavy work
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_load(data: bytes, name: str):
    return ip.load_image(data, name)


@st.cache_data(show_spinner="Analyzing colors...")
def cached_compare(digest_a, digest_b, color_space, resolution, _rgb_a, _rgb_b):
    result = ip.compare_images(_rgb_a, _rgb_b, color_space, resolution)
    result["stats_a"] = ip.bin_stats(_rgb_a, color_space, resolution)
    result["stats_b"] = ip.bin_stats(_rgb_b, color_space, resolution)
    return result


@st.cache_data(show_spinner=False)
def cached_experiment(digest_a, digest_b, _rgb_a, _rgb_b):
    return ip.experiment_table(_rgb_a, _rgb_b)


def load_uploaded(upload):
    if upload is None:
        return None
    try:
        return cached_load(upload.getvalue(), upload.name)
    except ip.ImageLoadError as exc:
        st.error(f"**{upload.name}**: {exc}")
    except Exception:  # last-resort guard: never show a stack trace
        st.error(f"**{upload.name}**: something went wrong while reading this file.")
    return None


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def pct_text(v):
    return f"{v:.0f}%" if v >= 1 else f"{v:.1f}%"


def chips_html(img):
    extra = ""
    if img.had_alpha:
        extra = '<span class="chip">transparency \u2192 white</span>'
    elif img.mode in ("L", "LA", "I", "I;16"):
        extra = '<span class="chip">grayscale \u2192 RGB</span>'
    return (
        '<div class="chips">'
        f'<span class="chip">{img.width} \u00d7 {img.height} px</span>'
        f'<span class="chip">{img.pixels / 1e6:.2f} megapixels</span>'
        f'<span class="chip">{img.fmt}</span>{extra}</div>'
    )


def tile_html(label, big, fill_pct, color, note):
    fill_pct = float(np.clip(fill_pct, 0, 100))
    return (
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-big">{big}</div>'
        f'<div class="meter"><div style="width:{fill_pct:.1f}%;background:{color}"></div></div>'
        f'<div class="tile-note">{note}</div></div>'
    )


def palette_html(colors):
    cards = ""
    for c in colors:
        r, g, b = c["rgb"]
        cards += (
            f'<div class="sw"><div class="sw-c" style="background:{c["hex"]}"></div>'
            f'<div class="sw-t"><div class="sw-name">{c["name"]}</div>'
            f'<div class="sw-pct">{pct_text(c["percent"])} of pixels</div>'
            f'{c["hex"]}<br>RGB({r}, {g}, {b})</div></div>'
        )
    return f'<div class="sw-row">{cards}</div>'


def avg_html(rgb):
    hex_code = "#%02X%02X%02X" % rgb
    return (
        f'<div class="avg"><div class="avg-box" style="background:{hex_code}"></div>'
        f'<div><b>Average color</b> of the whole image<br>{hex_code} \u00b7 '
        f'RGB({rgb[0]}, {rgb[1]}, {rgb[2]})</div></div>'
    )


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
CHANNEL_LABELS = {
    "HSV": ("Hue (which color)", "Saturation (how vivid)", "Brightness (how light)"),
    "RGB": ("Red", "Green", "Blue"),
}
AXIS_TITLES = {
    "HSV": ("Hue - the color itself, as an angle from 0\u00b0 to 360\u00b0",
            "Saturation - from 0% (gray) to 100% (vivid)",
            "Brightness (Value) - from 0% (black) to 100% (bright)"),
    "RGB": ("Red level (0 = none, 255 = full)",
            "Green level (0 = none, 255 = full)",
            "Blue level (0 = none, 255 = full)"),
}
CHANNEL_HELP = {
    ("HSV", 0): "Hue is the 'pure color' (red, orange, yellow, green, blue...). Tall bars mean "
                "many pixels have that color. Gray, black and white pixels have no real hue, "
                "so check the saturation chart too.",
    ("HSV", 1): "Saturation tells how vivid a color is. Bars on the left = gray or washed-out "
                "pixels, bars on the right = strong, vivid colors.",
    ("HSV", 2): "Brightness tells how light or dark pixels are. Bars on the left = dark "
                "pixels, bars on the right = bright pixels.",
    ("RGB", 0): "How much red light each pixel has. Left = little red, right = lots of red.",
    ("RGB", 1): "How much green light each pixel has. Left = little green, right = lots of green.",
    ("RGB", 2): "How much blue light each pixel has. Left = little blue, right = lots of blue.",
}


def bin_tick_labels(color_space, k, n):
    labels = []
    for i in range(n):
        if color_space == "HSV" and k == 0:
            labels.append(f"{round(i * 360 / n)}-{round((i + 1) * 360 / n)}\u00b0")
        elif color_space == "HSV":
            labels.append(f"{round(i * 100 / n)}-{round((i + 1) * 100 / n)}%")
        else:
            labels.append(f"{round(i * 256 / n)}-{round((i + 1) * 256 / n) - 1}")
    return labels


def bar_colors(color_space, k, nbins):
    out = []
    for i in range(nbins):
        t = (i + 0.5) / nbins
        if color_space == "RGB":
            rgb = [0.0, 0.0, 0.0]
            rgb[k] = 0.15 + 0.85 * t
            out.append(tuple(rgb))
        elif k == 0:
            out.append(mcolors.hsv_to_rgb((t, 1.0, 1.0)))
        elif k == 1:
            out.append(mcolors.hsv_to_rgb((0.58, 0.1 + 0.9 * t, 0.95)))
        else:
            g = 0.12 + 0.88 * t
            out.append((g, g, g))
    return out


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def text_color_for(rgb):
    return "black" if ip._luma(rgb) > 140 else "white"


def plot_composition(colors_a, colors_b):
    """Two bars = 100% of each image, split into its dominant colors."""
    fig, ax = plt.subplots(figsize=(11, 2.9))
    for y, colors in ((1, colors_a), (0, colors_b)):
        left = 0.0
        for c in colors:
            w = c["percent"]
            ax.barh(y, w, left=left, height=0.64, color=c["hex"],
                    edgecolor="#888", linewidth=0.6)
            if w >= 6:
                ax.text(left + w / 2, y, f'{c["name"]}\n{w:.0f}%', ha="center",
                        va="center", fontsize=8.5, color=text_color_for(c["rgb"]))
            left += w
        rest = max(0.0, 100.0 - left)
        if rest > 0.05:
            ax.barh(y, rest, left=left, height=0.64, color="#9a9a9a", alpha=0.3,
                    hatch="//", edgecolor="#888", linewidth=0.6)
            if rest >= 8:
                ax.text(left + rest / 2, y, f"Other colors\n{rest:.0f}%", ha="center",
                        va="center", fontsize=8.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Image B", "Image A"], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the image's pixels (%)")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_overlap(inter_pct):
    """How the score is made: shared part vs. part found in only one image."""
    fig, ax = plt.subplots(figsize=(11, 2.3))
    rest = 100.0 - inter_pct
    for y, label, col in ((1, "Image A", COLOR_A), (0, "Image B", COLOR_B)):
        ax.barh(y, inter_pct, height=0.6, color=SHARED)
        ax.barh(y, rest, left=inter_pct, height=0.6, color=col, alpha=0.45)
        if inter_pct >= 10:
            ax.text(inter_pct / 2, y, f"Shared {inter_pct:.0f}%", ha="center",
                    va="center", color="white", fontweight="bold")
        if rest >= 18:
            ax.text(inter_pct + rest / 2, y, f"Only in {label}: {rest:.0f}%",
                    ha="center", va="center", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Image B", "Image A"], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the image's pixels (%)")
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_butterfly(rows):
    """Mirror chart: Image A bars left, Image B bars right, color in the middle."""
    n = len(rows)
    fig = plt.figure(figsize=(11, 0.56 * n + 1.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 2.3, 5], wspace=0.03)
    axl, axm, axr = (fig.add_subplot(gs[0, i]) for i in range(3))
    ys = np.arange(n)[::-1]
    xmax = max(max(r["a"], r["b"]) for r in rows) * 1.22 + 0.5

    for ax in (axl, axr, axm):
        ax.set_ylim(-0.7, n - 0.3)
    axl.barh(ys, [r["a"] for r in rows], height=0.66, color=COLOR_A)
    axr.barh(ys, [r["b"] for r in rows], height=0.66, color=COLOR_B)
    axl.set_xlim(xmax, 0)
    axr.set_xlim(0, xmax)
    for y, r in zip(ys, rows):
        axl.text(r["a"], y, f' {pct_text(r["a"])} ', ha="right", va="center", fontsize=9)
        axr.text(r["b"], y, f' {pct_text(r["b"])}', ha="left", va="center", fontsize=9)
    for ax in (axl, axr):
        ax.set_yticks([])
        _clean(ax)
        ax.set_xlabel("% of this image's pixels")
    axl.set_title("Image A", color=COLOR_A, fontweight="bold")
    axr.set_title("Image B", color=COLOR_B, fontweight="bold")

    axm.set_xlim(0, 1)
    axm.axis("off")
    for y, r in zip(ys, rows):
        axm.add_patch(Rectangle((0.04, y - 0.3), 0.24, 0.6, fc=r["hex"], ec="#888", lw=0.8))
        axm.text(0.33, y, f'{r["name"]}\n{r["hex"]}', va="center", fontsize=8.5, linespacing=1.15)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.12)
    return fig


def plot_channel(hist_a, hist_b, color_space, bins, k):
    """One channel, Image A on top and Image B below, bars in their real colors."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, sharey=True)
    x = np.arange(bins[k])
    colors = bar_colors(color_space, k, bins[k])
    margins = [ip.channel_marginal(h, k) * 100.0 for h in (hist_a, hist_b)]
    ymax = max(m.max() for m in margins)
    for ax, pct, label in zip(axes, margins, ("Image A", "Image B")):
        ax.bar(x, pct, color=colors, edgecolor="#888", linewidth=0.6, width=0.82)
        for xi, v in zip(x, pct):
            if v >= 1:
                ax.text(xi, v + ymax * 0.015, f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=9)
        ax.set_ylabel(f"{label}\n% of pixels", fontsize=10)
        ax.set_ylim(0, ymax * 1.2 + 1)
        _clean(ax)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_tick_labels(color_space, k, bins[k]),
                            rotation=30 if bins[k] > 6 else 0,
                            ha="right" if bins[k] > 6 else "center", fontsize=9)
    axes[1].set_xlabel(AXIS_TITLES[color_space][k])
    fig.tight_layout()
    return fig


def plot_channel_overlay(hist_a, hist_b, color_space, bins, k):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(bins[k])
    ma = ip.channel_marginal(hist_a, k) * 100.0
    mb = ip.channel_marginal(hist_b, k) * 100.0
    ax.bar(x - 0.2, ma, 0.4, color=COLOR_A, label="Image A")
    ax.bar(x + 0.2, mb, 0.4, color=COLOR_B, label="Image B")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_tick_labels(color_space, k, bins[k]),
                       rotation=30 if bins[k] > 6 else 0,
                       ha="right" if bins[k] > 6 else "center", fontsize=9)
    ax.set_xlabel(AXIS_TITLES[color_space][k])
    ax.set_ylabel("% of pixels")
    ax.legend(frameon=False)
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_fingerprints(fp_a, fp_b):
    x = np.arange(fp_a.size)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.fill_between(x, np.minimum(fp_a, fp_b) * 100, step="mid", color="#8d8d8d",
                    alpha=0.35, label="Overlap = min(A, B)")
    ax.plot(x, fp_a * 100, drawstyle="steps-mid", lw=1.1, color=COLOR_A, label="Image A")
    ax.plot(x, fp_b * 100, drawstyle="steps-mid", lw=1.1, color=COLOR_B, label="Image B")
    ax.set_xlabel("Histogram bin index (each index is a separate color cell; "
                  "neighbouring indices are not 'similar colors')")
    ax.set_ylabel("% of pixels")
    ax.legend(frameon=False, ncol=3)
    _clean(ax)
    fig.tight_layout()
    return fig


def plot_experiment(rows):
    labels = [f'{r["Color space"]}\n{r["Resolution"].split("(")[1].rstrip(")")}' for r in rows]
    cols = [COLOR_A if r["Color space"] == "HSV" else COLOR_B for r in rows]
    inter = [r["Histogram intersection (%)"] for r in rows]
    dist = [r["Bhattacharyya distance"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, vals, title, fmt in (
        (a1, inter, "Color overlap (%) - higher = more similar", "{:.0f}%"),
        (a2, dist, "Color distance - lower = more similar", "{:.2f}"),
    ):
        ax.bar(range(len(vals)), vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2 + 0.01)
        _clean(ax)
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report / About
# --------------------------------------------------------------------------- #
def build_report(summary, lines, rows, insights):
    text = [
        "COLORPRINT - COMPARISON REPORT",
        "=" * 34,
        "",
        f"Image A : {summary['image_a_filename']} "
        f"({summary['image_a_width']} x {summary['image_a_height']})",
        f"Image B : {summary['image_b_filename']} "
        f"({summary['image_b_width']} x {summary['image_b_height']})",
        f"Color space          : {summary['color_space']}",
        f"Histogram resolution : {summary['histogram_resolution_bins']} bins "
        f"({summary['bin_configuration']})",
        "",
        f"Histogram intersection : {summary['histogram_intersection_percent']:.2f} %",
        f"Bhattacharyya distance : {summary['bhattacharyya_distance']:.4f}",
        "",
        "Interpretation",
        "--------------",
        *lines,
        ip.INTERPRETATION_RULE,
        "",
        "In plain words",
        "--------------",
        *[f"- {i.replace('**', '')}" for i in insights],
        "",
        "WARNING: " + ip.SEMANTIC_WARNING,
        "",
        "Experiment Lab (same pair, all configurations)",
        "----------------------------------------------",
    ]
    for r in rows:
        text.append(
            f"{r['Color space']} | {r['Resolution']:<18} | {r['Bin configuration']:<18} | "
            f"I = {r['Histogram intersection (%)']:6.2f} % | D = {r['Bhattacharyya distance']:.4f}"
        )
    text += [
        "",
        "Algorithm",
        "---------",
        "1. Load both images and convert to 8-bit RGB (alpha composited on white).",
        "2. Optionally convert to HSV (OpenCV ranges H 0-179, S 0-255, V 0-255).",
        "3. Quantize each channel into bins and count pixels in a 3-D histogram.",
        "4. Normalize the histogram to sum to 1 and flatten it: the color fingerprint.",
        "5. Histogram intersection S = sum(min(A_i, B_i)).",
        "6. Bhattacharyya distance D = sqrt(1 - sum(sqrt(A_i * B_i))).",
    ]
    return "\n".join(text)


ABOUT_MD = r"""
**Image histogram.** Counts how many pixels fall into each intensity or color
range. It records *how much* of each color is present but throws away *where*
it is.

**Color fingerprint.** A 3-D histogram (one axis per channel) is divided into
cells, normalized and flattened into one vector. With 192 bins the fingerprint is a
list of 192 numbers that sum to 1 - a compact summary of an image's color
distribution.

**RGB vs HSV.** RGB mixes brightness and color: a shadowed red and a bright red
land in very different RGB cells. HSV separates *Hue* (which color),
*Saturation* (how vivid) and *Value* (how bright), so hue-based fingerprints are
usually more robust to lighting changes. OpenCV stores H in 0-179, S and V in
0-255.

**Quantization.** Instead of 256 x 256 x 256 = 16.7 million possible colors, each
channel is grouped into a few bins (for example 8 x 8 x 8 = 512). This shrinks the
fingerprint and makes it tolerant to small color shifts, at the cost of detail.
Fewer bins usually means coarser, higher similarity scores.

**Normalization.** Dividing by the total pixel count makes every bin a fraction
of the image, so a 4000 x 3000 photo and a 400 x 300 thumbnail can be compared
fairly.

**Histogram intersection.** For normalized histograms $A$ and $B$:

$$S = \sum_i \min(A_i, B_i)$$

It is the shared area under both histograms (0 = nothing in common, 1 = identical).

**Bhattacharyya distance.**

$$BC = \sum_i \sqrt{A_i B_i}, \qquad D = \sqrt{1 - BC}$$

$D = 0$ for identical distributions and $D = 1$ when they share no bins at all.

**Worked example.** $A = (0.5, 0.5, 0)$, $B = (0.5, 0.25, 0.25)$.
Intersection $= 0.5 + 0.25 + 0 = 0.75$ (75%).
$BC = 0.5 + 0.354 + 0 = 0.854$, so $D = \sqrt{1 - 0.854} \approx 0.383$.

**Swatch colors.** Each dominant-color swatch shows the real *average color of the
pixels* that fall into that bin, so it looks like the photo. The bin itself is
only used to decide which pixels belong together.

**Color similarity is not semantic similarity.** A red car and a red apple can have
near-identical fingerprints, while the same scene at sunrise and at noon can have
very different ones. The method ignores shapes, texture and spatial layout.

**Advantages:** simple, fast, no training, invariant to image size, rotation and
position of objects.
**Limitations:** no spatial information, sensitive to lighting (especially in RGB),
sensitive to the chosen binning, and blind to objects and meaning.
"""


def render_about():
    with st.expander("About the algorithm (for the curious)", expanded=False):
        st.markdown(ABOUT_MD)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    '<p class="hero-title">\U0001F3A8 ColorPrint</p>'
    '<p class="hero-sub">Interactive Image Color Fingerprinting and Similarity Analysis'
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "ColorPrint compares two images by their **color distributions** using classical "
    "image processing (no machine learning)."
)
st.markdown(
    """
    <div class="steps">
      <div class="step"><span class="step-n">1</span><b style="display:inline">Upload two images</b>
        <small>JPG or PNG, any size.</small></div>
      <div class="step"><span class="step-n">2</span><b style="display:inline">Count the colors</b>
        <small>Each image becomes a color histogram, its "fingerprint".</small></div>
      <div class="step"><span class="step-n">3</span><b style="display:inline">Compare</b>
        <small>Two scores say how much the color lists overlap.</small></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.header("Analysis settings")
color_space = st.sidebar.radio(
    "Color model",
    ["HSV", "RGB"],
    format_func=lambda c: {"HSV": "HSV (hue, saturation, brightness)",
                           "RGB": "RGB (red, green, blue)"}[c],
    help="How colors are described before counting. HSV copes better with lighting changes.",
)
resolution = st.sidebar.selectbox(
    "Detail level (histogram resolution)",
    ip.SUPPORTED_RESOLUTIONS,
    index=1,
    format_func=lambda r: ip.RESOLUTION_LABELS[r],
    help="More bins = finer color groups. Fewer bins = nearby colors are merged.",
)
n_colors = st.sidebar.slider("Number of dominant colors", 3, 10, 6)
metric_choice = st.sidebar.radio(
    "Similarity metric",
    ["Histogram intersection", "Bhattacharyya distance", "Show both"],
    index=2,
)
detailed = st.sidebar.checkbox(
    "Show detailed histogram plots",
    value=False,
    help="Adds technical charts to the Histograms tab.",
)

try:
    sel_bins = ip.get_bins(color_space, resolution)
except ValueError as exc:
    st.sidebar.error(str(exc))
    st.stop()

st.sidebar.caption(
    f"**{color_space}** bins: {sel_bins[0]} x {sel_bins[1]} x {sel_bins[2]} = {resolution}. "
    "Each resolution is a *quantized* representation of color space: more bins keep more "
    "color detail, fewer bins merge nearby colors."
)

# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
st.subheader("1. Choose your images")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f'<b style="color:{COLOR_A};font-size:1.15rem">Image A</b>',
                unsafe_allow_html=True)
    upload_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"], key="a",
                                label_visibility="collapsed")
    img_a = load_uploaded(upload_a)
    if img_a:
        st.image(img_a.rgb, width="stretch")
        st.markdown(f"**{img_a.name}**")
        st.markdown(chips_html(img_a), unsafe_allow_html=True)
with col_b:
    st.markdown(f'<b style="color:{COLOR_B};font-size:1.15rem">Image B</b>',
                unsafe_allow_html=True)
    upload_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"], key="b",
                                label_visibility="collapsed")
    img_b = load_uploaded(upload_b)
    if img_b:
        st.image(img_b.rgb, width="stretch")
        st.markdown(f"**{img_b.name}**")
        st.markdown(chips_html(img_b), unsafe_allow_html=True)

if img_a is None or img_b is None:
    st.info("Upload **both** images to start the comparison.")
    render_about()
    st.stop()

# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
try:
    result = cached_compare(img_a.digest, img_b.digest, color_space, resolution,
                            img_a.rgb, img_b.rgb)
    exp_rows = cached_experiment(img_a.digest, img_b.digest, img_a.rgb, img_b.rgb)
except (ValueError, ip.ImageLoadError) as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

hist_a, hist_b = result["hist_a"], result["hist_b"]
fp_a, fp_b = result["fp_a"], result["fp_b"]
stats_a, stats_b = result["stats_a"], result["stats_b"]
bins = result["bins"]
inter_pct = 100.0 * result["intersection"]
bhatt = result["bhattacharyya"]

colors_a = ip.dominant_colors(hist_a, color_space, resolution, n_colors, stats_a)
colors_b = ip.dominant_colors(hist_b, color_space, resolution, n_colors, stats_b)
rows_top = ip.top_union_bins(fp_a, fp_b, color_space, resolution, 10, stats_a, stats_b)
avg_a, avg_b = ip.average_color(img_a.rgb), ip.average_color(img_b.rgb)
insights = ip.describe_pair(rows_top, avg_a, avg_b)

lvl_inter = ip.interpret_intersection(inter_pct)
lvl_dist = ip.interpret_distance(bhatt)
main_level = lvl_dist if metric_choice == "Bhattacharyya distance" else lvl_inter

TITLES = {
    "high": ("Very similar colors",
             "The two images use very similar amounts of the same colors."),
    "moderate": ("Partly similar colors",
                 "The images share some colors, but each also has colors the other does not."),
    "low": ("Mostly different colors",
            "The two images mostly use different colors."),
}

# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
st.subheader("2. Result")
left, right = st.columns([3, 2])
with left:
    title, text = TITLES[main_level]
    st.markdown(
        f'<div class="verdict" style="border-left-color:{LEVEL_COLORS[main_level]}">'
        f'<div class="v-title">{title}</div><div class="v-text">{text}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**In plain words**")
    for line in insights:
        st.markdown(f"- {line}")
with right:
    if metric_choice in ("Histogram intersection", "Show both"):
        st.markdown(
            tile_html("Color overlap (histogram intersection)", f"{inter_pct:.1f}%",
                      inter_pct, LEVEL_COLORS[lvl_inter],
                      "How much the two color lists overlap. 100% = identical, 0% = nothing shared."),
            unsafe_allow_html=True)
    if metric_choice in ("Bhattacharyya distance", "Show both"):
        st.markdown(
            tile_html("Color distance (Bhattacharyya)", f"{bhatt:.3f}",
                      bhatt * 100, LEVEL_COLORS[lvl_dist],
                      "0 = identical color lists, 1 = completely different."),
            unsafe_allow_html=True)
    st.markdown(
        tile_html("Fingerprint size", f"{fp_a.size} values", 100, "#7c8a9a",
                  f"{color_space}, {bins[0]} x {bins[1]} x {bins[2]} color groups, "
                  f"normalized (sum A = {fp_a.sum():.4f}, sum B = {fp_b.sum():.4f})."),
        unsafe_allow_html=True)

st.caption(
    f"Image sizes: A is {img_a.width} x {img_a.height}, B is {img_b.width} x {img_b.height} "
    "(sizes do not matter, because the histograms are normalized).  \n"
    + ip.INTERPRETATION_RULE
)
st.warning(ip.SEMANTIC_WARNING)

# Interpretation lines (also used in the downloadable report)
lines = []
if metric_choice in ("Histogram intersection", "Show both"):
    lines.append(f"Histogram intersection of {inter_pct:.1f}% -> {lvl_inter} "
                 "color-distribution similarity.")
if metric_choice in ("Bhattacharyya distance", "Show both"):
    lines.append(f"Bhattacharyya distance of {bhatt:.3f} -> {lvl_dist} "
                 "color-distribution similarity.")

# --------------------------------------------------------------------------- #
# Details (tabs)
# --------------------------------------------------------------------------- #
st.subheader("3. Explore the details")
tab_colors, tab_shared, tab_hist, tab_lab, tab_export, tab_about = st.tabs([
    "\U0001F3A8 Dominant colors", "\U0001F50D What is shared", "\U0001F4CA Histograms",
    "\U0001F9EA Experiment Lab", "\u2B07\uFE0F Export", "\U0001F4D8 About",
])

with tab_colors:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<b style="color:{COLOR_A};font-size:1.1rem">Image A</b>',
                    unsafe_allow_html=True)
        st.markdown(avg_html(avg_a), unsafe_allow_html=True)
        st.markdown(palette_html(colors_a), unsafe_allow_html=True)
    with c2:
        st.markdown(f'<b style="color:{COLOR_B};font-size:1.1rem">Image B</b>',
                    unsafe_allow_html=True)
        st.markdown(avg_html(avg_b), unsafe_allow_html=True)
        st.markdown(palette_html(colors_b), unsafe_allow_html=True)
    st.caption("Each swatch is the real average color of the pixels in one color group, "
               "shown with the share of pixels it represents. Color names are rough labels.")
    st.markdown("##### What is each image made of?")
    show_fig(plot_composition(colors_a, colors_b))
    st.caption("Each bar is 100% of one image. Every block is one dominant color and its width "
               "is the share of pixels. Hatched grey = all the smaller colors not shown.")

with tab_shared:
    st.markdown("##### How is the score made?")
    show_fig(plot_overlap(inter_pct))
    st.caption(
        f"The green part ({inter_pct:.1f}%) is the histogram intersection: the share of each "
        "image's pixels that sit in color groups both images use. The rest is color that "
        "only appears in that image."
    )
    st.markdown("##### Color by color")
    show_fig(plot_butterfly(rows_top))
    total_overlap = float(np.minimum(fp_a, fp_b).sum())
    shown_overlap = sum(min(r["a"], r["b"]) for r in rows_top) / 100.0
    share = 100.0 * shown_overlap / total_overlap if total_overlap > 0 else 0.0
    st.caption(
        "Each row is one color. Bar length = share of that image's pixels. Rows where both "
        "bars are long are colors the images share; rows with one long bar are colors that "
        f"mostly belong to one image. These {len(rows_top)} colors account for {share:.0f}% "
        "of the overlap."
    )

with tab_hist:
    st.markdown(
        "A **histogram** counts pixels by one property. Each bar answers *'how many pixels "
        "fall in this range?'* and is colored like the range it represents."
    )
    labels = CHANNEL_LABELS[color_space]
    k = st.radio("Choose a channel", [0, 1, 2], format_func=lambda i: labels[i],
                 horizontal=True, key=f"chan_{color_space}")
    st.info(CHANNEL_HELP[(color_space, k)])
    show_fig(plot_channel(hist_a, hist_b, color_space, bins, k))
    if detailed:
        st.markdown("##### Technical: Image A and Image B on the same axes")
        show_fig(plot_channel_overlay(hist_a, hist_b, color_space, bins, k))
        st.markdown("##### Technical: the raw fingerprint (every bin)")
        show_fig(plot_fingerprints(fp_a, fp_b))
        st.caption("This is the actual list of numbers that gets compared, one bar per "
                   "histogram bin. The grey area is the overlap and equals the histogram "
                   "intersection. Most bins are near zero because most images use only a "
                   "few colors.")
    else:
        st.caption("Tick 'Show detailed histogram plots' in the sidebar for technical charts.")

with tab_lab:
    st.markdown(
        "The same two images compared under **every** supported setting. Each row uses one "
        "identical bin configuration for both images."
    )
    show_fig(plot_experiment(exp_rows))
    st.dataframe(pd.DataFrame(exp_rows), hide_index=True, width="stretch")
    st.markdown(
        "- **More bins** keep finer color detail, so scores usually **drop** (two similar "
        "colors may land in different bins).\n"
        "- **Fewer bins** merge nearby colors, so scores usually **rise**, but unrelated "
        "colors can be merged too.\n"
        "- **RGB vs HSV** cut color space differently, so the same pair can score "
        "differently."
    )

with tab_export:
    summary = {
        "image_a_filename": img_a.name,
        "image_a_width": img_a.width,
        "image_a_height": img_a.height,
        "image_b_filename": img_b.name,
        "image_b_width": img_b.width,
        "image_b_height": img_b.height,
        "color_space": color_space,
        "histogram_resolution_bins": resolution,
        "bin_configuration": f"{bins[0]}x{bins[1]}x{bins[2]}",
        "histogram_intersection_percent": round(inter_pct, 4),
        "bhattacharyya_distance": round(bhatt, 6),
    }
    fp_df = pd.DataFrame({
        "bin_index": np.arange(fp_a.size),
        "image_a_normalized_value": fp_a,
        "image_b_normalized_value": fp_b,
    })
    st.markdown("Download the numbers behind the analysis:")
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button("Fingerprints (CSV)",
                       fp_df.to_csv(index=False, float_format="%.10g"),
                       file_name="colorprint_fingerprints.csv", mime="text/csv",
                       width="stretch")
    e2.download_button("Summary (CSV)", pd.DataFrame([summary]).to_csv(index=False),
                       file_name="colorprint_summary.csv", mime="text/csv",
                       width="stretch")
    e3.download_button("Summary (JSON)", json.dumps(summary, indent=2),
                       file_name="colorprint_summary.json", mime="application/json",
                       width="stretch")
    e4.download_button("Report (TXT)", build_report(summary, lines, exp_rows, insights),
                       file_name="colorprint_report.txt", mime="text/plain",
                       width="stretch")
    st.markdown("Preview of the fingerprint file (first 10 rows):")
    st.dataframe(fp_df.head(10), hide_index=True, width="stretch")

with tab_about:
    st.markdown(ABOUT_MD)

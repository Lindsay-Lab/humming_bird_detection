#!/usr/bin/env python3
"""Generate documentation images from training data.

Run this script from the project root to regenerate image assets::

    python docs/scripts/generate_doc_images.py

Outputs are written to ``docs/assets/images/``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "interim" / "train_data"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "images"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_images(
    directory: Path,
    n: int = 16,
    min_brightness: float = 30.0,
) -> list[np.ndarray]:
    """Load up to *n* images from *directory*, skipping very dark shots."""
    paths = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.JPG"))
    images: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if float(img.mean()) < min_brightness:
            continue  # skip night / underexposed frames
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if len(images) >= n:
            break
    return images


def make_grid(
    images: list[np.ndarray],
    cols: int = 4,
    cell_size: int = 224,
    gap: int = 4,
) -> np.ndarray:
    """Arrange *images* into a uniform grid with a thin gap between cells."""
    rows = (len(images) + cols - 1) // cols
    h = rows * cell_size + (rows - 1) * gap
    w = cols * cell_size + (cols - 1) * gap
    grid = np.full((h, w, 3), 200, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y0 = r * (cell_size + gap)
        x0 = c * (cell_size + gap)
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[y0 : y0 + cell_size, x0 : x0 + cell_size] = resized
    return grid


def letterbox(
    image: np.ndarray, target: int = 224
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Letterbox *image* to a *target* × *target* square.

    Returns ``(padded, (x0, y0, new_w, new_h))`` where ``(x0, y0)`` is the
    top-left corner of the pasted image region.
    """
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h))
    mean_color = resized.mean(axis=(0, 1)).astype(int)
    pad = np.full((target, target, 3), mean_color, dtype=np.uint8)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    pad[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return pad, (x0, y0, new_w, new_h)


# ---------------------------------------------------------------------------
# Pipeline overview flowchart
# ---------------------------------------------------------------------------


def save_pipeline_overview() -> None:
    """Render a pipeline flowchart and save it as a PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")

    bg = "#f8faf8"
    dark_green = "#1b5e20"
    green = "#2e7d32"
    light_green = "#c8e6c9"
    arrow_c = "#555555"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # (cx, cy, title, subtitle, color)
    stages = [
        (4.5, 9.0, "Camera-trap JPEG", "raw field-camera image", dark_green),
        (
            4.5,
            7.4,
            "OCR Metadata Extraction",
            "date · time · temperature · camera ID",
            green,
        ),
        (
            4.5,
            5.8,
            "Bird Detection",
            "YOLO11n direct pass  +  SAHI tiled fallback",
            green,
        ),
        (
            4.5,
            4.2,
            "Letterbox & Resize",
            "aspect-ratio-preserving crop  →  224 × 224 px",
            green,
        ),
        (
            4.5,
            2.6,
            "Hummingbird Classification",
            "EfficientNetV2-S  →  probability ∈ [0, 1]",
            green,
        ),
        (
            4.5,
            1.0,
            "Observation CSV",
            "filename · date · time · bbox · confidence · label",
            dark_green,
        ),
    ]

    box_w, box_h = 5.8, 0.9
    gap = 0.45

    for cx, cy, title, subtitle, color in stages:
        x0, y0 = cx - box_w / 2, cy - box_h / 2
        box = mpatches.FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.06",
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            cx,
            cy + 0.14,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        ax.text(
            cx,
            cy - 0.20,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
            color=light_green,
            zorder=4,
        )

    arrow_props = dict(arrowstyle="-|>", color=arrow_c, lw=1.5)
    for i in range(len(stages) - 1):
        cx = stages[i][0]
        y_start = stages[i][1] - gap
        y_end = stages[i + 1][1] + gap
        ax.annotate(
            "",
            xy=(cx, y_end),
            xytext=(cx, y_start),
            arrowprops=arrow_props,
            zorder=2,
        )

    ax.set_title(
        "Hummingbird Detection Pipeline",
        fontsize=14,
        fontweight="bold",
        pad=12,
        color=dark_green,
    )

    out_path = OUTPUT_DIR / "pipeline_overview.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Example crop grids
# ---------------------------------------------------------------------------


def save_example_crops() -> None:
    """Create 4 × 4 example crop grids for hummingbird and other classes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        for cls, label in (("hummingbird", "Hummingbird"), ("other", "Other bird")):
            src = DATA_DIR / split / cls
            if not src.exists():
                continue
            imgs = load_images(src, n=16)
            if not imgs:
                continue
            grid = make_grid(imgs, cols=4)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(grid)
            ax.axis("off")
            ax.set_title(
                f"{label} crops — {split} set",
                fontsize=13,
                pad=8,
                color="#1b5e20",
                fontweight="bold",
            )
            out_path = OUTPUT_DIR / f"example_{cls}_crops_{split}.png"
            fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Hummingbird vs. other comparison strip
# ---------------------------------------------------------------------------


def save_classification_comparison() -> None:
    """Two-row strip: 8 hummingbird crops on top, 8 other-bird crops below."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hb = load_images(DATA_DIR / "train" / "hummingbird", n=8)
    other = load_images(DATA_DIR / "train" / "other", n=8)
    if not hb or not other:
        print("Skipping classification comparison — missing training data.")
        return

    n = min(len(hb), len(other), 8)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    fig.suptitle(
        "Classifier input examples",
        fontsize=13,
        fontweight="bold",
        color="#1b5e20",
        y=1.02,
    )

    row_labels = ["Hummingbird", "Other bird"]
    for row_idx, (imgs, label) in enumerate(zip([hb[:n], other[:n]], row_labels)):
        for col_idx, img in enumerate(imgs):
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(
            label, fontsize=10, rotation=90, labelpad=10, color="#333333"
        )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "classification_examples.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Letterboxing illustration
# ---------------------------------------------------------------------------


def save_letterbox_illustration() -> None:
    """Show the letterboxing step vs. naive squish resize."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imgs = load_images(DATA_DIR / "train" / "hummingbird", n=1)
    if not imgs:
        print(
            "Skipping letterbox illustration — no hummingbird training images found."
        )
        return

    img = imgs[0]
    h, w = img.shape[:2]

    # Simulate a tall narrow detection crop
    crop_w = max(w // 3, 50)
    x0 = (w - crop_w) // 2
    tall_crop = img[:, x0 : x0 + crop_w]

    padded, (px, py, pw, ph) = letterbox(tall_crop, target=224)
    squished = cv2.resize(tall_crop, (224, 224))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    fig.suptitle(
        "Preprocessing: aspect-ratio-preserving letterbox vs. squish",
        fontsize=12,
        fontweight="bold",
        color="#1b5e20",
    )

    axes[0].imshow(tall_crop)
    axes[0].set_title(
        f"Detection crop\n{tall_crop.shape[1]} × {tall_crop.shape[0]} px",
        fontsize=9,
    )
    axes[0].axis("off")

    axes[1].imshow(padded)
    rect = plt.Rectangle(
        (px, py),
        pw,
        ph,
        linewidth=1.5,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
    )
    axes[1].add_patch(rect)
    axes[1].set_title(
        "Letterboxed  224 × 224\n(padding = crop mean colour)",
        fontsize=9,
    )
    axes[1].axis("off")

    axes[2].imshow(squished)
    axes[2].set_title(
        "Squish resize  224 × 224\n— distorted, not used",
        fontsize=9,
        color="#b71c1c",
    )
    axes[2].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "letterbox_illustration.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Detection strategy diagram
# ---------------------------------------------------------------------------


def save_detection_strategy() -> None:
    """Illustrate the direct-pass + SAHI fallback detection strategy."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    bg = "#f8faf8"
    fig.patch.set_facecolor(bg)
    fig.suptitle(
        "Detection strategy: direct pass  +  SAHI tiled fallback",
        fontsize=13,
        fontweight="bold",
        color="#1b5e20",
        y=1.02,
    )

    # --- Left panel: direct full-image pass ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.set_title("Direct full-image pass", fontsize=11, color="#2e7d32")

    # Image rectangle
    img_rect = mpatches.FancyBboxPatch(
        (0.5, 1),
        9,
        5.5,
        boxstyle="round,pad=0.1",
        facecolor="#dcedc8",
        edgecolor="#558b2f",
        lw=1.5,
    )
    ax.add_patch(img_rect)
    ax.text(
        5,
        5.8,
        "Full image  (e.g. 4096 × 3072 px)",
        ha="center",
        fontsize=9,
        color="#33691e",
    )

    # Bird bounding box
    bird = mpatches.FancyBboxPatch(
        (3.5, 2.5),
        2.5,
        2.5,
        boxstyle="round,pad=0.05",
        facecolor="none",
        edgecolor="#e53935",
        lw=2.0,
    )
    ax.add_patch(bird)
    ax.text(
        4.75,
        4.85,
        "bird ✓",
        ha="center",
        fontsize=9,
        color="#e53935",
        fontweight="bold",
    )

    ax.text(
        5,
        0.5,
        "Fast (<0.5 s)  ·  works for larger / closer birds",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    # --- Right panel: SAHI tiled fallback ---
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 7)
    ax2.set_facecolor(bg)
    ax2.axis("off")
    ax2.set_title(
        "SAHI tiled fallback (small / distant birds)", fontsize=11, color="#2e7d32"
    )

    img_rect2 = mpatches.FancyBboxPatch(
        (0.5, 1),
        9,
        5.5,
        boxstyle="round,pad=0.1",
        facecolor="#dcedc8",
        edgecolor="#558b2f",
        lw=1.5,
    )
    ax2.add_patch(img_rect2)

    # Draw tile grid
    tile_xs = [0.5, 3.2, 5.9]
    tile_ys = [1.0, 2.9]
    tile_w, tile_h = 2.9, 2.7
    for ty in tile_ys:
        for tx in tile_xs:
            tile = mpatches.FancyBboxPatch(
                (tx, ty),
                tile_w,
                tile_h,
                boxstyle="square,pad=0",
                facecolor="none",
                edgecolor="#1565c0",
                lw=1.0,
                linestyle="--",
            )
            ax2.add_patch(tile)

    # Tiny bird
    tiny = mpatches.FancyBboxPatch(
        (3.4, 4.4),
        0.6,
        0.6,
        boxstyle="round,pad=0.02",
        facecolor="none",
        edgecolor="#e53935",
        lw=2.0,
    )
    ax2.add_patch(tiny)
    ax2.text(
        3.7, 5.1, "bird ✓", ha="center", fontsize=8, color="#e53935", fontweight="bold"
    )

    ax2.text(
        5,
        0.5,
        "Slower (~30 s CPU)  ·  recovers small / distant birds",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "detection_strategy.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Generating documentation images…")
    print(f"  Data dir : {DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Output   : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print()

    save_pipeline_overview()
    save_example_crops()
    save_classification_comparison()
    save_letterbox_illustration()
    save_detection_strategy()

    print("\nDone.")


if __name__ == "__main__":
    main()

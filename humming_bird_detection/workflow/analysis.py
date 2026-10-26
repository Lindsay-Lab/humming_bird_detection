"""Comprehensive analysis and reporting for detection pipeline results.

Produces summary statistics, per-camera/time breakdowns, confidence
distributions, plots, and JSON export — all from a pipeline output CSV.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CAMERA_DTYPE = pd.StringDtype()
_DATE_TIME_FORMAT = "%d %b %Y %I:%M %p"


def _classifier_certainty(df: pd.DataFrame) -> pd.Series:
    """Return the classifier's confidence in its *predicted* class.

    ``classification_confidence`` stores the sigmoid probability of the
    hummingbird class (p).  For records classified as "other" the classifier's
    certainty is ``1 - p``, not p — averaging raw p across both classes
    produces a meaningless mid-scale value (e.g. 0.5 for a perfectly certain
    classifier on a balanced sample).  This helper folds the probability so
    that 1.0 always means "maximally certain about the assigned label".
    """
    p = df["classification_confidence"]
    is_hb = df["is_hummingbird"].astype(bool)
    return pd.Series(np.where(is_hb, p, 1.0 - p), index=df.index)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def compute_summary(
    df: pd.DataFrame, total_images: int | None = None
) -> dict[str, Any]:
    """Return a flat dict of aggregate pipeline statistics.

    Parameters
    ----------
    df:
        DataFrame as produced by :func:`pipeline.process_directory`.  Must
        contain at minimum the columns ``is_hummingbird``,
        ``detection_confidence``, ``classification_confidence``, and
        ``is_duplicate`` (if post-processing was applied).
    total_images:
        Total number of JPEG files processed.  When None, this is read from
        ``df.attrs[\"total_images\"]`` if present; otherwise it is estimated
        from ``df[\"filename\"].nunique()``.

    Returns
    -------
    dict
        Keys: total_images, blank_images, total_detections,
        hummingbird_detections, other_detections, hummingbird_rate,
        mean_detection_conf, median_detection_conf,
        mean_classification_conf, median_classification_conf,
        duplicate_detections, duplicate_rate, suspected_duplicate_detections,
        suspected_duplicate_rate.  Fields that depend on optional columns are
        omitted when those columns are absent.
    """
    if df.empty:
        return {
            "total_images": total_images or 0,
            "blank_images": total_images or 0,
            "total_detections": 0,
            "hummingbird_detections": 0,
            "other_detections": 0,
            "hummingbird_rate": 0.0,
            "mean_detection_conf": float("nan"),
            "median_detection_conf": float("nan"),
            "mean_classification_conf": float("nan"),
            "median_classification_conf": float("nan"),
        }

    n_total = len(df)
    n_hb = int(df["is_hummingbird"].sum())
    n_other = n_total - n_hb

    # Resolve total image count.
    resolved_total = total_images
    if resolved_total is None:
        resolved_total = df.attrs.get("total_images")
    if resolved_total is None and "filename" in df.columns:
        resolved_total = int(df["filename"].nunique())

    blank_images = df.attrs.get("blank_images")
    if blank_images is None and resolved_total is not None:
        n_photos_with_birds = (
            int(df["filename"].nunique()) if "filename" in df.columns else n_total
        )
        blank_images = resolved_total - n_photos_with_birds

    summary: dict[str, Any] = {
        "total_images": resolved_total,
        "blank_images": blank_images,
        "total_detections": n_total,
        "hummingbird_detections": n_hb,
        "other_detections": n_other,
        "hummingbird_rate": round(n_hb / n_total, 4) if n_total else 0.0,
        "mean_detection_conf": round(float(df["detection_confidence"].mean()), 4),
        "median_detection_conf": round(float(df["detection_confidence"].median()), 4),
        "mean_classification_conf": round(
            float(df["classification_confidence"].mean()), 4
        ),
        "median_classification_conf": round(
            float(df["classification_confidence"].median()), 4
        ),
    }

    # Classifier certainty: confidence in the *predicted* class (see
    # _classifier_certainty).  Raw classification_confidence is the
    # hummingbird-class probability, so it is also reported split by class.
    certainty = _classifier_certainty(df)
    summary["mean_classifier_certainty"] = round(float(certainty.mean()), 4)
    summary["median_classifier_certainty"] = round(float(certainty.median()), 4)

    hb_mask = df["is_hummingbird"].astype(bool)
    if bool(hb_mask.any()):
        summary["mean_hummingbird_conf"] = round(
            float(df.loc[hb_mask, "classification_confidence"].mean()), 4
        )
    if bool((~hb_mask).any()):
        summary["mean_other_conf"] = round(
            float(df.loc[~hb_mask, "classification_confidence"].mean()), 4
        )

    if "is_duplicate" in df.columns:
        n_dup = int(df["is_duplicate"].sum())
        summary["duplicate_detections"] = n_dup
        summary["duplicate_rate"] = round(n_dup / n_total, 4) if n_total else 0.0

    if "is_suspected_duplicate" in df.columns:
        n_sus = int(df["is_suspected_duplicate"].sum())
        summary["suspected_duplicate_detections"] = n_sus
        summary["suspected_duplicate_rate"] = round(n_sus / n_total, 4) if n_total else 0.0

    return summary


def compute_per_camera(
    df: pd.DataFrame, total_images: int | None = None
) -> pd.DataFrame:
    """Return a per-camera breakdown DataFrame.

    Columns: camera, total_detections, hummingbird_detections,
    hummingbird_rate, mean_detection_conf, mean_classification_conf,
    blank_images, efficiency (bird photos / total photos), photos_with_birds.

    When *total_images* is None, the total image count is resolved from
    ``df.attrs``.
    """
    if df.empty or "camera" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("camera", dropna=False)
    hb_sum = grouped["is_hummingbird"].sum().astype(int)
    total = grouped.size()

    result = pd.DataFrame(
        {
            "total_detections": total,
            "hummingbird_detections": hb_sum,
        }
    )
    result["hummingbird_rate"] = (
        (result["hummingbird_detections"] / total).round(4)
        if len(result) > 0
        else 0.0
    )
    result["mean_detection_conf"] = grouped["detection_confidence"].mean().round(4)
    result["mean_classification_conf"] = (
        grouped["classification_confidence"].mean().round(4)
    )
    # Certainty in the *predicted* class — raw classification_confidence is the
    # hummingbird probability, so its per-camera mean is misleading when a
    # camera saw mostly "other" birds (a perfect classifier would score ~0).
    certainty_df = df.assign(_certainty=_classifier_certainty(df))
    result["mean_classifier_certainty"] = (
        certainty_df.groupby("camera", dropna=False)["_certainty"].mean().round(4)
    )

    # Photos with birds per camera.
    photos_per_cam = df.groupby("camera")["filename"].nunique()
    result["photos_with_birds"] = photos_per_cam.astype(int)

    # Blank stats — only meaningful when total_images is known.
    resolved_total = total_images
    if resolved_total is None:
        resolved_total = df.attrs.get("total_images")
    total_photos = resolved_total if resolved_total is not None else 0

    # Distribute blanks proportionally when total image count is global.
    # If per-camera blank info is stored in attrs, use it.
    blank_map = df.attrs.get("blank_per_camera")
    if blank_map is not None:
        result["blank_images"] = result.index.map(
            lambda c: blank_map.get(c, blank_map.get("Unknown", 0))
        ).fillna(0).astype(int)
    elif total_photos > 0:
        # Fallback: estimate from global blank count.
        global_blanks = df.attrs.get("blank_images", 0)
        if global_blanks > 0:
            n_cameras = len(result)
            blanks_each = global_blanks // n_cameras
            remainder = global_blanks % n_cameras
            blanks_list = [blanks_each + 1] * remainder + [blanks_each] * (
                n_cameras - remainder
            )
            result["blank_images"] = blanks_list[:n_cameras]
        else:
            result["blank_images"] = 0
    else:
        result["blank_images"] = 0

    total_per_cam = result["photos_with_birds"] + result["blank_images"]
    result["efficiency"] = (
        (result["photos_with_birds"] / total_per_cam.replace(0, pd.NA)).round(4)
    )

    # Fill NaN camera with "Unknown" for display clarity
    result.index = result.index.fillna("Unknown").astype(str)
    return result.sort_values("total_detections", ascending=False)


def compute_suspected_duplicate_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return summary statistics about suspected (composition-based) duplicates.

    Requires the ``is_suspected_duplicate`` column produced by
    :func:`postprocessing.flag_suspected_duplicates`.
    """
    if df.empty or "is_suspected_duplicate" not in df.columns:
        return {}

    n_total_photos = int(df["filename"].nunique())
    photo_flags = df.drop_duplicates("filename").set_index("filename")[
        "is_suspected_duplicate"
    ]
    n_sus_photos = int(photo_flags.sum())
    n_kept_photos = n_total_photos - n_sus_photos

    n_sus_birds = int(df["is_suspected_duplicate"].sum())
    n_total_birds = len(df)

    return {
        "total_photos_with_birds": n_total_photos,
        "kept_photos": n_kept_photos,
        "suspected_duplicate_photos": n_sus_photos,
        "photo_duplicate_rate": (
            round(n_sus_photos / n_total_photos, 4) if n_total_photos else 0.0
        ),
        "total_bird_records": n_total_birds,
        "suspected_duplicate_bird_records": n_sus_birds,
        "bird_record_duplicate_rate": (
            round(n_sus_birds / n_total_birds, 4) if n_total_birds else 0.0
        ),
    }


def compute_per_camera_hourly(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return a dict mapping camera name to its hourly-activity DataFrame.

    Each per-camera DataFrame has columns ``hour``, ``hummingbird``, ``other``,
    ``total`` as in :func:`compute_hourly_activity`.
    """
    result: dict[str, pd.DataFrame] = {}
    if df.empty or "camera" not in df.columns:
        return result

    dt = _parse_datetime_column(df)
    if dt is None:
        return result

    df = df.copy()
    df["_hour"] = dt.dt.hour

    cameras = df["camera"].fillna("Unknown").unique()
    for cam in cameras:
        cam_df = df[df["camera"].fillna("Unknown") == cam]
        hb = cam_df[cam_df["is_hummingbird"] == 1].groupby("_hour").size()
        ot = cam_df[cam_df["is_hummingbird"] == 0].groupby("_hour").size()
        cam_result = pd.DataFrame(index=range(24))
        cam_result.index.name = "hour"
        cam_result["hummingbird"] = hb.reindex(range(24), fill_value=0).astype(int)
        cam_result["other"] = ot.reindex(range(24), fill_value=0).astype(int)
        cam_result["total"] = cam_result["hummingbird"] + cam_result["other"]
        result[str(cam)] = cam_result.reset_index()

    return result


def compute_hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Return detections binned by hour of day (0–23).

    Requires ``date`` and ``time`` columns.  Rows with unparseable timestamps
    are excluded.
    """
    if df.empty:
        return pd.DataFrame(columns=["hour", "total", "hummingbird", "other"])

    dt = _parse_datetime_column(df)
    if dt is None or dt.dropna().empty:
        return pd.DataFrame(columns=["hour", "total", "hummingbird", "other"])

    df = df.assign(_hour=dt.dt.hour)
    hb = df[df["is_hummingbird"] == 1].groupby("_hour").size()
    ot = df[df["is_hummingbird"] == 0].groupby("_hour").size()

    result = pd.DataFrame(index=range(24))
    result.index.name = "hour"
    result["hummingbird"] = hb.reindex(range(24), fill_value=0).astype(int)
    result["other"] = ot.reindex(range(24), fill_value=0).astype(int)
    result["total"] = result["hummingbird"] + result["other"]
    return result.reset_index()


def compute_daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Return detections aggregated by calendar date.

    Requires ``date`` and ``time`` columns.  Rows with unparseable timestamps
    are excluded.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "total", "hummingbird", "other"])

    dt = _parse_datetime_column(df)
    if dt is None or dt.dropna().empty:
        return pd.DataFrame(columns=["date", "total", "hummingbird", "other"])

    df = df.assign(_date=dt.dt.date)
    hb = df[df["is_hummingbird"] == 1].groupby("_date").size()
    ot = df[df["is_hummingbird"] == 0].groupby("_date").size()

    all_dates = sorted(set(hb.index) | set(ot.index))
    result = pd.DataFrame(index=pd.Index(all_dates, name="date"))
    result["hummingbird"] = hb.reindex(all_dates, fill_value=0).astype(int)
    result["other"] = ot.reindex(all_dates, fill_value=0).astype(int)
    result["total"] = result["hummingbird"] + result["other"]
    return result.reset_index()


def compute_confidence_correlation(df: pd.DataFrame) -> dict[str, float | None]:
    """Return Pearson correlation between detection confidence and classifier certainty.

    Classifier certainty (confidence in the predicted class) is used instead of
    raw ``classification_confidence`` (hummingbird-class probability) because
    the raw value is inversely coded for "other" records — correlating it with
    detection confidence across mixed classes mostly measures class balance,
    not any detector↔classifier relationship.
    """
    if df.empty or len(df) < 2:
        return {"pearson_r": None, "p_value": None}
    from scipy.stats import pearsonr

    certainty = _classifier_certainty(df)
    if certainty.nunique() < 2 or df["detection_confidence"].nunique() < 2:
        return {"pearson_r": None, "p_value": None}
    r, p = pearsonr(df["detection_confidence"], certainty)
    return {"pearson_r": round(float(r), 4), "p_value": round(float(p), 6)}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(df: pd.DataFrame, cfg: dict | None = None) -> str:
    """Produce a human-readable terminal report string.

    Parameters
    ----------
    df:
        Pipeline output DataFrame (post-processed).
    cfg:
        Optional pipeline config dict; used to annotate thresholds.

    Returns
    -------
    str
        Multi-line plain-text report suitable for `click.echo`.
    """
    summary = compute_summary(df)
    per_camera = compute_per_camera(df)
    hourly = compute_hourly_activity(df)
    daily = compute_daily_activity(df)
    sus_dup = compute_suspected_duplicate_summary(df)

    lines: list[str] = []
    _sep = "─" * 60

    lines.append("")
    lines.append("╔" + "═" * 58 + "╗")
    lines.append("║" + "  HUMMINGBIRD DETECTION — ANALYSIS REPORT".ljust(58) + "║")
    lines.append("╚" + "═" * 58 + "╝")
    lines.append("")

    # --- Summary ---
    lines.append("── Overview ──")
    total_img = summary.get("total_images")
    blank_img = summary.get("blank_images")
    if total_img is not None and total_img > 0:
        lines.append(f"  Total images processed  : {total_img:>8}")
        if blank_img is not None:
            lines.append(f"  Blank images (no birds) : {blank_img:>8}")
            bird_img = total_img - (blank_img or 0)
            eff = bird_img / total_img if total_img else 0
            lines.append(f"  Images with birds       : {bird_img:>8} ({eff:.1%})")
    lines.append(f"  Total bird detections   : {summary['total_detections']:>8}")
    lines.append(f"  Hummingbird detections  : {summary['hummingbird_detections']:>8}")
    lines.append(f"  Other bird detections   : {summary['other_detections']:>8}")
    lines.append(
        f"  Hummingbird rate        : {summary['hummingbird_rate']:>8.2%}"
    )
    lines.append("")

    # --- Duplicates ---
    has_dup = "duplicate_detections" in summary
    has_sus = sus_dup
    if has_dup or has_sus:
        lines.append("── Duplicates ──")
    if has_dup:
        lines.append(
            f"  Motion-event duplicates : {summary['duplicate_detections']:>8}"
            f"  ({summary['duplicate_rate']:.2%})"
        )
    if has_sus:
        lines.append("  ── Composition-based suspected duplicates (ceil(N/3)) ──")
        lines.append(
            f"  Photos with birds       : {sus_dup['total_photos_with_birds']:>8}"
        )
        lines.append(
            f"  Kept photos             : {sus_dup['kept_photos']:>8}"
        )
        lines.append(
            f"  Suspected duplicates    : {sus_dup['suspected_duplicate_photos']:>8}"
            f"  ({sus_dup['photo_duplicate_rate']:.2%})"
        )
        lines.append(
            f"  Bird records removed    : {sus_dup['suspected_duplicate_bird_records']:>8}"
            f"  ({sus_dup['bird_record_duplicate_rate']:.2%})"
        )
        lines.append("")

    # --- Confidence ---
    lines.append("── Confidence ──")
    lines.append(
        "  Detection confidence    :"
        f"  mean {summary['mean_detection_conf']:.3f}"
        f"  median {summary['median_detection_conf']:.3f}"
    )
    if "mean_classifier_certainty" in summary:
        lines.append(
            "  Classifier certainty    :"
            f"  mean {summary['mean_classifier_certainty']:.3f}"
            f"  median {summary['median_classifier_certainty']:.3f}"
        )
    if "mean_hummingbird_conf" in summary:
        lines.append(
            f"  Mean P(hummingbird)     :  HB records {summary['mean_hummingbird_conf']:.3f}"
            + (
                f"  other records {summary['mean_other_conf']:.3f}"
                if "mean_other_conf" in summary
                else ""
            )
        )
    if cfg:
        det_thresh = cfg.get("detector", {}).get("confidence_threshold")
        cls_thresh = cfg.get("classifier", {}).get("confidence_threshold")
        if det_thresh is not None:
            lines.append(f"  Detector threshold       : {det_thresh:>8}")
        if cls_thresh is not None:
            lines.append(f"  Classifier threshold     : {cls_thresh:>8}")

    corr = compute_confidence_correlation(df)
    if corr["pearson_r"] is not None:
        lines.append(
            f"  Det↔Certainty Pearson r : {corr['pearson_r']:>8.3f}"
            f"  (p={corr['p_value']:.4f})"
        )
    lines.append("")

    # --- Per camera ---
    if not per_camera.empty:
        lines.append("── Per Camera ──")
        has_eff = "efficiency" in per_camera.columns
        header = (
            f"  {'Camera':<16s} {'Total':>6s} {'HB':>6s}"
            f"  {'HB rate':>8s}"
            + (f" {'Birds':>6s} {'Blanks':>7s} {'Eff.':>7s}" if has_eff else "")
            + f" {'DetConf':>8s} {'Certainty':>10s}"
        )
        lines.append(header)
        lines.append("  " + "─" * (len(header) - 2))
        for cam, row in per_camera.iterrows():
            base = (
                f"  {cam:<16s} {int(row['total_detections']):>6d}"
                f" {int(row['hummingbird_detections']):>6d}"
                f"  {row['hummingbird_rate']:>7.2%}"
            )
            if has_eff:
                base += (
                    f" {int(row.get('photos_with_birds', 0)):>6d}"
                    f" {int(row.get('blank_images', 0)):>7d}"
                    f" {row.get('efficiency', 0):>6.1%} "
                )
            certainty = row.get(
                "mean_classifier_certainty", row["mean_classification_conf"]
            )
            base += (
                f" {row['mean_detection_conf']:>8.3f}"
                f" {certainty:>10.3f}"
            )
            lines.append(base)
        lines.append(
            "  (Certainty = classifier confidence in its predicted class;"
            " 1.0 = fully certain)"
        )
        lines.append("")

    # --- Hourly activity ---
    if not hourly.empty and hourly["total"].sum() > 0:
        lines.append("── Hourly Activity ──")
        _append_bar_chart(lines, hourly, "hour", bar_width=30)
        lines.append("")

    # --- Daily activity ---
    if not daily.empty and daily["total"].sum() > 0:
        lines.append("── Daily Activity ──")
        _append_bar_chart(lines, daily, "date", bar_width=40)
        lines.append("")

    lines.append(_sep)
    return "\n".join(lines)


def _append_bar_chart(
    lines: list[str],
    df: pd.DataFrame,
    label_col: str,
    bar_width: int = 30,
) -> None:
    """Append ASCII bar chart rows to *lines*."""
    max_total = int(df["total"].max())
    if max_total == 0:
        return
    for _, row in df.iterrows():
        label = str(row[label_col])
        total = int(row["total"])
        hb = int(row["hummingbird"])
        ot = int(row["other"])
        bar_len = int(round(total / max_total * bar_width)) if max_total else 0
        hb_len = int(round(hb / max_total * bar_width)) if max_total and hb else 0
        ot_len = bar_len - hb_len
        bar = ("█" * hb_len + "░" * ot_len).ljust(bar_width)
        lines.append(
            f"  {label:<12s} {bar} {total:>4d}  (HB:{hb} other:{ot})"
        )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_summary_json(
    df: pd.DataFrame,
    output_path: str | Path,
    cfg: dict | None = None,
) -> Path:
    """Write a machine-readable JSON summary to *output_path*.

    The JSON bundle includes aggregate statistics, per-camera breakdown,
    hourly/daily activity, confidence correlation, suspected-duplicate
    summary, and config metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = compute_summary(df)
    per_camera = compute_per_camera(df)
    hourly = compute_hourly_activity(df)
    daily = compute_daily_activity(df)
    corr = compute_confidence_correlation(df)
    sus_dup = compute_suspected_duplicate_summary(df)

    # Convert DataFrames to nested dicts for clean JSON serialization.
    def _df_to_records(d: pd.DataFrame, orient: str = "index") -> Any:
        if d.empty:
            return []
        result = d.to_dict(orient=orient)
        # Ensure all keys are JSON-serializable (datetime/date → str).
        if isinstance(result, dict):
            result = {str(k): v for k, v in result.items()}
        return result

    bundle: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "per_camera": (
            _df_to_records(per_camera) if not per_camera.empty else {}
        ),
        "suspected_duplicates": sus_dup or {},
        "hourly_activity": _df_to_records(
            hourly.set_index("hour") if not hourly.empty else hourly,
            orient="index",
        ),
        "daily_activity": _df_to_records(
            daily.set_index("date") if not daily.empty else daily,
            orient="index",
        ),
        "confidence_correlation": corr,
        "config": cfg or {},
    }

    with output_path.open("w") as fh:
        json.dump(bundle, fh, indent=2, default=str)
    logger.info("JSON summary written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def generate_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
    cfg: dict | None = None,
) -> list[Path]:
    """Generate and save publication-ready analysis plots.

    Produces:
    - ``confidence_histogram.png`` — side-by-side detection & classification
    - ``hourly_activity.png`` — stacked bar of hummingbird vs other by hour
    - ``per_camera.png`` — grouped bar chart per camera
    - ``daily_timeline.png`` — stacked bar timeline by date

    Parameters
    ----------
    df:
        Pipeline output DataFrame.
    output_dir:
        Directory to write PNG files.
    cfg:
        Optional pipeline config for threshold annotations.

    Returns
    -------
    list[Path]
        Absolute paths to the generated plot files.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    saved: list[Path] = []

    # --- 1. Confidence histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_confidence_hist(axes[0], df, "detection_confidence", "Detection Confidence")
    _plot_confidence_hist(
        axes[1], df, "classification_confidence", "Classification Confidence"
    )
    fig.suptitle("Confidence Distributions", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "confidence_histogram.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- 2. Hourly activity ---
    hourly = compute_hourly_activity(df)
    if not hourly.empty and hourly["total"].sum() > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        x = hourly["hour"]
        ax.bar(x, hourly["hummingbird"], label="Hummingbird", color="#2ca02c")
        ax.bar(
            x,
            hourly["other"],
            bottom=hourly["hummingbird"],
            label="Other bird",
            color="#d62728",
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Detections")
        ax.set_title("Hourly Activity")
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        fig.tight_layout()
        path = output_dir / "hourly_activity.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # --- 3. Per camera ---
    per_camera = compute_per_camera(df)
    if not per_camera.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        cameras = per_camera.index.tolist()
        x = np.arange(len(cameras))
        w = 0.35
        ax.bar(x - w / 2, per_camera["total_detections"], w, label="Total", color="#1f77b4")
        ax.bar(
            x + w / 2,
            per_camera["hummingbird_detections"],
            w,
            label="Hummingbird",
            color="#2ca02c",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(cameras, rotation=30, ha="right")
        ax.set_ylabel("Detections")
        ax.set_title("Detections per Camera")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "per_camera.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # --- 4. Daily timeline ---
    daily = compute_daily_activity(df)
    if not daily.empty and daily["total"].sum() > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        dates = [str(d) for d in daily["date"]]
        ax.bar(dates, daily["hummingbird"], label="Hummingbird", color="#2ca02c")
        ax.bar(
            dates,
            daily["other"],
            bottom=daily["hummingbird"],
            label="Other bird",
            color="#d62728",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Detections")
        ax.set_title("Daily Activity Timeline")
        ax.legend()
        fig.autofmt_xdate(rotation=45, ha="right")
        fig.tight_layout()
        path = output_dir / "daily_timeline.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    logger.info("Saved %d plot(s) to %s", len(saved), output_dir)
    return saved


def _plot_confidence_hist(
    ax: "plt.Axes", df: pd.DataFrame, column: str, title: str
) -> None:
    """Plot a histogram of confidence values, split by hummingbird/other."""
    hb = df[df["is_hummingbird"] == 1][column]
    ot = df[df["is_hummingbird"] == 0][column]

    bins = np.linspace(0, 1, 26)
    ax.hist(
        [ot, hb],
        bins=bins,
        label=["Other bird", "Hummingbird"],
        color=["#d62728", "#2ca02c"],
        alpha=0.8,
        edgecolor="white",
    )
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize="small")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_datetime_column(df: pd.DataFrame) -> pd.Series | None:
    """Parse date + time columns into a datetime Series; return None on failure."""
    if "date" not in df.columns or "time" not in df.columns:
        return None
    # Work on the first-row-per-filename view to match dedup logic
    first_rows = df.drop_duplicates(subset="filename")[
        ["filename", "date", "time"]
    ].copy()
    first_rows["_dt"] = pd.to_datetime(
        first_rows["date"] + " " + first_rows["time"],
        format=_DATE_TIME_FORMAT,
        errors="coerce",
    )
    dt_map = first_rows.set_index("filename")["_dt"]
    return df["filename"].map(dt_map)

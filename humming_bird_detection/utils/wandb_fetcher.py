"""Fetch WandB recorded training data — local cache first, then API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_WANDB_DIR = Path("wandb")
_HISTORY_FILE = "wandb-history.jsonl"


# ── history (training metrics per epoch/step) ──────────────────────────────


def _local_from_run_object(wandb_run) -> pd.DataFrame | None:
    try:
        history = wandb_run.history(pandas=True)
        if isinstance(history, pd.DataFrame) and not history.empty:
            logger.info("Fetched %d rows from live wandb Run history.", len(history))
            return history
    except Exception:
        logger.debug("Could not fetch history from wandb Run object.", exc_info=True)
    return None


def _local_from_cache_dir(wandb_dir: Path, run_id: str) -> pd.DataFrame | None:
    if not wandb_dir.is_dir():
        return None
    for run_dir in wandb_dir.iterdir():
        if run_id not in run_dir.name or not run_dir.is_dir():
            continue
        history_path = run_dir / "files" / _HISTORY_FILE
        if history_path.exists():
            try:
                rows = []
                with open(history_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        record.pop("_timestamp", None)
                        record.pop("_runtime", None)
                        record.pop("_step", None)
                        rows.append(record)
                if rows:
                    df = pd.DataFrame(rows)
                    logger.info(
                        "Fetched %d rows from local wandb cache: %s",
                        len(df),
                        history_path,
                    )
                    return df
            except Exception:
                logger.debug(
                    "Could not read local wandb history from %s",
                    history_path,
                    exc_info=True,
                )
                return None
        for candidate in run_dir.glob("*.wandb"):
            try:
                history = pd.read_csv(candidate)
                if not history.empty:
                    logger.info(
                        "Fetched %d rows from local .wandb file: %s",
                        len(history),
                        candidate,
                    )
                    return history
            except Exception:
                pass
    return None


def _history_from_api(
    run_id: str,
    project: str,
    entity: str | None = None,
    num_samples: int = 500,
) -> pd.DataFrame | None:
    import wandb  # noqa: PLC0415

    try:
        api = wandb.Api()
        path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
        run = api.run(path)
        history = run.history(samples=num_samples, pandas=True)
        if isinstance(history, pd.DataFrame) and not history.empty:
            logger.info(
                "Fetched %d rows from wandb API for run %s.", len(history), run_id
            )
            return history
    except Exception:
        logger.debug(
            "Could not fetch history from wandb API for run %s.", run_id, exc_info=True
        )
    return None


def fetch_wandb_history(
    wandb_run: Any = None,
    run_id: str | None = None,
    project: str | None = None,
    entity: str | None = None,
    wandb_cache_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Fetch recorded training history (per-epoch/step metrics) from WandB.

    Lookup order (returns the first successful result):
    1. Live ``wandb_run`` object (``.history()``).
    2. Local wandb cache directory.
    3. WandB public API.

    Returns ``None`` if no data could be retrieved from any source.
    """
    if wandb_run is not None:
        result = _local_from_run_object(wandb_run)
        if result is not None:
            return result
        if run_id is None:
            try:
                run_id = wandb_run.id
            except Exception:
                pass
        if project is None:
            try:
                project = wandb_run.project
            except Exception:
                pass
        if entity is None:
            try:
                entity = wandb_run.entity
            except Exception:
                pass

    cache_dir = wandb_cache_dir or _WANDB_DIR
    if run_id:
        result = _local_from_cache_dir(cache_dir, run_id)
        if result is not None:
            return result

    if run_id and project:
        result = _history_from_api(run_id, project, entity=entity)
        if result is not None:
            return result

    logger.warning("Could not retrieve wandb history from any source.")
    return None


# ── system metrics (GPU, CPU, memory, disk, network over time) ─────────────


def fetch_wandb_system_metrics(
    wandb_run: Any = None,
    run_id: str | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> pd.DataFrame | None:
    """Fetch system metrics (GPU, CPU, memory, disk, network) recorded during training.

    Uses the live ``wandb_run`` if provided, otherwise the wandb API.
    Returns ``None`` if no system metrics were recorded.
    """
    import wandb  # noqa: PLC0415

    try:
        if wandb_run is not None:
            sys_df = wandb_run.history(stream="system", pandas=True)
            if isinstance(sys_df, pd.DataFrame) and not sys_df.empty:
                logger.info(
                    "Fetched %d system metric rows from live wandb Run.", len(sys_df)
                )
                return sys_df
            # Fall through to API

        if run_id and project:
            api = wandb.Api()
            path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
            run = api.run(path)
            sys_df = run.history(stream="system", pandas=True)
            if isinstance(sys_df, pd.DataFrame) and not sys_df.empty:
                logger.info(
                    "Fetched %d system metric rows from wandb API for run %s.",
                    len(sys_df),
                    run_id,
                )
                return sys_df
    except Exception:
        logger.debug(
            "Could not fetch system metrics for run %s.", run_id, exc_info=True
        )
    return None


# ── metadata (compute resources, environment) ──────────────────────────────


def fetch_wandb_metadata(
    wandb_run: Any = None,
    run_id: str | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> dict | None:
    """Fetch run metadata: OS, CPU, GPU, memory, SLURM info, git commit, etc.

    Uses the live ``wandb_run`` if provided (extracts from ``.config`` and
    ``wandb-metadata.json``), otherwise the wandb API.
    """
    import wandb  # noqa: PLC0415

    try:
        if wandb_run is not None:
            metadata = {}
            # Extract info from the live run object
            for attr in ("id", "name", "project", "entity", "url", "group", "tags"):
                try:
                    val = getattr(wandb_run, attr, None)
                    if val is not None:
                        metadata[attr] = val
                except Exception:
                    pass
            # Try to get the metadata JSON file
            try:
                meta_file = wandb_run.file("wandb-metadata.json")
                if meta_file is not None:
                    raw_text = _read_downloaded(meta_file.download(replace=True))
                    if raw_text.strip():
                        metadata["_metadata"] = json.loads(raw_text)
            except Exception:
                logger.debug("Could not read wandb-metadata.json from run object.")
            if metadata:
                return metadata

        if run_id and project:
            api = wandb.Api()
            path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
            run = api.run(path)

            metadata = {
                "run_id": run.id,
                "name": run.name,
                "project": run.project,
                "entity": run.entity,
                "url": run.url,
                "state": run.state,
                "tags": list(run.tags) if run.tags else [],
                "created_at": str(run.created_at) if run.created_at else None,
            }

            # Fetch wandb-metadata.json from the run
            try:
                meta_file = run.file("wandb-metadata.json")
                raw_text = _read_downloaded(meta_file.download(replace=True))
                if raw_text.strip():
                    metadata["_metadata"] = json.loads(raw_text)
            except Exception:
                logger.debug(
                    "Could not read wandb-metadata.json from API for run %s.",
                    run_id,
                )

            logger.info("Fetched metadata for run %s.", run_id)
            return metadata
    except Exception:
        logger.debug("Could not fetch metadata for run %s.", run_id, exc_info=True)
    return None


# ── summary (final aggregate metrics) ──────────────────────────────────────


def fetch_wandb_summary(
    wandb_run: Any = None,
    run_id: str | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> dict | None:
    """Fetch run summary with final aggregate metrics.

    Uses the live ``wandb_run`` if provided, otherwise the wandb API.
    """
    import wandb  # noqa: PLC0415

    try:
        if wandb_run is not None:
            summary = dict(wandb_run.summary)
            if summary:
                logger.info("Fetched summary from live wandb Run.")
                return {k: v for k, v in summary.items() if not k.startswith("_")}

        if run_id and project:
            api = wandb.Api()
            path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
            run = api.run(path)
            summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
            if summary:
                logger.info("Fetched summary from wandb API for run %s.", run_id)
                return summary
    except Exception:
        logger.debug("Could not fetch summary for run %s.", run_id, exc_info=True)
    return None


# ── output log ─────────────────────────────────────────────────────────────


def _read_downloaded(raw: Any) -> str:
    """Normalise wandb ``File.download()`` return value to a string.

    ``.download(replace=True)`` may return a ``TextIOWrapper`` (text logs),
    ``BytesIO`` (binary), a path string, or ``None``.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        # Could be an already-read string, or a file path
        try:
            return Path(raw).read_text()
        except (OSError, UnicodeDecodeError):
            return raw
    if hasattr(raw, "read"):
        content = raw.read()
        return content if isinstance(content, str) else content.decode()
    return str(raw)


def fetch_wandb_output_log(
    wandb_run: Any = None,
    run_id: str | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> str | None:
    """Fetch the captured stdout/stderr output log from the run.

    Uses the live ``wandb_run`` if provided, otherwise the wandb API.
    """
    import wandb  # noqa: PLC0415

    source = "live wandb Run" if wandb_run is not None else f"wandb API ({run_id})"

    try:
        if wandb_run is not None:
            try:
                log_file = wandb_run.file("output.log")
                if log_file is not None:
                    text = _read_downloaded(log_file.download(replace=True))
                    if text.strip():
                        logger.info("Fetched output log from %s.", source)
                        return text
            except Exception:
                pass

        if run_id and project:
            api = wandb.Api()
            path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
            run = api.run(path)
            log_file = run.file("output.log")
            text = _read_downloaded(log_file.download(replace=True))
            if text.strip():
                logger.info("Fetched output log from %s.", source)
                return text
    except Exception:
        logger.debug("Could not fetch output log for run %s.", run_id, exc_info=True)
    return None


# ── save helpers ───────────────────────────────────────────────────────────


def _save_csv(df: pd.DataFrame, output_dir: Path, base_name: str, suffix: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{base_name}_{suffix}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved %s (%d rows)", path, len(df))
    return path


def _save_json(data: dict, output_dir: Path, base_name: str, suffix: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{base_name}_{suffix}.json"

    def _default(o: Any) -> str:
        return str(o)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)
    logger.info("Saved %s", path)
    return path


# ── convenience: fetch and save all wandb data ──────────────────────────────


def _merge_into_metadata(
    base_name: str,
    out_dir: Path,
    wandb_metadata: dict | None,
    wandb_summary: dict | None,
) -> Path | None:
    """Merge wandb metadata and summary into the existing ``_metadata.json`` file.

    The file is expected to have been created by the training script with a
    ``"wandb"`` key already present.  This function adds ``wandb.metadata``
    and ``wandb.summary`` sub-keys and writes the file back in-place.

    Returns the path to the updated file, or ``None`` if no metadata file
    existed and there was no wandb data to write.
    """
    meta_path = out_dir / f"{base_name}_metadata.json"

    if meta_path.exists():
        with open(meta_path) as f:
            config = json.load(f)
    elif wandb_metadata or wandb_summary:
        config = {}
    else:
        return None

    wandb_section = config.setdefault("wandb", {})

    if wandb_metadata is not None:
        wandb_section["metadata"] = wandb_metadata
    if wandb_summary is not None:
        wandb_section["summary"] = wandb_summary

    def _default(o: Any) -> str:
        return str(o)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(config, f, indent=2, default=_default)
    logger.info("Updated metadata file: %s", meta_path)
    return meta_path


def save_all_wandb_data(
    wandb_run: Any,
    output_path: str | Path,
) -> dict[str, Path | None]:
    """Fetch all available WandB data for a training run and save alongside output_path.

    Fetches: model metrics (CSV), system metrics (CSV), output log (TXT), and
    merges wandb metadata + summary into the existing ``_metadata.json``
    file (expected to have been created by the training script).

    Args:
        wandb_run: A live wandb Run object (before ``.finish()``).
        output_path: Path to the saved model checkpoint (e.g.
            ``models/hummingbird_classifier_20260430_193523.pt``).

    Returns:
        Dict mapping data type keys to saved file paths (or ``None`` when
        a data type could not be retrieved).
    """
    output_path = Path(output_path)
    out_dir = output_path.parent
    stem = output_path.stem

    wandb_history = fetch_wandb_history(wandb_run=wandb_run)
    system_metrics = fetch_wandb_system_metrics(wandb_run=wandb_run)
    wandb_metadata = fetch_wandb_metadata(wandb_run=wandb_run)
    wandb_summary = fetch_wandb_summary(wandb_run=wandb_run)
    output_log = fetch_wandb_output_log(wandb_run=wandb_run)

    saved: dict[str, Path | None] = {}

    if wandb_history is not None:
        saved["model_metrics"] = _save_csv(
            wandb_history, out_dir, stem, "wandb_model_metrics"
        )
    else:
        saved["model_metrics"] = None
        logger.warning("No wandb model metrics data retrieved.")

    if system_metrics is not None:
        saved["system_metrics"] = _save_csv(
            system_metrics, out_dir, stem, "wandb_system_metrics"
        )
    else:
        saved["system_metrics"] = None

    saved["metadata"] = _merge_into_metadata(
        stem, out_dir, wandb_metadata, wandb_summary
    )

    if output_log is not None:
        _log_path = out_dir / f"{stem}_wandb_output.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_path.write_text(output_log)
        logger.info("Saved %s", _log_path)
        saved["output_log"] = _log_path
    else:
        saved["output_log"] = None

    return saved

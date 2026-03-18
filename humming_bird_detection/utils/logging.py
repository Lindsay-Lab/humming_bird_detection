#logging.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from loguru import logger

def _console_sink():
    """
    Returns a callable sink that plays nicely with tqdm if available,
    else falls back to print().
    """
    try:
        from tqdm import tqdm
        return lambda msg: tqdm.write(msg, end="")
    except ModuleNotFoundError:
        return lambda msg: print(msg, end="")

def setup_logging(
    logs_dir: Path, name: str, level: str = "INFO",
    also_console: bool = True,) -> Path:
    """
    Sets up Loguru logging for both console and file.
    Always writes a timestamped log file to logs_dir.
    If SLURM_JOB_ID exists, appends it to the filename.
    Returns the log file path.
    """
    logger.remove()

    if also_console:
        logger.add(sink=_console_sink(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
            level=level, enqueue=True,)

    logs_dir.mkdir(parents=True, exist_ok=True)
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    suffix = f"_job{slurm_job_id}" if slurm_job_id else ""
    log_file = logs_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}{suffix}.log"
    logger.add(log_file, level=level, enqueue=True)
    logger.info(f"Log file: {log_file.resolve()}")
    return log_file
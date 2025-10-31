from pathlib import Path
import os
from datetime import datetime
from loguru import logger
from roboflow import Roboflow

try:
    from humming_bird_detection.config import LOGS_DIR
except Exception:
    LOGS_DIR = Path("logs")

def setup_logging() -> None:
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        enqueue=True,
    )
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"rf_download_{datetime.now():%Y%m%d_%H%M%S}_job{slurm_job_id}.log"
        logger.add(log_file, enqueue=True,)
        logger.info(f"File logging enabled: {log_file.resolve()}")

def validate_download(dataset):
    dl_path = Path(getattr(dataset, "location", ".")).resolve()
    logger.success(f"Downloaded dataset to: {dl_path}")

    if not dl_path.is_dir():
        logger.error(f"Downloaded path is not a directory: {dl_path}")
        return dl_path

    data_yaml = dl_path / "data.yaml"
    imgs = list((dl_path / "train" / "images").glob("*.*")) + \
           list((dl_path / "valid" / "images").glob("*.*")) + \
           list((dl_path / "val" / "images").glob("*.*"))

    if data_yaml.exists() and imgs:
        logger.success(f"Valid dataset — data.yaml found, total images: {len(imgs)}")
    else:
        logger.warning("Dataset may be incomplete or not in YOLO format.")
        logger.info(f"data.yaml exists: {data_yaml.exists()} | total images: {len(imgs)}")

    return dl_path

def download_dataset(api_key):
    logger.info(f"Connecting to Roboflow")
    rf = Roboflow(api_key=api_key)
    rf = Roboflow(api_key="emHcbgLhITmU2KHvC6I7")
    project = rf.workspace("humming-bird-detection").project("label-birdfeeder-camera-observations")
    version = project.version(3)
    dataset = version.download("yolov8")
    validate_download(dataset)

def main():
    setup_logging()
    api_key = os.getenv("ROBOFLOW_API_KEY", "emHcbgLhITmU2KHvC6I7")
    return download_dataset(api_key)

if __name__ == "__main__":
    main()

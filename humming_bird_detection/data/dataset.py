from pathlib import Path
import os
from datetime import datetime
from loguru import logger
from roboflow import Roboflow
from humming_bird_detection.utils.logging import setup_logging
from humming_bird_detection.configs.paths import LOGS_DIR

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
    project = rf.workspace("humming-bird-detection").project("label-birdfeeder-camera-observations")
    version = project.version(3)
    dataset = version.download("yolov8")
    dl_path = validate_download(dataset)
    return dl_path

def main():
    setup_logging(LOGS_DIR, name="rf_download", level="INFO")
    api_key = os.getenv("ROBOFLOW_API_KEY", "emHcbgLhITmU2KHvC6I7")
    return download_dataset(api_key)

if __name__ == "__main__":
    main()

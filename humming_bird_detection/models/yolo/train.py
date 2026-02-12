from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import torch
import sys
import argparse

DATA_DIR = Path("data/processed/hb_detection/hb_ready")
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:>8}</level> | <cyan>{message}</cyan>")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model to use (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--img-size", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    return parser.parse_args()

def main():
    args = get_args()
    batch_size = args.batch_size
    epochs = args.epochs
    img_size = args.img_size
    model_name = args.model
    lr = args.lr
    logger.info(f"Training with model={model_name}, batch_size={batch_size}, epochs={epochs}, img_size={img_size}, lr={lr}")

    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run preprocess.py first.")

    logger.info(f"Training YOLO model on dataset: {DATA_DIR} (device={DEVICE})")
    model = YOLO(model_name)

    try:
        logger.info(f"Starting training: epochs={epochs}, batch={batch_size}, imgsz={img_size}, lr0={lr}")
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr,
            project=str(CHECKPOINTS_DIR),
            name="hb_detection",
            exist_ok=True,
            device=DEVICE,
            verbose=True,
            save_period=1,
        )
        logger.success("Training finished")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    project_weights_dir = CHECKPOINTS_DIR / "hb_detection" / "weights"
    best_ckpt = project_weights_dir / "best.pt"
    last_ckpt = project_weights_dir / "last.pt"

    if best_ckpt.exists():
        ckpt = best_ckpt
    elif last_ckpt.exists():
        logger.warning("best.pt not found; falling back to last.pt")
        ckpt = last_ckpt
    else:
        pts = sorted(project_weights_dir.glob("*.pt"))
        if pts:
            ckpt = pts[-1]
            logger.warning(f"No best/last checkpoint found; using {ckpt.name}")
        else:
            raise FileNotFoundError(f"No checkpoints found under {project_weights_dir}")

    logger.info(f"Loading checkpoint for evaluation: {ckpt}")
    best_model = YOLO(str(ckpt))

    try:
        logger.info("Evaluating on test split...")
        results = best_model.val(data=str(data_yaml), split="test", batch=batch_size, device=DEVICE)
        logger.info(f"Eval results object: {results}")
        try:
            logger.info(f"Metrics: {getattr(results, 'metrics', None)}")
        except Exception:
            logger.info("Couldn't extract results.metrics")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

    test_images_dir = DATA_DIR / "test" / "images"
    if test_images_dir.exists():
        test_images = sorted(test_images_dir.glob("*"))[:5]
        if test_images:
            out_preds_dir = CHECKPOINTS_DIR / "predictions"
            out_preds_dir.mkdir(parents=True, exist_ok=True)
            try:
                logger.info(f"Running inference on {len(test_images)} test images...")
                best_model.predict(source=[str(p) for p in test_images], save=True, project=str(out_preds_dir), name="test_preds")
                logger.info(f"Saved predictions under {out_preds_dir}")
            except Exception as e:
                logger.error(f"Inference failed: {e}")
    else:
        logger.warning(f"No test images directory found at {test_images_dir}")

    logger.success(f"Evaluation complete! Best model used: {ckpt}")

if __name__ == "__main__":
    main()

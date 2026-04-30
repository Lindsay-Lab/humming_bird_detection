"""EfficientNetV2-S hummingbird binary classifier — inference and training."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from humming_bird_detection.project_paths import project_paths as _project_paths

logger = logging.getLogger(__name__)

# ImageNet channel-wise mean and std used to normalise EfficientNetV2-S inputs.
# These values match the statistics of the ImageNet-1K training set and must be
# consistent with the EfficientNet_V2_S_Weights.IMAGENET1K_V1 pretrained weights
# loaded below.  Changing them would shift the input distribution away from what
# the pretrained filters expect.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def _build_model() -> nn.Module:
    """Return EfficientNetV2-S with a binary classification head."""
    backbone = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),  # single logit + BCEWithLogitsLoss: numerically
        # stabler than 2-class softmax and natural for binary tasks with class
        # imbalance (pos_weight can be set per-sample without extra arguments).
    )
    return backbone


class HummingbirdClassifier:
    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> None:
        """Load weights from model_path, or initialise from ImageNet backbone."""
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = _build_model()

        if model_path is not None:
            resolved = Path(model_path)
            if resolved.exists():
                state = torch.load(resolved, map_location=device, weights_only=True)
                self._model.load_state_dict(state)
                logger.info("Loaded classifier weights from %s", resolved)
            else:
                logger.warning(
                    "model_path '%s' does not exist; using ImageNet-pretrained backbone",
                    resolved,
                )

        self._model.to(device)
        self._model.eval()

    def predict(self, images: list[np.ndarray]) -> list[tuple[int, float]]:
        """Classify a list of 224×224 BGR uint8 arrays as hummingbird (1) or other (0)."""
        from PIL import Image as PILImage  # noqa: PLC0415

        tensors = []
        for bgr in images:
            rgb = bgr[:, :, ::-1].copy()
            pil = PILImage.fromarray(rgb.astype(np.uint8))
            tensors.append(_VAL_TRANSFORMS(pil))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits = self._model(batch).squeeze(1)
            probs = torch.sigmoid(logits)

        results: list[tuple[int, float]] = []
        for p in probs.tolist():
            label = 1 if p >= self.confidence_threshold else 0
            results.append((label, float(p)))
        return results

    def train_model(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        epochs: int = 30,
        lr: float = 3e-4,
        batch_size: int = 32,
        output_path: str | Path | None = None,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 5,
        wandb_run=None,
        num_workers: int = 0,
    ) -> dict:
        """Fine-tune the classifier on labelled crops.

        Each of ``train_dir`` / ``val_dir`` must contain two sub-directories:
        ``hummingbird/`` and ``other/``.  Uses ``ImageFolder`` to load data.

        Training proceeds in two phases:
        - Phase 1 (epochs 1–5): backbone frozen, only the classification head
          is updated at the full learning rate.
        - Phase 2 (remaining epochs): all parameters unfrozen, LR reduced to
          ``lr / 10``, ``CosineAnnealingLR`` applied for smooth decay.

        If ``epochs`` ≤ 5 the backbone remains frozen for the entire run
        (Phase 2 never executes); a warning is logged in this case.

        ``num_workers`` is passed to both DataLoaders.  The default (0) ensures
        reproducible sample ordering by avoiding multiprocessing non-determinism;
        increase on multi-core machines for faster data loading.

        Early stopping monitors validation loss; the best checkpoint is saved
        at ``output_path`` (defaults to
        ``project_paths.models / "hummingbird_classifier.pt"``).

        Returns ``{"best_val_loss": float, "best_epoch": int}``.
        """
        train_dir = Path(train_dir)
        val_dir = Path(val_dir)
        save_path = (
            Path(output_path)
            if output_path is not None
            else (_project_paths.models / "hummingbird_classifier.pt")
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        _TRAIN_TRANSFORMS = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # shift up to 10% of image dims
                    scale=(0.9, 1.1),  # zoom ±15%
                ),
                transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

        train_dataset = datasets.ImageFolder(
            str(train_dir), transform=_TRAIN_TRANSFORMS
        )
        val_dataset = datasets.ImageFolder(str(val_dir), transform=_VAL_TRANSFORMS)

        # ImageFolder sorts subdirs alphabetically: hummingbird→0, other→1.
        # The model is trained to output high logits for hummingbird (positive=1).
        # We remap so that hummingbird samples get target=1 in BCEWithLogitsLoss.
        hb_idx = train_dataset.class_to_idx.get("hummingbird")
        if hb_idx is None:
            raise ValueError(
                "Expected a 'hummingbird' subdirectory inside train_dir; "
                f"found classes: {list(train_dataset.class_to_idx.keys())}"
            )

        n_positive = sum(1 for _, lbl in train_dataset.samples if lbl == hb_idx)
        n_negative = len(train_dataset) - n_positive
        if n_positive == 0 or n_negative == 0:
            raise ValueError(
                f"Training set must contain both classes; "
                f"found {n_positive} hummingbird and {n_negative} other samples."
            )
        pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32).to(
            self.device
        )
        logger.info(
            "Training set: %d hummingbird, %d other  →  pos_weight=%.3f",
            n_positive,
            n_negative,
            pos_weight.item(),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        head_freeze_epochs = min(5, epochs)

        if epochs <= head_freeze_epochs:
            logger.warning(
                "epochs=%d ≤ head_freeze_epochs=%d: the backbone will remain frozen "
                "for the entire training run and Phase 2 (full fine-tuning) will "
                "never execute. Set epochs > %d to enable backbone fine-tuning.",
                epochs,
                head_freeze_epochs,
                head_freeze_epochs,
            )

        # --- Phase 1: head-only ---
        for param in self._model.features.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_state: dict = copy.deepcopy(self._model.state_dict())

        self._model.to(self.device)

        for epoch in range(1, epochs + 1):
            # Switch to full fine-tuning after head_freeze_epochs
            if epoch == head_freeze_epochs + 1:
                for param in self._model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(
                    self._model.parameters(),
                    lr=lr / 10,
                    weight_decay=weight_decay,
                )
                remaining = epochs - head_freeze_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=remaining,
                    eta_min=1e-6,
                )
                logger.info("Epoch %d: unfreezing backbone, lr=%.2e", epoch, lr / 10)

            # --- train ---
            self._model.train()
            train_loss = 0.0
            train_correct = 0
            for images, folder_labels in train_loader:
                images = images.to(self.device)
                targets = folder_labels.eq(hb_idx).float().to(self.device)
                optimizer.zero_grad()
                logits = self._model(images).squeeze(1)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(images)
                train_correct += (
                    (torch.sigmoid(logits) >= 0.5).eq(targets.bool()).sum().item()
                )
            train_loss /= len(train_dataset)
            train_acc = train_correct / len(train_dataset)

            if epoch > head_freeze_epochs:
                scheduler.step()  # type: ignore[possibly-undefined]
            current_lr = optimizer.param_groups[0]["lr"]

            # --- validate ---
            self._model.eval()
            val_loss = 0.0
            val_preds: list[float] = []
            val_tgts: list[float] = []
            with torch.no_grad():
                for images, folder_labels in val_loader:
                    images = images.to(self.device)
                    targets = folder_labels.eq(hb_idx).float().to(self.device)
                    logits = self._model(images).squeeze(1)
                    val_loss += criterion(logits, targets).item() * len(images)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_preds.extend(preds.cpu().tolist())
                    val_tgts.extend(targets.cpu().tolist())
            val_loss /= len(val_dataset)

            # per-epoch classification metrics
            vp = torch.tensor(val_preds)
            vt = torch.tensor(val_tgts)
            val_acc = vp.eq(vt).float().mean().item()
            tp = int(((vp == 1) & (vt == 1)).sum())
            fp = int(((vp == 1) & (vt == 0)).sum())
            fn = int(((vp == 0) & (vt == 1)).sum())
            val_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            val_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val_f1 = (
                2 * val_prec * val_rec / (val_prec + val_rec)
                if (val_prec + val_rec) > 0
                else 0.0
            )

            is_best = val_loss < best_val_loss

            logger.info(
                "Epoch %d/%d  lr=%.2e  train_loss=%.4f  train_acc=%.3f"
                "  val_loss=%.4f  val_acc=%.3f  val_prec=%.3f"
                "  val_rec=%.3f  val_f1=%.3f%s",
                epoch,
                epochs,
                current_lr,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_prec,
                val_rec,
                val_f1,
                "  ← best" if is_best else "",
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "train/lr": current_lr,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "val/precision": val_prec,
                        "val/recall": val_rec,
                        "val/f1": val_f1,
                        "val/is_best": is_best,
                    },
                    step=epoch,
                )

            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(self._model.state_dict())
                torch.save(best_state, save_path)
                logger.info("  → New best — checkpoint saved to %s", save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch,
                        early_stopping_patience,
                    )
                    break

        # Restore best weights in-memory
        self._model.load_state_dict(best_state)
        self._model.eval()
        logger.info(
            "Training complete — best_val_loss=%.4f at epoch %d",
            best_val_loss,
            best_epoch,
        )
        return {"best_val_loss": best_val_loss, "best_epoch": best_epoch}


def build_classifier(cfg: dict) -> HummingbirdClassifier:
    """Instantiate HummingbirdClassifier from a pipeline config dict."""
    cls_cfg = cfg["classifier"]
    model_path = _project_paths.models / cls_cfg["model"]
    return HummingbirdClassifier(
        model_path=model_path,
        confidence_threshold=cls_cfg["confidence_threshold"],
    )

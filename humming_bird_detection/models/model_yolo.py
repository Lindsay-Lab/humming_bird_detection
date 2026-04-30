import pytorch_lightning as pl
from PIL import Image
from ultralytics import YOLO

from humming_bird_detection.project_paths import project_paths


class BirdYOLO(pl.LightningModule):
    """Pytorch Lightning wrapper around an Ultralytics YOLO model.

    This class exposes YOLO through the Lightning interface for potential
    integration with Lightning trainers and callbacks.  Note that the standard
    Ultralytics training loop (``model.train(data=...)``) is preferred for
    fine-tuning because it handles data augmentation, class-count adjustment,
    and checkpoint logic natively.  This wrapper is provided for cases where
    Lightning-specific features (e.g. multi-GPU strategies or custom callbacks)
    are required.
    """

    def __init__(
        self,
        yolo_version: str = "yolo11n",
        n_classes: int = 80,
        **kwargs,
    ) -> None:

        # Store the keyword arguments as attributes
        model_args = {k: v for k, v in locals().items() if k != "self"}
        for name, value in model_args.items():
            if name != "self":
                setattr(self, name, value)

        # Call LightningModule constructor
        super(BirdYOLO, self).__init__()

        # Load pretrained model
        self._load_pretrained_model(yolo_version=self.yolo_version)

        # Edit the architecture to fit the number of classes
        self._customize_architecture(n_classes=self.n_classes)

        # make all parameters trainable
        self.trainable_parameter_names = [p for p in list(self.state_dict().keys())]

    def _load_pretrained_model(self, yolo_version: str):
        """Load a pretrained YOLO checkpoint from the project models directory."""
        self.yolo_version = (
            yolo_version if ".pt" in yolo_version else f"{yolo_version}.pt"
        )
        pretrained_path = project_paths.models / "YOLO" / self.yolo_version
        self.model = YOLO(pretrained_path)
        return None

    def _customize_architecture(self, n_classes: int) -> None:
        """Adjust the model for a custom number of output classes.

        When fine-tuning via the Ultralytics training API
        (``model.train(data='data.yaml', ...)``), the detection head is resized
        automatically based on the ``nc`` field in data.yaml.  Manual head
        modification is therefore not required for the standard workflow and
        this method is intentionally a no-op.

        Override this method in a subclass if non-standard architectural
        changes are needed (e.g. adding a secondary classification branch).
        """
        pass
        return None

    def forward(self, x):
        """Run YOLO inference on input x."""
        return self.model(x)


if __name__ == "__main__":
    # Example: load the model and run inference on a sample image.
    model = BirdYOLO(n_classes=80)

    image_number = 253176611  # example: 1 bottle with 2 birds
    example_image_path = (
        project_paths.data.raw / "Trochilidae" / "images" / f"{image_number}.jpg"
    )
    example_image = Image.open(example_image_path)

    output = model(example_image)
    for result in output:
        print(result.boxes)
        result.show()

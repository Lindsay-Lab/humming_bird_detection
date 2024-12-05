from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from ultralytics import YOLO

from humming_bird_detection.project_paths import project_paths


class BirdYOLO(pl.LightningModule):
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
        self._customize_architecturen(n_classes=self.n_classes)
        
        # make all parameters trainable
        self.trainable_parameter_names = [p for p in list(self.state_dict().keys())]
        
    def _load_pretrained_model(self, yolo_version: str):
        self.yolo_version = yolo_version if ".pt" in yolo_version else f"{yolo_version}.pt"
        pretrained_path = project_paths.models / 'YOLO' / self.yolo_version
        self.model = YOLO(pretrained_path)
        return None
        
    def _customize_architecturen(self, n_classes: int) -> None:
        pass
        return None
    
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self, recurrence_lr_factor=0.1):
        optimizer_class = getattr(torch.optim, self.optimizer)

        lr = getattr(self, "lr", getattr(self, "learning_rate", None))

        # Reduce the learning rate for the recurrence weights
        lr_recurrence = lr * recurrence_lr_factor

        # Split the parameters into recurrent and non-recurrent
        recurrence_params, params = [], []

        self.print_trainable_parameter_names()

        for name, param in self.named_parameters():
            if name in self.trainable_parameter_names:
                if "recurrence" in name:
                    recurrence_params.append(param)
                else:
                    params.append(param)

        # Create the optimizers
        params_args = [{"params": params}]
        if recurrence_params:
            params_args.append({"params": recurrence_params, "lr": lr_recurrence})

        self.optimizer = optimizer_class(params_args, lr=lr)

        return super(BirdYOLO, self).configure_optimizers()


if __name__ == "__main__":
    model = BirdYOLO(n_classes=80)

    # Load image from example_image_path
    image_number = 253176611    # 1 bottle with 2 birds
    # image_number = 253160453    # bird    
    example_image_path = Path(f"/home/rgutzen/Projects/humming_bird_detection/data/raw/Trochilidae/{image_number}.jpg")
    example_image = Image.open(example_image_path)

    # Pass the image through the model
    output = model(example_image)

    for result in output:
        print(result.boxes)  # Print detection boxes
        result.show()  # Display the annotated image
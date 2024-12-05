import pytorch_lightning as pl
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
        self._customize_architecture(n_classes=self.n_classes)
        
        # make all parameters trainable
        self.trainable_parameter_names = [p for p in list(self.state_dict().keys())]
        
    def _load_pretrained_model(self, yolo_version: str):
        self.yolo_version = yolo_version if ".pt" in yolo_version else f"{yolo_version}.pt"
        pretrained_path = project_paths.models / 'YOLO' / self.yolo_version
        self.model = YOLO(pretrained_path)
        return None
        
    def _customize_architecture(self, n_classes: int) -> None:
        detect_module = self.model._modules['model']._modules['model'][-1]        
        pass
        return None
    
    def forward(self, x):
        return self.model(x)
        
        
if __name__ == "__main__":
    model = BirdYOLO(n_classes=80)

    # Load image from example_image_path
    image_number = 253176611    # 1 bottle with 2 birds
    image_number = 253160453    # bird    
    example_image_path = project_paths.data.raw / 'Trochilidae' / 'images' / f"{image_number}.jpg"
    example_image = Image.open(example_image_path)

    # Pass the image through the model
    output = model(example_image)

    for result in output:
        print(result.boxes)  # Print detection boxes
        result.show()  # Display the annotated image
    
    # fine_output = model.model.train(
    #     data=project_paths.data.interim / 'finetune_data.yaml',
    #     epochs=5,
    #     imgsz=640
    # )
    
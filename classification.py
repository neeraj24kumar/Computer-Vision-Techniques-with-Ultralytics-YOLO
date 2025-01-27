from ultralytics import YOLO
from enum import Enum

# Define YOLO models for classification
class ModelType(Enum):
    YOLO11n_Classify = 'yolo11n-cls.pt' 


class Camera(Enum):
    LAPTOP = '0'  # Laptop webcam (default)
   

def live_classification(modelType: ModelType):
    # Load the YOLO model
    model = YOLO(modelType.value)
    print(f"Using model: {modelType.name} for classification")

    # Perform classification
    model.predict(source=Camera.LAPTOP.value, task="classify", show=True)

if __name__ == '__main__':
    live_classification(ModelType.YOLO11n_Classify)

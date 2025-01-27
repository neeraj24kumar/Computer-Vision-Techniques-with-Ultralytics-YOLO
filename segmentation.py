from ultralytics import YOLO
from enum import Enum

# Define available YOLO models for segmentation
class ModelType(Enum):
    YOLO11n_Segment = 'yolo11n-seg.pt'  

class Camera(Enum):
    LAPTOP = '0'  # Laptop webcam (default)
   

def live_segmentation(modelType: ModelType):
    # Load the YOLO model
    model = YOLO(modelType.value)
    print(f"Using model: {modelType.name} for segmentation")

    # Perform segmentation
    model.predict(source=Camera.LAPTOP.value, task="segment", show=True)

if __name__ == '__main__':
    # Select the YOLO model for segmentation
    live_segmentation(ModelType.YOLO11n_Segment)

from ultralytics import YOLO
from enum import Enum

# Define available YOLO models for object detection
class ModelType(Enum):
    YOLO11n_Detect = 'yolo11n.pt'  # actual YOLO model trained for detection
    #YOLO11x_Detect = 'yolo11x.pt'

# Define available camera sources
class Camera(Enum):
    LAPTOP = '0'  # Laptop webcam (default)
   

def live_detection(modelType: ModelType):
    # Load the YOLO model
    model = YOLO(modelType.value)
    print(f"Using model: {modelType.name} for object detection")

    # Perform detection
    model.predict(source=Camera.LAPTOP.value, task="detect", show=True)

if __name__ == '__main__':
    # Select the YOLO model for object detection
    live_detection(ModelType.YOLO11n_Detect)

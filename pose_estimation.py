from ultralytics import YOLO
from enum import Enum

# Define available YOLO models for pose estimation
class ModelType(Enum):
    YOLO11n_Pose = 'yolo11n-pose.pt'  

class Camera(Enum):
    LAPTOP = '0'  # Laptop webcam (default)

def live_pose_estimation(modelType: ModelType):
    # Load the YOLO model
    model = YOLO(modelType.value)
    print(f"Using model: {modelType.name} for pose estimation")

    # Perform pose estimation
    model.predict(source=Camera.LAPTOP.value, task="pose", show=True)
    

if __name__ == '__main__':
    # Select the YOLO model for pose estimation
    live_pose_estimation(ModelType.YOLO11n_Pose)



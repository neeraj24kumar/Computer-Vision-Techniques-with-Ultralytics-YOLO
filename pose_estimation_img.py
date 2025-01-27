from ultralytics import YOLO
from enum import Enum
import cv2
import matplotlib.pyplot as plt

# Define available YOLO models for pose estimation
class ModelType(Enum):
    YOLO11n_Pose = 'yolo11n-pose.pt'  

def estimate_pose_on_image(modelType: ModelType, image_path: str):
    try:
        # Load the YOLO model
        model = YOLO(modelType.value)
        print(f"Using model: {modelType.name} for pose estimation")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at the provided path: {image_path}")

        # Perform pose estimation on the image
        results = model.predict(source=image, task="pose", show=True)

        # Display results using matplotlib
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axes
        plt.title('Pose Estimation Result')
        plt.show()

    except Exception as e:
        print(f"An error occurred during pose estimation: {e}")

if __name__ == '__main__':
    # Specify the path to your image
    image_path = 'D:/yoloproject/.venv/man-1282232_1280.jpg'  
    estimate_pose_on_image(ModelType.YOLO11n_Pose, image_path)
# Computer-Vision-Techniques-with-Ultralytics-YOLO
Explored YOLO for real-time object detection, classification, segmentation, tracking, and pose estimation. Analyzed YOLO's performance in terms of speed, accuracy, and scalability in dynamic environments.

The objective of the project "Computer Vision Techniques with Ultralytics YOLO" is to explore and implement various computer vision tasks using the latest YOLOv11 model, focusing on real-time object detection capabilities. By developing different programs tailored to specific tasks such as object detection, classification, and segmentation, the project aims to evaluate the performance of YOLOv11 in predicting multiple objects accurately. The analysis will involve testing the model with five distinct objects and documenting the results in a tabular format to assess its effectiveness and efficiency compared to previous versions.
### System Components and Technologies Used:
- Ultralytics YOLOv11: The latest version of the YOLO object detection model for real-time applications.
- OpenCV: A library used for image processing and video capture.
- NumPy: A numerical computing library for handling data arrays.
- Python: The programming language used for implementing the YOLO models and computer vision tasks.

### Project Flow:
•	Initialize the YOLOv11 model and load necessary libraries.
•	Capture video frames or images for analysis.
•	Process each frame using YOLOv11 to detect specified objects.
•	Record predictions and performance metrics for each object.
•	Present results in a structured table for comparison.

### About Ultralytics YOLO
Ultralytics YOLO is a state-of-the-art real-time object detection and image segmentation model, with the latest version being YOLOv11. Announced at the YOLO Vision 2024 event, YOLOv11 offers significant improvements in speed and accuracy compared to its predecessors, such as YOLOv8. The model is designed for various tasks, including object detection, image classification, pose estimation, and real-time tracking. YOLOv11 features a streamlined architecture that reduces parameters while enhancing performance, achieving faster inference times (up to 5-6 ms per image) suitable for real-time applications. It supports multiple platforms, from edge devices to cloud systems, making it versatile for different industries. Users can easily integrate YOLOv11 into their workflows for training and deployment. The model also emphasizes user-friendliness with extensive documentation and community support, allowing developers to leverage its capabilities effectively in diverse applications like autonomous vehicles and surveillance systems.

### Key Features of Ultralytics YOLOv11
•	Enhanced Feature Extraction: Improved backbone and neck architecture for more precise object detection and complex task performance.
•	Optimized for Efficiency and Speed: Refined architectural designs and optimized training pipelines for faster processing speeds while balancing accuracy and performance.
•	Greater Accuracy with Fewer Parameters: Achieves higher mean Average Precision (mAP) on the COCO dataset using 22% fewer parameters than YOLOv8m, enhancing computational efficiency without compromising accuracy.
•	Adaptability Across Environments: Seamlessly deployable on edge devices, cloud platforms, and systems with NVIDIA GPUs for maximum flexibility.
•	Broad Range of Supported Tasks: Supports various computer vision challenges, including object detection, instance segmentation, image classification, pose estimation, and oriented object detection (OBB).

![image](https://github.com/user-attachments/assets/595744be-1518-4549-ad37-853eb46c50e5)
Figure 1: Performance comparison of different YOLO versions

### Yolo11n Object Detection Model: - 
Object detection identifies and locates objects within an image using bounding boxes. YOLOv11 excels at real-time object detection, providing fast and accurate predictions of object classes and their positions in the frame.

![image](https://github.com/user-attachments/assets/8b2446ca-cf1e-484a-bf13-b24943676555)

Table 1: Yolo11n Object Detection Model

![image](https://github.com/user-attachments/assets/bfe68041-e25a-4ca6-bd85-12967bc52bf3)

Figure 2: Object 1

![image](https://github.com/user-attachments/assets/332ba843-7512-4d6d-bbe0-69f11c0d6539)

Figure 3: Object 2

![image](https://github.com/user-attachments/assets/867767d1-a4c3-448b-bf0a-5dcbca80d9d6)

Figure 4: Object 3

![image](https://github.com/user-attachments/assets/8059e003-f617-4665-bb3f-0157d1e906b9)

Figure 5: Object 4

![image](https://github.com/user-attachments/assets/cd11ab81-79b6-43e8-924f-de0b77edbe27)

Figure 6: Object 5

### Result: - 
Among the five objects mentioned above, object 5 achieved the highest confidence level of 0.93, while object 2 had the lowest confidence level of 0.35. The model can predict four objects with confidence levels greater than 0.50, indicating that it is a good detection model that requires less training time.


### Yolo11n-cls Object Classification Model: - 
The task of identifying and labelling objects within an image. YOLOv11 can classify images into predefined categories, such as distinguishing between different types of animals or vehicles.

![image](https://github.com/user-attachments/assets/2a66c681-0282-4f98-9441-e17f58b67b37)

Table 2: Yolo11n-cls Object Classification Model

![image](https://github.com/user-attachments/assets/c8af8530-20dd-4e78-853a-ca9aed347597)

Figure 7: Object 1

![image](https://github.com/user-attachments/assets/9d680dce-a952-4024-b2e5-a5f6c8aea11f)

Figure 8: Object 2


  
Figure 9: Object 3

  
Figure 10: Object 4

  
Figure 11: Object 5

Result: - 
Among the five objects mentioned above, object 2 achieved the highest confidence level of 0.38, while object 5 had the lowest confidence level of 0. The model can predict four objects with confidence levels below 0.50, suggesting that it is a poor classification model that requires more training time.
.

### Yolo11n-seg Object Segmentation Model: - 
This involves partitioning an image into multiple segments or regions to simplify the representation of an image. YOLOv11 performs instance segmentation, which not only detects objects but also delineates their boundaries, allowing for pixel-level classification.


Table 3: Yolo11n-seg Object Segmentation Model
  
Figure 24: Object 1


  
Figure 25: Object 2



  
Figure 26: Object 3

  
Figure 27: Object 4

  
Figure 28: Object 5

Result: - 
Among the five objects mentioned above, object 5 achieved the highest confidence level of 0.93, while object 2 had the lowest confidence level of 0.47. The model can predict all five objects with confidence levels greater than 0.50, indicating that it is a good segmentation model that requires less training time.

Yolo11n-pose Object Pose Estimation Model: - 
This task estimates the positions of key body joints or landmarks in an image to understand human poses. YOLOv11 can detect and analyse human poses, making it useful for applications in sports analytics and human-computer interaction.

  
Figure 19: Pose Estimation
Yolo11n Object Tracking Model: - 
Tracking involves following detected objects across video frames over time. YOLOv11 supports tracking capabilities, enabling continuous identification of moving objects in a sequence of frames.

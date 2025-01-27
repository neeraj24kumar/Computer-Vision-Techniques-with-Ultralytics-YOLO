from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("D:/yoloproject/.venv/f1car.mp4",save=True, show=True)
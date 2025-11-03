from ultralytics import YOLO

model = YOLO("yolo11n.pt")
# model = YOLO("rtdetr-l.pt")

results = model("../sample-images/photo-1.jpg", conf=0.1)

for result in results:
    result.show()
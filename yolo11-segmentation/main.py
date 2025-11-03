from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
results = model("../sample-images/photo-1.jpg", conf=0.2) 

for result in results:
    result.show()
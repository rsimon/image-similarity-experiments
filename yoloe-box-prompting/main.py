from PIL import Image, ImageDraw
from ultralytics import YOLOE

model = YOLOE("yoloe-11l-seg.pt")

scene_image_path = "../sample-images/map-1.jpg"
scene_image = Image.open(scene_image_path).convert("RGB")

visual_prompts = {
    "bboxes": [
        [1073, 1333, 1073 + 186, 1333 + 139], # [x1, y1, x2, y2] 
        [1320, 1042, 1320 + 117, 1042 + 146]
    ],
    "cls": [0, 0]  # Class IDs for each bounding box (temporary identifiers)
}

results = model.predict(scene_image_path, refer_image="../sample-images/map-1.jpg", visual_prompts=visual_prompts, conf=0.2)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
    
    draw = ImageDraw.Draw(scene_image)
    
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=10)

scene_image.show()

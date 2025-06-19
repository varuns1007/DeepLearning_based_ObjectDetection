import os
import json
import glob
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Evaluate a pretrained model.")
parser.add_argument('--images', type=str, required=True, help='Path to the images dir.')
parser.add_argument('--weights', type=str,default=False, required=True, help='Path to the pretrained model weights.')
parser.add_argument('--output', type=str, required=True, help='Path to save prediction results.')


args = parser.parse_args()

# Load YOLOv8 model
model = YOLO(args.weights)

# Define source image directory (can contain images or folders of images)
source_dir = args.images

# Get all image paths recursively
image_paths = sorted(glob.glob(os.path.join(source_dir, "**", "*.*"), recursive=True))
image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]

# Create image ID mapping
images = []
image_id_map = {}  # maps filename to ID
for idx, path in enumerate(image_paths, start=1):
    filename = os.path.basename(path)
    image_id_map[filename] = idx
    images.append({
        "id": idx,
        "file_name": os.path.relpath(path, source_dir)
    })

# Run prediction
results = model.predict(source=image_paths, save=False)

# Build annotations
annotations = []
ann_id = 1

for result in results:
    filename = os.path.basename(result.path)
    image_id = image_id_map[filename]

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(box.cls[0].item()),
            "bbox": [x1, y1, w, h],
            "score": float(box.conf[0].item())
        }
        annotations.append(ann)
        ann_id += 1

# Final COCO-style predictions
coco_preds = {
    "images": images,
    "annotations": annotations
}

os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Save to file
with open(args.output, "w") as f:
    json.dump(coco_preds, f, indent=2)

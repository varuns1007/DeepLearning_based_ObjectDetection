from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
import supervision as sv
from PIL import Image
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import torch
import torch.nn as nn
import argparse


def get_model(prefix_type,pre_existing_weights_path):
    model = load_model(CONFIG_PATH, pre_existing_weights_path, device=device)

    if prefix_type == 'prefix_tuning':
        # Add prefix tuning to the text encoder (assuming it's CLIP-based)
        pass


    return model

def visualize(prompt_type,model):
    all_annotations = []
    all_images = []
    annotation_id = 1


    for img_id in image_ids:
        filename = id_to_filename.get(img_id)
        # img_path = os.path.join(val_img_root, filename)
        img_path = filename

        
        # print(f"\nProcessing: {filename}")
        all_images.append({
            "id": img_id,
            "file_name": os.path.relpath(img_path, val_img_root)
        })
    
        image_source, image = load_image(img_path)

        if prompt_type == 'basic':
            boxes, logits, phrases = predict(
                model=model, 
                image=image, 
                caption=TEXT_PROMPT, 
                box_threshold=BOX_TRESHOLD, 
                text_threshold=TEXT_TRESHOLD,
                device=device
            )
        elif prompt_type == 'prefix_tuning':
            pass
            # boxes, logits, phrases = predict_with_prefix(
            #     model=model, 
            #     image=image, 
            #     caption=TEXT_PROMPT, 
            #     box_threshold=BOX_TRESHOLD, 
            #     text_threshold=TEXT_TRESHOLD
            # )
        
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    
        directory, filename = os.path.split(img_path)
        save_path = os.path.join(os.path.dirname(output_path),filename)
        os.makedirs(f"{os.path.dirname(output_path)}/{directory}", exist_ok=True)
        
        # Image.fromarray(annotated_frame).save(save_path)

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        
        for box, score, label in zip(boxes, logits.numpy(), phrases):

            x0,y0,x1,y1 = box_cxcywh_to_xyxy(box).tolist()
            
            all_annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": label_to_id.get(label,1),
                "bbox": [x0,y0,x1-x0,y1-y0],
                "score": float(score)
            })
            annotation_id += 1

    seen = set()
    unique_images = []
    for img in all_images:
        if img["id"] not in seen:
            seen.add(img["id"])
            unique_images.append(img)

    detection_results = {
        "images": unique_images,
        "annotations": all_annotations
    } 

    # Step 2: Save predictions to a temporary JSON file
    with open(output_path, "w") as f:
        json.dump(detection_results, f)
    




parser = argparse.ArgumentParser(description="Evaluate a pretrained model.")
parser.add_argument('--images', type=str,default=None,  required=False, help='Path to the image directory.')
parser.add_argument('--data', type=str,default=None, required=False, help='Path to the image directory.')
parser.add_argument('--weights', type=str, required=False, help='Path to the pretrained model weights.')
parser.add_argument('--output', type=str, required=True, help='Path to save prediction results.')
parser.add_argument('--strategy', type=str, required=False, help='Strategy for training')
parser.add_argument('--step', type=str, required=False, help='Training or Validation on trained weights.')


args = parser.parse_args()


CONFIG_PATH = './grounding_dino_dataset/GroundingDINO_SwinT_OGC.py'
# WEIGHTS_PATH = './grounding_dino_dataset/groundingdino_swint_ogc.pth'
WEIGHTS_PATH = args.weights


val_img_root = args.images if args.images != None else args.data
# 3. Paths

output_path = args.output
os.makedirs(os.path.dirname(output_path), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT_PROMPT = "person"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

label_to_id = {"person":1,"car":2,"train":3,"rider":4,"truck":5,"motorcycle":6,"bicycle":7, "bus":8}

image_dir = val_img_root
image_files = []

# Walk through the directory and subdirectories
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            # Get full path to the image
            image_files.append(os.path.join(root, file))

# Sort the image files (optional)
image_files = sorted(image_files)

# Map filenames to unique IDs
filename_to_id = {
    filename: idx for idx, filename in enumerate(image_files)
}

id_to_filename = {idx: filename for idx, filename in enumerate(image_files)}

image_ids = list(id_to_filename.keys())

# print("filtered_annotations:",len(filtered_annotations))
# print("image_ids:",len(image_ids))

model = get_model('basic',WEIGHTS_PATH)
visualize('basic',model)
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageFont
import json
import argparse
from torchvision import transforms



def model_loader(train_type, pre_exisiting_weights_path=None):
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    
    if train_type == 'full':
        # UnFreeze all
        for param in model.parameters():
            param.requires_grad = True
        
    elif train_type == 'decoder_only':
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only decoder
        for param in model.model.decoder.parameters():
            param.requires_grad = True
        
        # # Check
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

    elif train_type == 'encoder_only':
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze encoder parameters
        for param in model.model.encoder.parameters():
            param.requires_grad = True
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

    if pre_exisiting_weights_path != None:
        model.load_state_dict(torch.load(pre_exisiting_weights_path, map_location=torch.device(device)))

    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=2e-5,
        weight_decay=1e-4
    )

    model.to(device)
        
    return model,optimizer 

# filename_to_id = {}
# id_counter = 1
# 2. Dataset
class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = []
        
        # Walk through the directory and subdirectories
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    # Get full path to the image
                    self.image_files.append(os.path.join(root, file))
        
        # Sort the image files (optional)
        self.image_files = sorted(self.image_files)
        
        # Map filenames to unique IDs
        self.filename_to_id = {
            os.path.basename(filename): idx for idx, filename in enumerate(self.image_files)
        }

        # Define transformation (if provided)
        self.transform = transform or transforms.Compose([
            transforms.Resize((800, 800)),  # match model input if needed
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        image_id = self.filename_to_id[os.path.basename(img_path)]

        return {
            "image": image,
            "image_id": image_id,
            "file_name": img_path,
            "annotations": []  # No annotations for unlabeled images
        }
    
    # def __init__(self, image_dir, transform=None):
    #     self.image_dir = image_dir
    #     self.image_files = sorted([
    #         f for f in os.listdir(image_dir)
    #         if f.lower().endswith((".png", ".jpg", ".jpeg"))
    #     ])
    #     self.filename_to_id = {
    #         filename: idx for idx, filename in enumerate(self.image_files)
    #     }
    #     self.transform = transform or transforms.Compose([
    #         transforms.Resize((800, 800)),  # match model input if needed
    #         transforms.ToTensor()
    #     ])

    # def __len__(self):
    #     return len(self.image_files)

    # def __getitem__(self, idx):
    #     filename = self.image_files[idx]
    #     img_path = os.path.join(self.image_dir, filename)
    #     image = Image.open(img_path).convert("RGB")

    #     return {
    #         "image": image,
    #         "image_id": self.filename_to_id[filename],  # integer ID
    #         "file_name": filename,
    #         "annotations": []  # No GT annotations
    #     }
# Global mapping for image filenames to integer IDs


def collate_fn(batch):
    # global id_counter
    images = [item["image"] for item in batch]
    annotations = []

    for item in batch:
        ann = item["annotations"].copy()  # Create a copy to prevent modification of original data
        obj = {
            "image_id":item["image_id"],
            "annotations":ann
        } 
        annotations.append(obj)

    # Process images and annotations through the processor (assuming it is properly defined)
    encoding = processor(images=images,annotations=annotations, return_tensors="pt")
    
    # Add the image IDs to the encoding
    encoding["image_ids"] = [item["image_id"] for item in batch]
    encoding["filenames"] = [item["file_name"] for item in batch]

    
    # Add raw images and original sizes to the encoding
    encoding["raw_images"] = images
    encoding["original_sizes"] = [item["image"].size for item in batch]  # (H, W)
    
    return encoding

def create_class_mapping(my_classes, coco_classes):
    # This will store the mapped class IDs
    class_mapping = {}

    for my_id, my_class in my_classes.items():
        # Find the COCO class ID that matches your class name
        for coco_id, coco_class in coco_classes.items():
            if my_class.lower() == coco_class.lower():
                class_mapping[my_id] = coco_id
                break
    
    return class_mapping

def val(model, confidence_threshold, coco_idx_to_label):
    # ===== Validation =====
    model.eval()
    

    class_mapping = create_class_mapping(coco_idx_to_label, category_mapping)
    print("class_mapping:", class_mapping)

    all_annotations = []
    all_images = []
    annotation_id = 1  # start counting annotation IDs

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)

            labels = [{k: v.to(device) for k, v in t.items()} for t in batch.get("labels", [{}]*len(batch["image_ids"]))]

            outputs = model(pixel_values=pixel_values, labels=labels, pixel_mask=pixel_mask)

            target_sizes = torch.stack([
                torch.tensor(size[::-1]) for size in batch["original_sizes"]  # (height, width)
            ])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)

            for j, result in enumerate(results):
                image_id = int(batch["image_ids"][j])  # assume numeric string or int
                # image_filename = f"{image_id}.jpg"  # or adjust if your filenames differ
                image_filename = f"{batch['filenames'][j]}"  # or adjust if your filenames differ


                all_images.append({
                    "id": image_id,
                    "file_name": os.path.relpath(image_filename, val_img_root)
                })

                img = batch["raw_images"][j].copy()
                draw = ImageDraw.Draw(img)

                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x0, y0, x1, y1 = box
                    width = x1 - x0
                    height = y1 - y0

                    label_text = f"{coco_idx_to_label.get(label,label)}: {score:.2f}"

                    # Draw
                    bbox = font.getbbox(label_text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_background = [x0, y0 - text_height - 4, x0 + text_width + 4, y0]
                    draw.rectangle(text_background, fill="green")
                    draw.text((x0 + 2, y0 - text_height - 2), label_text, fill="white", font=font)
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

                    all_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_mapping.get(label, label)),
                        "bbox": [float(x0), float(y0), float(width), float(height)],
                        "score": float(score)
                    })
                    annotation_id += 1

                img.save(os.path.join(os.path.dirname(save_path), f"val_epoch_batch{i}_img{j}.png"))

    # Remove duplicates (in case same image_id appears multiple times)
    seen = set()
    unique_images = []
    for img in all_images:
        if img["id"] not in seen:
            seen.add(img["id"])
            unique_images.append(img)

    predictions = {
        "images": unique_images,
        "annotations": all_annotations
    }

    # with open(f"{save_path}/predictions.json", "w") as f:
    #     json.dump(predictions, f)
    with open(save_path, "w") as f:
        json.dump(predictions, f)
 

font = ImageFont.load_default()

# 1. Load Model and Processor
processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Evaluate a pretrained model.")
parser.add_argument('--images', type=str,default=None,  required=False, help='Path to the image directory.')
parser.add_argument('--data', type=str,default=None, required=False, help='Path to the image directory.')
parser.add_argument('--weights', type=str, required=False, help='Path to the pretrained model weights.')
parser.add_argument('--output', type=str, required=True, help='Path to save prediction results.')
parser.add_argument('--strategy', type=str, required=False, help='Strategy for training')
parser.add_argument('--step', type=str, required=False, help='Training or Validation on trained weights.')
parser.add_argument('--mode', type=str, required=True, help='Training or Zero shot Validation.')




args = parser.parse_args()

val_img_root = args.images if args.images != None else args.data

# 4. Datasets and Dataloaders
val_dataset = UnlabeledImageDataset(val_img_root)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)


category_mapping = {1:'person',2:"car",3:"train",4:"rider",5:"truck",6:"motorcycle",7:"bicycle", 8:"bus"}
coco_idx_to_label_after_trained = {1:'person',2:"car",3:"train",4:"rider",5:"truck",6:"motorcycle",7:"bicycle", 8:"bus"}
coco_idx_to_label_zeroshot = {0: 'N/A', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'N/A', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'N/A', 27: 'backpack', 28: 'umbrella', 29: 'N/A', 30: 'N/A', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'N/A', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'N/A', 67: 'dining table', 68: 'N/A', 69: 'N/A', 70: 'toilet', 71: 'N/A', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'N/A', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# 6. Training + Validation Loop
save_path = args.output
os.makedirs(os.path.dirname(save_path), exist_ok=True)

if args.mode == '1':
    model,_ = model_loader(None,args.weights)
    val(model=model,confidence_threshold=0.5,coco_idx_to_label=coco_idx_to_label_zeroshot)

else:   
    model,_ = model_loader(None,args.weights)
    val(model=model,confidence_threshold=0.3,coco_idx_to_label=coco_idx_to_label_after_trained)



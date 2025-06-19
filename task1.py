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

# 2. Dataset
class COCODataset(Dataset):
    def __init__(self, img_root, ann_file, processor):
        self.img_root = img_root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.processor = processor

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        path = img_info['file_name']  # use the path from annotation
        img_path = os.path.join(self.img_root, path)
        image = Image.open(img_path).convert("RGB")

        target = []
        for ann in anns:
            if 'bbox' in ann:
                target.append({
                    "bbox": ann["bbox"],
                    "category_id": ann["category_id"],
                })

        return {
            "image": image,
            "annotations": target,
            "image_id": img_id
        }

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    images = [item['image'] for item in batch]
    original_sizes = [image.size for image in images]

    annotations = []
    image_ids = []
    for item in batch:
        anns = []
        for ann in item['annotations']:
            bbox = ann['bbox']
            x, y, width, height = bbox
            anns.append({
                'bbox': bbox,
                'category_id': ann['category_id'],
                'area': width * height,     # <<<<<< added
                'iscrowd': 0                # <<<<<< added
            })
        annotations.append({
            'image_id': item['image_id'],
            'annotations': anns
        })
        image_ids.append(item['image_id'])

    encoding = processor(images=images, annotations=annotations, return_tensors="pt")
    encoding["original_sizes"] = original_sizes
    encoding["raw_images"] = images
    encoding["image_ids"] = image_ids
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


def train_and_val(num_epochs,optimizer,model):

    # 5. Optimizer
    coco_idx_to_label = {1:'person',2:"car",3:"train",4:"rider",5:"truck",6:"motorcycle",7:"bicycle", 8:"bus"}
    class_mapping = create_class_mapping(coco_idx_to_label,category_mapping)
    print("class_mapping:",class_mapping)

    train_loss_list = []
    test_loss_list = []
    epochs_list = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
    
        # ===== Train =====
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
    
            outputs = model(pixel_values=pixel_values, labels=labels)
    
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            optimizer.zero_grad()
    
            train_loss += loss.item()
    
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        
        # ===== Validation =====
        coco_predictions = []
        coco_groundtruths = []
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                
                outputs = model(pixel_values=pixel_values, labels=labels, pixel_mask=pixel_mask)
                loss = outputs.loss
    
                val_loss += loss.item()
    
                # Post-process outputs
                target_sizes = torch.stack([
                    torch.tensor(size[::-1]) for size in batch["original_sizes"]  # flip to (height, width)
                ])
                results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.005)
    
                # Save visualization
                for j, result in enumerate(results):
                    img = batch["raw_images"][j].copy()
                    draw = ImageDraw.Draw(img)
    
                    boxes = result["boxes"].cpu().numpy()
                    scores = result["scores"].cpu().numpy()
                    labels = result["labels"].cpu().numpy()
    
                    for box, score, label in zip(boxes, scores, labels):
                        x0, y0, x1, y1 = box
                        label_text = f"{coco_idx_to_label.get(label, label)}: {score:.2f}"
    
                        # Measure text size
                        bbox = font.getbbox(label_text)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Set text box coordinates (above the rectangle)
                        text_background = [x0, y0 - text_height - 4, x0 + text_width + 4, y0]
                    
                        # Draw the green background rectangle
                        draw.rectangle(text_background, fill="green")
                    
                        # Draw the text over it
                        draw.text((x0 + 2, y0 - text_height - 2), label_text, fill="white", font=font)
                        
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    
                        coco_predictions.append({
                            "image_id": batch["image_ids"][j],  # You need correct image ids here!
                            "category_id": int(class_mapping.get(label,label)),
                            "bbox": [
                                float(x0),
                                float(y0),
                                float(x1 - x0),
                                float(y1 - y0),
                            ],
                            "score": float(score),
                        })
    
                    # img.save(os.path.join(os.path.dirname(save_path), f"val_epoch{epoch+1}_batch{i}_img{j}.png"))

        #saving checkpoints
        
        torch.save(model.state_dict(), save_path)
        print(f"Saved model checkpoint at {save_path}")

    
        train_loss_list.append(train_loss / len(train_loader))
        test_loss_list.append(val_loss / len(val_loader))
        epochs_list.append(epoch)
    
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_list, train_loss_list, label="Train Loss", marker='o')
        plt.plot(epochs_list, test_loss_list, label="Val Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Val Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{os.path.dirname(save_path)}/train_val_loss_plot.png")
    
        # Save predictions
        # with open(f"{os.path.dirname(save_path)}/predictions.json", "w") as f:
        #     json.dump(coco_predictions, f)
    
        # Evaluate
        # coco_gt = COCO(val_ann_file)
        # coco_dt = coco_gt.loadRes(f"{os.path.dirname(save_path)}/predictions.json")
        
        # coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
    
        # metric_names = [
        #     "Average Precision (AP) @[IoU=0.50:0.95 | area=all | maxDets=100]",
        #     "Average Precision (AP) @[IoU=0.50      | area=all | maxDets=100]",
        #     "Average Precision (AP) @[IoU=0.75      | area=all | maxDets=100]",
        #     "Average Precision (AP) @[IoU=0.50:0.95 | area=small | maxDets=100]",
        #     "Average Precision (AP) @[IoU=0.50:0.95 | area=medium | maxDets=100]",
        #     "Average Precision (AP) @[IoU=0.50:0.95 | area=large | maxDets=100]",
        #     "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=1]",
        #     "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=10]",
        #     "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=100]",
        #     "Average Recall (AR)    @[IoU=0.50:0.95 | area=small | maxDets=100]",
        #     "Average Recall (AR)    @[IoU=0.50:0.95 | area=medium | maxDets=100]",
        #     "Average Recall (AR)    @[IoU=0.50:0.95 | area=large | maxDets=100]",
        # ]
        
        # Save all metrics
        # with open(f"{os.path.dirname(save_path)}/coco_eval_full_results.txt", 'w') as f:
        #     for name, value in zip(metric_names, coco_eval.stats):
        #         f.write(f'{name}: {value:.4f}\n')

def val_zeroshot(model,confidence_threshold):
    # ===== Validation =====
    coco_predictions = []
    coco_groundtruths = []
    model.eval()
    val_loss = 0

    coco_idx_to_label = {0: 'N/A', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'N/A', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'N/A', 27: 'backpack', 28: 'umbrella', 29: 'N/A', 30: 'N/A', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'N/A', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'N/A', 67: 'dining table', 68: 'N/A', 69: 'N/A', 70: 'toilet', 71: 'N/A', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'N/A', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
    class_mapping = create_class_mapping(coco_idx_to_label,category_mapping)
    print("class_mapping:",class_mapping)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values, labels=labels, pixel_mask=pixel_mask)
            loss = outputs.loss

            val_loss += loss.item()

            # Post-process outputs
            target_sizes = torch.stack([
                torch.tensor(size[::-1]) for size in batch["original_sizes"]  # flip to (height, width)
            ])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)

            # Save visualization
            for j, result in enumerate(results):
                img = batch["raw_images"][j].copy()
                draw = ImageDraw.Draw(img)

                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x0, y0, x1, y1 = box

                    label_text = f"{coco_idx_to_label.get(label,label)}: {score:.2f}"
    
                    # Measure text size
                    bbox = font.getbbox(label_text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Set text box coordinates (above the rectangle)
                    text_background = [x0, y0 - text_height - 4, x0 + text_width + 4, y0]
                
                    # Draw the green background rectangle
                    draw.rectangle(text_background, fill="green")
                
                    # Draw the text over it
                    draw.text((x0 + 2, y0 - text_height - 2), label_text, fill="white", font=font)
                    
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                    # draw.text((x0, y0), f"{coco_idx_to_label.get(label,label)}:{score:.2f}", fill="red")

                    coco_predictions.append({
                        "image_id": batch["image_ids"][j],  # You need correct image ids here!
                        "category_id": int(class_mapping.get(label,label)),
                        "bbox": [
                            float(x0),
                            float(y0),
                            float(x1 - x0),
                            float(y1 - y0),
                        ],
                        "score": float(score),
                    })

                img.save(os.path.join(save_path, f"val_epoch_batch{i}_img{j}.png"))

    # Save predictions
    with open(f"{os.path.dirname(save_path)}/predictions.json", "w") as f:
        json.dump(coco_predictions, f)
    
    # Evaluate

    coco_dt = coco_gt.loadRes(f"{os.path.dirname(save_path)}/predictions.json")
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metric_names = [
        "Average Precision (AP) @[IoU=0.50:0.95 | area=all | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50      | area=all | maxDets=100]",
        "Average Precision (AP) @[IoU=0.75      | area=all | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50:0.95 | area=small | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50:0.95 | area=medium | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50:0.95 | area=large | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=1]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=10]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=small | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=medium | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=large | maxDets=100]",
    ]
    
    # Save all metrics
    with open(f"{os.path.dirname(save_path)}/coco_eval_full_results.txt", 'w') as f:
        for name, value in zip(metric_names, coco_eval.stats):
            f.write(f'{name}: {value:.4f}\n')

def val(model,confidence_threshold):
    # ===== Validation =====
    coco_predictions = []
    coco_groundtruths = []
    model.eval()
    val_loss = 0

    coco_idx_to_label = {1:'person',2:"car",3:"train",4:"rider",5:"truck",6:"motorcycle",7:"bicycle", 8:"bus"}
    class_mapping = create_class_mapping(coco_idx_to_label,category_mapping)
    print("class_mapping:",class_mapping)


    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values, labels=labels, pixel_mask=pixel_mask)
            loss = outputs.loss

            val_loss += loss.item()

            # Post-process outputs
            target_sizes = torch.stack([
                torch.tensor(size[::-1]) for size in batch["original_sizes"]  # flip to (height, width)
            ])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence_threshold)

            # Save visualization
            for j, result in enumerate(results):
                img = batch["raw_images"][j].copy()
                draw = ImageDraw.Draw(img)

                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x0, y0, x1, y1 = box

                    label_text = f"{coco_idx_to_label.get(label,label)}: {score:.2f}"
    
                    # Measure text size
                    bbox = font.getbbox(label_text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Set text box coordinates (above the rectangle)
                    text_background = [x0, y0 - text_height - 4, x0 + text_width + 4, y0]
                
                    # Draw the green background rectangle
                    draw.rectangle(text_background, fill="green")
                
                    # Draw the text over it
                    draw.text((x0 + 2, y0 - text_height - 2), label_text, fill="white", font=font)
                    
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                    # draw.text((x0, y0), f"{coco_idx_to_label.get(label,label)}:{score:.2f}", fill="red")

                    coco_predictions.append({
                        "image_id": batch["image_ids"][j],  # You need correct image ids here!
                        "category_id": int(class_mapping.get(label,label)),
                        "bbox": [
                            float(x0),
                            float(y0),
                            float(x1 - x0),
                            float(y1 - y0),
                        ],
                        "score": float(score),
                    })

                img.save(os.path.join(save_path, f"val_epoch_batch{i}_img{j}.png"))

    # Save predictions
    with open(f"{os.path.dirname(save_path)}/predictions.json", "w") as f:
        json.dump(coco_predictions, f)
    
    # Evaluate

    coco_dt = coco_gt.loadRes(f"{os.path.dirname(save_path)}/predictions.json")
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metric_names = [
        "Average Precision (AP) @[IoU=0.50:0.95 | area=all | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50      | area=all | maxDets=100]",
        "Average Precision (AP) @[IoU=0.75      | area=all | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50:0.95 | area=small | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50:0.95 | area=medium | maxDets=100]",
        "Average Precision (AP) @[IoU=0.50:0.95 | area=large | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=1]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=10]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=all | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=small | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=medium | maxDets=100]",
        "Average Recall (AR)    @[IoU=0.50:0.95 | area=large | maxDets=100]",
    ]
    
    # Save all metrics
    with open(f"{os.path.dirname(save_path)}/coco_eval_full_results.txt", 'w') as f:
        for name, value in zip(metric_names, coco_eval.stats):
            f.write(f'{name}: {value:.4f}\n')



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
parser.add_argument('--mode', type=str, required=True, help='Training or Zero shot Validation.')
parser.add_argument('--strategy', type=str, required=False, help='Strategy for training')
parser.add_argument('--step', type=str, required=False, help='Training or Validation on trained weights.')




args = parser.parse_args()

data_root = args.images if args.images != None else args.data
# 3. Paths
train_img_root = os.path.join(data_root,"foggy_dataset_A3_train")
train_ann_file = os.path.join(data_root,"annotations_train.json")
val_img_root = os.path.join(data_root,"foggy_dataset_A3_val")
val_ann_file = os.path.join(data_root,"annotations_val.json")

# print(f"{train_img_root}:{os.path.exists(train_img_root)},{train_ann_file}:{os.path.exists(train_ann_file)},{val_img_root}:{os.path.exists(val_img_root)},{val_ann_file}:{os.path.exists(val_ann_file)}")

# 4. Datasets and Dataloaders
train_dataset = COCODataset(train_img_root, train_ann_file, processor)
val_dataset = COCODataset(val_img_root, val_ann_file, processor)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)



# print("coco_idx_to_label:",coco_idx_to_label)

coco_gt = COCO(val_ann_file)
categories = coco_gt.loadCats(coco_gt.getCatIds())
category_mapping = {category['id']: category['name'] for category in categories}

# print("class_mapping:",class_mapping)

# 6. Training + Validation Loop
save_path = args.output
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# print(args)

# if args.mode == '1':
#     model,_ = model_loader('val',args.weights)
#     val_zeroshot(model=model,confidence_threshold=0.5)

# else:
    
if args.step == '1':
    train_type = 'full'

    if args.strategy == '2':
        train_type = 'decoder_only'
    elif args.strategy == '3':
        train_type = 'encoder_only'
        
    print(f"Train-type:{train_type}")
    model,optimizer = model_loader(train_type)
    train_and_val(num_epochs=15,optimizer=optimizer,model=model)

    # elif args.step == '2':

    #     model,_ = model_loader(None,args.weights)
    #     val(model=model,confidence_threshold=0.3)


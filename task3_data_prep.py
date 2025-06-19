import os
import json
from tqdm import tqdm
from PIL import Image
import argparse

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    return (x * dw, y * dh, w * dw, h * dh)

def find_image_path(image_dir, target_filename):
    # Search recursively for the file
    image_dir = os.path.join(image_dir,target_filename.split('/')[0])
    for root, _, files in os.walk(image_dir):
        if target_filename.split('/')[1] in files:
           
            # print("path:",os.path.join(root, target_filename.split('/')[1]))
            return os.path.join(root, target_filename.split('/')[1])
    return None

def convert_coco_to_yolo(coco_json_path, image_dir, output_dir, data_type):
    with open(coco_json_path) as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    if data_type == "train": 
        images_out_dir = os.path.join(output_dir, "images/train")
        labels_out_dir = os.path.join(output_dir, "labels/train")
    elif data_type == "val":
        images_out_dir = os.path.join(output_dir, "images/val")
        labels_out_dir = os.path.join(output_dir, "labels/val")
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    category_mapping = {cat['id']: i for i, cat in enumerate(coco['categories'])}
    image_id_to_name = {img['id']: img['file_name'] for img in coco['images']}

    annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        annotations.setdefault(img_id, []).append(ann)

    for img_id, anns in tqdm(annotations.items(), desc="Converting"):
        img_filename = image_id_to_name[img_id]
        img_path = find_image_path(image_dir, img_filename)
        if not img_path:
            print(f"Warning: {img_filename} not found in {image_dir}. Skipping.")
            continue

        try:
            img = Image.open(img_path)
            w, h = img.size
        except:
            print(f"Warning: Could not open image {img_path}. Skipping.")
            continue

        # Copy image to YOLO image directory
        os.makedirs(os.path.join(images_out_dir, img_filename.split('/')[0]), exist_ok=True)
        os.makedirs(os.path.join(labels_out_dir, img_filename.split('/')[0]), exist_ok=True)
        
        img_dest_path = os.path.join(images_out_dir, img_filename)
        if not os.path.exists(img_dest_path):
            img.save(img_dest_path)

        label_path = os.path.join(labels_out_dir, os.path.splitext(img_filename)[0] + ".txt")
        with open(label_path, 'w') as out_file:
            for ann in anns:
                bbox = ann['bbox']
                category_id = category_mapping[ann['category_id']]
                bb = convert_bbox((w, h), bbox)
                out_file.write(f"{category_id} {' '.join(map(str, bb))}\n")

    print(f"\n✅ Conversion completed! YOLO dataset at: {output_dir}")

parser = argparse.ArgumentParser(description="Evaluate a pretrained model.")
parser.add_argument('--images', type=str,default=None,  required=False, help='Path to the image directory.')
parser.add_argument('--data', type=str,default=None, required=False, help='Path to the image directory.')


args = parser.parse_args()

data_root = args.images if args.images != None else args.data
# 3. Paths
train_img_root = os.path.join(data_root,"foggy_dataset_A3_train")
train_ann_file = os.path.join(data_root,"annotations_train.json")
val_img_root = os.path.join(data_root,"foggy_dataset_A3_val")
val_ann_file = os.path.join(data_root,"annotations_val.json")

# Example usage
convert_coco_to_yolo(
    coco_json_path=train_ann_file,
    image_dir=train_img_root,  # Root folder with subdirectories
    output_dir="./datasets/yolo_dataset",
    data_type="train"
)

convert_coco_to_yolo(
    coco_json_path=val_ann_file,
    image_dir=val_img_root,  # Root folder with subdirectories
    output_dir="./datasets/yolo_dataset",
    data_type="val"
)

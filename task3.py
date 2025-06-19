from ultralytics import YOLO
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description="Evaluate a pretrained model.")
parser.add_argument('--weights', type=str,default=False, required=False, help='Path to the pretrained model weights.')
parser.add_argument('--output', type=str, required=True, help='Path to save prediction results.')
parser.add_argument('--mode', type=str, required=False, help='Training or Validation on trained weights.')


args = parser.parse_args()


MODEL_NAME = args.weights if args.weights else 'yolov8n.pt'        
DATA_YAML = "./datasets/yolo_dataset/data.yaml"  
EPOCHS = 250
IMG_SIZE = 1024
SAVE_DIR = f"{os.path.dirname(args.output)}" 
# VAL_OUTPUT_DIR = f"{args.output}/val"

os.makedirs(SAVE_DIR,exist_ok=True)

model = YOLO(MODEL_NAME)

if args.mode == '1':
    print("🔧 Starting Training...")
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, project=SAVE_DIR, name="train", exist_ok=True)

    os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)

    shutil.copy(f"{SAVE_DIR}/train/weights/best.pt", args.output)

else:
    print("🔍 Running Validation...")
    metrics = model.val(data=DATA_YAML,save=True, save_txt=True, save_conf=True, split='val')


    print("\n📊 Validation Results:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # Save mAP metrics to file
    text = f"mAP@0.5: {metrics.box.map50:.4f}\n"
    text += f"mAP@0.5:0.95: {metrics.box.map:.4f}\n"
    text += f"Precision: {metrics.box.mp:.4f}\n"
    text += f"Recall: {metrics.box.mr:.4f}\n"

    with open(f"{SAVE_DIR}/evaluation_result.txt", "w") as file:
        file.write(text)


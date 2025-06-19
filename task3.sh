#!/bin/bash

CURRENT_DIR=$(pwd)
mkdir -p $CURRENT_DIR/datasets/yolo_dataset

cat <<EOF > $CURRENT_DIR/datasets/yolo_dataset/data.yaml
train: $CURRENT_DIR/datasets/yolo_dataset/images/train
val: $CURRENT_DIR/datasets/yolo_dataset/images/val

nc: 8
names: ["person","car","train","rider","truck","motorcycle","bicycle", "bus"]
EOF

pip install ultralytics

STEP=$1

if [ "$STEP" -eq 1 ]; then
    DATASET_ROOT=$2
    TRAINED_MODEL_OUTPUT=$3

    python task3_data_prep.py --data "$2"
    python task3.py --output "$TRAINED_MODEL_OUTPUT" --mode "$STEP"

elif [ "$STEP" -eq 2 ]; then
    IMAGE_DIR=$2
    TRAINED_MODEL_INPUT=$3
    OUTPUT_PREDICTIONS=$4
    python task3_val.py --images "$IMAGE_DIR" --weights "$TRAINED_MODEL_INPUT" --output "$OUTPUT_PREDICTIONS"

else
    echo "Invalid STEP for Task 3."
fi

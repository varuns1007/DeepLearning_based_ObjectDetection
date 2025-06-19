#!/bin/bash

# Install GroundingDINO from GitHub
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

MODE=$1
STEP=$2

if [ "$MODE" -eq 1 ]; then
    # Zero-Shot Inference
    IMAGE_DIR=$3
    PRETRAINED_MODEL=$4
    OUTPUT_PREDICTIONS=$5
    python task2_zeroshot.py --images "$IMAGE_DIR" --weights "$PRETRAINED_MODEL" --output "$OUTPUT_PREDICTIONS"

elif [ "$MODE" -eq 2 ]; then
    if [ "$STEP" -eq 1 ]; then
        # Prompt Tuning - Training
        DATASET_ROOT=$3
        TRAINED_MODEL_OUTPUT=$4
        python task2.py --data "$DATASET_ROOT" --output "$TRAINED_MODEL_OUTPUT" --mode "$MODE" --step "$STEP"
    elif [ "$STEP" -eq 2 ]; then
        # Prompt Tuning - Inference
        IMAGE_DIR=$3
        TRAINED_MODEL_INPUT=$4
        OUTPUT_PREDICTIONS=$5
        python task2.py --images "$IMAGE_DIR" --weights "$TRAINED_MODEL_INPUT" --output "$OUTPUT_PREDICTIONS" --mode "$MODE" --step "$STEP"
    else
        echo "Invalid STEP for prompt tuning."
    fi
else
    echo "Invalid MODE for Task 2."
fi

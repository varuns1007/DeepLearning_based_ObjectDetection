#!/bin/bash

MODE=$1
STRATEGY=$2
STEP=$3

if [ "$MODE" -eq 1 ]; then
    # Evaluation of Pretrained Model
    IMAGE_DIR=$4
    PRETRAINED_MODEL=$5
    OUTPUT_PREDICTIONS=$6
    python task1_val.py --images "$IMAGE_DIR" --weights "$PRETRAINED_MODEL" --output "$OUTPUT_PREDICTIONS" --mode "$MODE" --step "$STEP"

elif [ "$MODE" -eq 2 ]; then
    # Fine-Tuning Strategies
    DATASET_ROOT=$4
    if [ "$STEP" -eq 1 ]; then
        TRAINED_MODEL_OUTPUT=$5
        python task1.py --strategy "$STRATEGY" --data "$DATASET_ROOT" --output "$TRAINED_MODEL_OUTPUT" --mode "$MODE" --step "$STEP"
    elif [ "$STEP" -eq 2 ]; then
        IMAGE_DIR=$4
        TRAINED_MODEL_INPUT=$5
        OUTPUT_PREDICTIONS=$6
        python task1_val.py --strategy "$STRATEGY" --images "$IMAGE_DIR" --weights "$TRAINED_MODEL_INPUT" --output "$OUTPUT_PREDICTIONS" --mode "$MODE" --step "$STEP"
    else
        echo "Invalid STEP for fine-tuning strategy."
    fi
else
    echo "Invalid MODE."
fi

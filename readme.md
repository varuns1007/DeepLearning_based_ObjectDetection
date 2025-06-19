## ✅ Task 1: Supervised Fine-Tuning and Evaluation

### 🔍 Evaluation of Pretrained Model

```bash
./task1.sh 1 0 2 <image_dir> <pretrained_model> <output_predictions>
```

### 🛠️ Fine-Tuning Strategies
#### Strategy 1
```bash
# Train the model
./task1.sh 2 1 1 <dataset_root> <trained_model_output>

# Inference using the trained model
./task1.sh 2 1 2 <image_dir> <trained_model_input> <output_predictions>
```

#### Strategy 2
```bash
# Train the model
./task1.sh 2 2 1 <dataset_root> <trained_model_output>

# Inference using the trained model
./task1.sh 2 2 2 <image_dir> <trained_model_input> <output_predictions>
```

#### Strategy 3
```bash
# Train
./task1.sh 2 3 1 <dataset_root> <trained_model_output>

# Inference
./task1.sh 2 3 2 <image_dir> <trained_model_input> <output_predictions>
```

## ✅ Task 2: Zero-Shot and Prompt Tuning using GroundingDino

### ❄️ Zero-Shot Inference
```bash
./task2.sh 1 2 <image_dir> <pretrained_model> <output_predictions>
```

### ✏️ Prompt Tuning
```bash
# Train
./task2.sh 2 1 <dataset_root> <trained_model_output>

# Inference
./task2.sh 2 2 <image_dir> <trained_model_input> <output_predictions>
```

## ✅ Task 3: Custom Extension or Advanced Strategy
```bash
# Train
./task3.sh 1 <dataset_root> <trained_model_output>

# Inference
./task3.sh 2 <image_dir> <trained_model_input> <output_predictions>
```

## 🧾 Argument Descriptions

| Argument                 | Description                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------|
| `<dataset_root>`         | Path to the root directory of the dataset. Must match the `foggy_cityscapes` folder structure.  |
| `<image_dir>`            | Path to the directory containing input images. Should follow the `train/val` structure.          |
| `<trained_model_output>` | Path (including filename) where trained model weights should be saved.                          |
| `<trained_model_input>`  | Path (including filename) from which the model weights should be loaded.                        |
| `<pretrained_model>`     | Path to the pretrained model to use for evaluation or zero-shot inference.                      |
| `<output_predictions>`   | Path (including filename) where the output prediction file should be saved.                     |

---

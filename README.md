# 🧵 Fabric Type Classifier

A deep learning image classifier that identifies fabric types (Cotton, Wool, Rayon, Silk) using a fine-tuned ResNet50 model. Includes a data augmentation pipeline to expand small datasets and a testing script that outputs side-by-side prediction charts.

## Files

| File | Description |
|---|---|
| `fabric_classifier.py` | Main training script — builds, trains, and saves the model |
| `augment.py` | Augmentation script — expands raw images with 11 transforms |
| `test_classifier.py` | Testing script — runs predictions and saves result images |
| `fabric_classifier.h5` | Saved trained model (generated after training) |
| `labels.json` | Class index-to-label mapping (generated after training) |

## Requirements

```bash
pip install tensorflow opencv-python matplotlib numpy
```

## Folder Structure

The scripts expect this directory layout:

```
Fabric detection/
├── Fabric detection1/        # Training data
│   ├── Cotton/
│   ├── Wool/
│   ├── Rayon/
│   └── Silk/
├── Fabric detection2/
│   └── Test/                 # Test images (.jpeg)
│       └── Results/          # Output folder (auto-created)
├── Fabric detection3/        # Validation data (also used for augmentation)
│   ├── Cotton/
│   ├── Wool/
│   ├── Rayon/
│   └── Silk/
├── fabric_classifier.h5
└── labels.json
```

> **Note:** Update the hardcoded `Path(r"D:\...")` variables in each script to match your local directory before running.

## Classes

| Label | Fabric Type |
|---|---|
| 0 | Cotton |
| 1 | Wool |
| 2 | Rayon |
| 3 | Silk |

---

## Scripts

### 1. `augment.py` — Data Augmentation

Run this first to expand your raw dataset before training. Each image generates 11 augmented variants saved as `.jpeg` in the same folder.

```bash
python augment.py
```

**Augmentations applied:**

| Suffix | Transform |
|---|---|
| `aug0` | Horizontal flip |
| `aug1` | Gaussian blur |
| `aug2` | 90° clockwise rotation |
| `aug3` | Brightness increase |
| `aug4` | Grayscale conversion |
| `aug5` | Bitwise invert |
| `aug6` | Darkening |
| `aug7` | Median blur |
| `aug10` | Random brightness/contrast |
| `aug12` | Erosion |
| `aug13` | Dilation |

---

### 2. `fabric_classifier.py` — Training

Trains a ResNet50-based classifier with progressive layer unfreezing and saves the model and label map.

```bash
python fabric_classifier.py
```

**Training details:**

| Parameter | Value |
|---|---|
| Base model | ResNet50 (ImageNet weights) |
| Input size | 224 × 224 |
| Batch size | 16 |
| Max epochs | 50 |
| Early stopping | patience = 12 on `val_loss` |
| Optimizer | Adam (lr = 0.003) |
| Loss | Categorical crossentropy |
| Dropout | 0.4 |

**Progressive unfreezing schedule:**

| Epoch | Layers unfrozen |
|---|---|
| 5 | Last 30 layers of ResNet50 |
| 10 | Last 50 layers of ResNet50 |
| 15 | All layers |

**Outputs:**
- `fabric_classifier.h5` — saved model
- `labels.json` — class index mapping

---

### 3. `test_classifier.py` — Testing

Loads the saved model and runs predictions on all `.jpeg` images in the `Test/` folder. Saves a side-by-side image of the original photo and a confidence bar chart to `Test/Results/`.

```bash
python test_classifier.py
```

**Output per image:**

```
[original image] | [bar chart: Cotton / Wool / Rayon / Silk % confidence]
```

Saved as `result_<original_filename>.jpeg` in the `Results/` folder.

---

## Classifying a Single Image

Inside `fabric_classifier.py`, call `classify_image()` after training:

```python
classify_image("path/to/your/image.jpeg")
```

This saves a bar chart of confidence scores as `classification_result.jpeg`.

## Notes

- All image paths are hardcoded to a local Windows directory — update them before running
- Training images should be organized into subfolders named exactly `Cotton`, `Wool`, `Rayon`, `Silk`
- Augmentation only processes `.jpeg` files — rename images if needed
- The testing script skips subdirectories and only reads `.jpeg` files directly in `Test/`
- `fabric_classifier.h5` and `labels.json` must be present in the model directory before running `test_classifier.py`

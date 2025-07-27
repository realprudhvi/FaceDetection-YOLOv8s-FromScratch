# 🧠 Face Detection with YOLOv8s (Trained from Scratch on WIDER FACE)

This repository demonstrates how to train a **YOLOv8s** model **from scratch** (without pre-trained weights) on the [WIDER FACE dataset](https://www.kaggle.com/datasets/canomercik/wider-face-dataset-for-yolov12-format). The result is a lightweight, real-time face detector suitable for both image and webcam input.

---

## Key Features

- **Model:** Implements the YOLOv8s (small) architecture via `yolov8s.yaml`.
- **Training from Scratch:** No pre-trained weights used (`pretrained=False`).
- **Dataset:** Uses the extensive WIDER FACE dataset formatted for YOLO.
- **Real-Time Ready:** The small YOLOv8s model is fast and effective for live webcam or video stream detection.
- **Reproducible Pipeline:** The provided notebooks guide you from training to evaluation with complete transparency.

---

##  Model Performance

After **50 epochs** of training on Kaggle using a P100 GPU, the model achieved the following results:

| Metric      | Value |
|-------------|--------|
| mAP50       | 0.666  |
| mAP50-95    | 0.367  |
| Precision   | 0.851  |
| Recall      | 0.582  |

Detailed charts and metrics can be found in the `Results/` directory.

---

## 📁 Directory Structure

```
mainfolder/
│
├── main.ipynb # Notebook for running inference on test images.
├── train-wider-yolov8s.ipynb # Notebook for training the model from scratch on Kaggle.
├── best.pt # Best trained model weights (ready for inference).
│
├── Test/ # Directory to place your images for testing.
│ └── (add your .png, .jpg, .jpeg, .webp images here)
│
└── Results/ # Directory for outputs and performance logs.
├── results.csv # Log of metrics from all training epochs.
└── Results_Visualization.ipynb # Notebook for custom visualization of training results.
```


---

##  Dataset

**WIDER FACE YOLOv12 Format Dataset**  
[Train on Kaggle](https://www.kaggle.com/datasets/canomercik/wider-face-dataset-for-yolov12-format)

Add the dataset into your notebook's input before training.

---

## ⚙️ Training Configuration

Training was performed on **Kaggle with a P100 GPU** using the following configuration:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.yaml")

model.train(
    pretrained=False,
    data="/kaggle/input/wider-face-dataset-for-yolov12-format/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    project="face-yolov8",
    name="tuned",

    # Optimization
    lr0=0.005,
    lrf=0.01,
    weight_decay=0.0005,
    optimizer="SGD",

    # Training Logic
    patience=10,
    cos_lr=True,
    cache=True,
    close_mosaic=15,

    # Logging & Saving
    save=True,
    save_period=5,
    val=True,
    plots=True,
    verbose=True
)
```

---

## 🖼️ Supported Input Formats

The model supports face detection in the following image formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

---

##  Installation

To run `main.ipynb`, install the required packages using the following command:

```bash
pip install ultralytics torch torchvision opencv-python matplotlib Pillow ipywidgets
```
---
## ▶️ Running Inference

Make sure `best.pt` is downloaded and placed inside the main directory.

Then, open and run `main.ipynb` to perform:

- 📷 **Webcam-based real-time detection**
- 🖼️ **Image-based detection using test samples**
---

## 📌 Notes

- All training and validation logs are stored in `Results/results.csv`.
- You can use `Results_Visualization.ipynb` to plot custom metrics and visualize training progress.
- The model uses **YOLOv8s** trained from **random initialization** — no transfer learning or pre-trained weights were used.

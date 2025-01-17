# Real-time Garbage Classification using CNN and Transfer Learning

This project implements real-time garbage classification using both Custom **CNN** and **Transfer Learning** approaches (MobileNetV2 & NASNetMobile). The system can identify different types of waste materials through computer vision to help automate waste sorting processes and improve recycling efficiency. Includes the [complete guide](https://github.com/g-wtham/trash-classification-cnn-transferlearning/tree/main/complete_document_guide) to re-implement this on your own! 

## Dataset Details

The project uses the TrashNet dataset:
- Contains 2527 images across 6 categories
- Image Resolution: 300x300 pixels
- Format: JPG
- Categories: cardboard, glass, metal, paper, plastic, trash
- Source: [TrashNet on Kaggle](https://www.kaggle.com/datasets/feyzazkefe/trashnet)

## Models & Results

### 1. Custom CNN Architecture
- Training Accuracy: 79.35%
- Validation Accuracy: 72.11%
- Average inference time: ~40ms per frame
- Architecture: 4 convolutional blocks with Conv2D layers (32→64 filters)

### 2. Transfer Learning Results

#### NASNet Mobile (30 epochs)
- Training Accuracy: 99.92%
- Validation Accuracy: 75.75%
- More balanced predictions across categories

#### MobileNetV2 (20 epochs)
- Training Accuracy: 99.63%
- Validation Accuracy: 34.79%
- Note: Shows significant misclassification of most categories as 'paper'

## Results Visualization

### CNN Results
Training after 50 Epochs<br><br>
<img src="https://github.com/g-wtham/trash-classification-cnn-transferlearning/blob/main/test_images_and_results/results/Accuracy%20after%2050%20Epochs.png" alt="Training after 50 Epochs" width="500" height="300">
<img src="https://github.com/g-wtham/trash-classification-cnn-transferlearning/blob/main/test_images_and_results/results/Training%20%26%20Validation%20Accuracy.png" alt="Train Validation Accuracy" width="500" height="300">

### MobileNetV2 Results
<img src="https://github.com/g-wtham/trash-classification-cnn-transferlearning/blob/main/test_images_and_results/results/MobileNetV2%20-%20Training%20%26%20Validation%20Accuracy.png" alt="MobileNetV2 Training & Validation Accuracy" width="500" height="300">
<img src="https://github.com/g-wtham/trash-classification-cnn-transferlearning/blob/main/test_images_and_results/results/MobileNetV2%20-%20Confusion%20Matrix%20-%20Trash%20Classification.png" alt="MobileNetV2 Confusion Matrix" width="500" height="500">

### NASNetMobile Results
<img src="https://github.com/g-wtham/trash-classification-cnn-transferlearning/blob/main/test_images_and_results/results/NasNetMobile%20-%20Training%20%26%20Validation%20Accuracy.png" alt="NASNetMobile Training & Validation Accuracy" width="500" height="300">
<img src="https://github.com/g-wtham/trash-classification-cnn-transferlearning/blob/main/test_images_and_results/results/NasNetMobile%20-%20Confusion%20Matrix%20-%20Trash%20Classification.png" alt="NASNetMobile Confusion Matrix" width="500" height="500">


## Requirements

### Hardware Requirements:
- Webcam for real-time detection
- Minimum 8GB RAM
- GPU recommended for faster training (or use Colab T4 GPU)

### Software Requirements:
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Seaborn
- Scikit-learn

## Installation & Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install tensorflow opencv-python numpy pillow matplotlib seaborn scikit-learn
```

3. Download the trained models:
- [CNN Model](https://drive.google.com/open?id=1ro1bbAyhnPL-LtV_qGtG-DnMXa-uXMcD)
- [MobileNetV2 Model](https://drive.google.com/open?id=1MJzpYQo7jK-fZjjuM29YF9_WvunILpvE)
- [NASNetMobile Model](https://drive.google.com/open?id=14lXAJL3we5qllEsYdZPFLYmD5QbMcNeN)

4. Run real-time detection:
```bash
python realtime_trash_detection_TRANSFER_LEARNING.py  # For transfer learning models
python realtime_trash_detection_CNN.py               # For CNN model
```

## Project Structure
```

├── models/
│   ├── CNN_trained_model.keras
│   ├── trash_classification_tf_mobilenet.h5
│   └── trash_classification_tf_nasnet.h5
├── /
│   ├── CNN Training.ipynb
│   ├── MobileNetV2 Transfer Learning.ipynb
│   └── NasNetMobile Transfer Learning.ipynb
└── /
    ├── realtime_trash_detection_CNN.py
    └── realtime_trash_detection_TRANSFER_LEARNING.py
```

## Conclusion

Based on the results, NASNetMobile performs better than both the custom CNN and MobileNetV2 for this specific waste classification task, achieving higher validation accuracy and more balanced predictions across categories.

## Done by
- Gowtham M, IIIrd Year (VCET)

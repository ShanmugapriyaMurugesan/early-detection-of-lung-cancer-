# Early detection of lung cancer using deep learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Lung cancer is a leading cause of cancer-related mortality worldwide. This project provides an automated, deep-learning-based system for the early detection and classification of lung cancer from CT scan images. Utilizing the **Xception** architecture with **Transfer Learning**, the model classifies scans into four categories:
* **Adenocarcinoma**
* **Large Cell Carcinoma**
* **Squamous Cell Carcinoma**
* **Normal (Healthy)**

## 🚀 Key Features
* **Architecture:** Xception backbone with custom classification layers.
* **Transfer Learning:** Pre-trained on ImageNet to enhance feature extraction.
* **Preprocessing:** Image normalization, resizing ($350 \times 350$), and data augmentation.
* **High Sensitivity:** Optimized to reduce False Negatives (critical for medical diagnosis).

## 📊 Dataset
The project uses the **Chest CT-Scan Images** dataset from Kaggle.
* **Total Images:** ~1,000 CT scans.
* **Train/Test/Val Split:** 613 / 315 / 72.
* **Classes:** 4 (Adenocarcinoma, Large Cell, Squamous Cell, Normal).

## 🛠️ Tech Stack
* **Language:** Python
* **Frameworks:** TensorFlow, Keras
* **Libraries:** NumPy, Pandas, Matplotlib, OpenCV
* **Environment:** Google Colab / Jupyter Notebook

## 🏗️ Model Architecture
1.  **Input Layer:** $350 \times 350 \times 3$ images.
2.  **Base Model:** Xception (Last 20 layers unfrozen for fine-tuning).
3.  **GlobalAveragePooling2D:** To condense features and prevent overfitting.
4.  **Dense Layers:** 256 units with ReLU activation.
5.  **Regularization:** Dropout layers (0.4 and 0.3).
6.  **Output Layer:** 4 units with Softmax activation.





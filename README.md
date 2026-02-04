# 🩻 The Radiologist’s Eye: Explainable AI for Chest X-Rays

> **Detecting 14 thoracic pathologies using Transfer Learning (ResNet50) and visualizing the diagnosis with Grad-CAM.**

![Project Banner](https://img.shields.io/badge/Status-Complete-green) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview
"The Radiologist's Eye" is a Deep Learning project designed to assist medical professionals in diagnosing chest X-rays. Unlike standard "black box" classifiers, this project emphasizes **Explainability (XAI)**.

It uses a fine-tuned **ResNet50** architecture to perform **Multi-Label Classification** on X-ray images (detecting multiple diseases simultaneously, such as Pneumonia AND Cardiomegaly). Crucially, it implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate heatmaps, pinpointing exactly *where* in the lungs the model is looking to make its decision.

---

## 🖼️ The "Why": Solving the Black Box Problem
In medical AI, accuracy is not enough. A model that predicts "Pneumonia" with 99% confidence is useless if it's looking at the patient's shoulder or a text marker.

**Key Engineering Challenge:**
During development, the initial model fell into a **Shortcut Learning** trap (the "Clever Hans" effect). It learned to predict "Sick" based on the text "PORTABLE" or the angle of the clavicles (shoulders) rather than lung opacity. 

**The Solution:**
I engineered a robust preprocessing pipeline using **Aggressive Center Cropping** (discarding the outer 15% of images) and **Zoom Augmentation**. This physically removes artifacts and text markers, forcing the model to focus exclusively on the lung anatomy.

### 🔍 Visualization Demo
*(The model correctly identifying the heart region for Cardiomegaly vs. ignoring background artifacts)*

![Grad-CAM Example](assets/heatmap_demo.png)
*(Note: Replace this path with your actual image path)*

---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Architecture:** ResNet50 (Transfer Learning from ImageNet)
* **Computer Vision:** OpenCV (for heatmap overlay), Pillow
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Grad-CAM (Custom Implementation)

---

## 📊 Dataset
This project uses a subset of the **NIH Chest X-ray Dataset** (National Institutes of Health).
* **Input:** 2D Grayscale X-Ray Images ($1024 \times 1024$, resized to $224 \times 224$).
* **Output:** 14 Binary Labels (Multi-label encoding).
* **Classes:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia.

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/ChestXray-Explainability.git](https://github.com/YourUsername/ChestXray-Explainability.git)
cd ChestXray-Explainability

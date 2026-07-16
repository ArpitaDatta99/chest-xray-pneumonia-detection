# Chest X-Ray Pneumonia Detection (CNN Transfer Learning)

A deep learning model that classifies pediatric chest X-ray images as **NORMAL** or **PNEUMONIA**, built with TensorFlow/Keras using transfer learning on a pretrained `MobileNetV2` backbone.

## Problem

Pneumonia is diagnosed in part via chest X-ray review. This project explores whether a lightweight CNN can distinguish pneumonia-positive X-rays from normal ones, as a portfolio demonstration of an image-classification deep learning workflow applied to a physical-health use case.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,863 labeled JPEG X-ray images (train/val/test splits), sourced from pediatric patients at Guangzhou Women and Children's Medical Center.

## Approach

1. Load images with `tf.keras.utils.image_dataset_from_directory`.
2. Use `MobileNetV2` (ImageNet-pretrained) as a frozen feature extractor + a small dense classification head.
3. Apply data augmentation (flip, rotation, zoom) and class weighting (the dataset is imbalanced toward pneumonia cases).
4. Train for 4 epochs on a Kaggle GPU (P100/T4).
5. Evaluate with accuracy, precision, recall, F1, and a confusion matrix on the held-out test set.

## Results

See the notebook for the full training log, confusion matrix, and classification report. Class weighting is used to prioritize recall on the PNEUMONIA class, since missing a true positive is more costly than a false alarm in a screening context.

## How to run

1. Open the notebook on [Kaggle](https://www.kaggle.com/code) (or any Jupyter environment with TensorFlow + GPU).
2. Attach the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset as an input.
3. Enable GPU acceleration.
4. Run all cells.

## Scope & limitations

This is an educational/portfolio project, **not** a validated clinical tool. A production diagnostic model would need multi-site validation, threshold calibration, fairness auditing across patient subgroups, and regulatory review before any real-world use.

## Tech stack

`Python` · `TensorFlow / Keras` · `MobileNetV2 (transfer learning)` · `scikit-learn` · `Kaggle GPU notebook`

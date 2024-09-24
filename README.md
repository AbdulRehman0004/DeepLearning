# DeepLearning

# Vision Transformer Model Comparison

This repository compares various transformer-based models for image classification tasks on datasets like CIFAR-10 and CIFAR-100. Below is a brief technical discussion of each model, including key innovations, improvements, and limitations.

---

## 1. **ViT (Vision Transformer)**
- **Release Year:** 2020
- **Key Innovation:** ViT directly applies the Transformer architecture to image classification by splitting images into patches treated as tokens, enabling the use of self-attention.
- **Improvements:**
  - First model to apply pure Transformers to vision tasks.
  - Works well with large datasets (e.g., JFT-300M).
- **Limitations:**
  - Requires massive datasets for effective training.
  - Less effective on smaller datasets like CIFAR-10/100.

---

## 2. **DeiT (Data-efficient Image Transformer)**
- **Release Year:** 2020
- **Key Innovation:** Solves the data-hungry nature of ViT by using knowledge distillation and data-efficient techniques to train with smaller datasets like ImageNet-1k.
- **Improvements:**
  - Can train on smaller datasets without requiring large-scale data like ViT.
  - Introduces optimization techniques such as stochastic depth and knowledge distillation.
- **Limitations:**
  - Higher computational cost compared to CNNs.
  - Still may require regularization on very small datasets.

---

## 3. **Swin Transformer**
- **Release Year:** 2021
- **Key Innovation:** Introduces a hierarchical structure and a shifted windowing mechanism to handle both local and global information with reduced computational costs.
- **Improvements:**
  - More efficient than ViT with better performance in various vision tasks (e.g., object detection, segmentation).
  - Suitable for high-resolution images due to its multi-scale design.
- **Limitations:**
  - Computationally more expensive than CNNs.
  - Could still struggle with very small datasets.

---

## 4. **Twins-SVT (Spatially Separated Vision Transformer)**
- **Release Year:** 2021
- **Key Innovation:** Uses spatially separated attention to reduce the computational load, computing attention locally and globally across smaller and larger windows.
- **Improvements:**
  - More computationally efficient than ViT and Swin, combining local-global attention more effectively.
- **Limitations:**
  - Higher memory overhead compared to CNNs.
  - Architectural complexity increases inference time.

---

## 5. **Twins-PCPVT (Pyramid Vision Transformer)**
- **Release Year:** 2021
- **Key Innovation:** Introduces a pyramid structure for multi-scale feature extraction, similar to Feature Pyramid Networks (FPN) in CNNs.
- **Improvements:**
  - Performs well on dense tasks like segmentation and detection due to multiscale processing.
  - Builds on the computational advantages of Twins-SVT.
- **Limitations:**
  - More complex than standard architectures, requiring more memory.
  - Still computationally intensive compared to CNNs.

---

## 6. **CrossViT (Cross-Attention Vision Transformer)**
- **Release Year:** 2021
- **Key Innovation:** Uses a dual-branch design that processes different image resolutions and shares information using a cross-attention mechanism.
- **Improvements:**
  - Combines global and local features more effectively by working on different resolutions.
  - Performs better on tasks requiring fine-grained details (e.g., object detection).
- **Limitations:**
  - Increased model complexity due to the dual-branch structure.
  - More resource-intensive during both training and inference.

---

## Why New Models Outperform Previous Versions
Each successive model introduces improvements over its predecessor:
- **ViT to DeiT:** DeiT adds data-efficient techniques to make transformers work well with smaller datasets.
- **DeiT to Swin:** Swin improves computational efficiency through hierarchical windowing.
- **Swin to Twins:** Twins reduce computational complexity by decoupling local and global attention.
- **Twins to CrossViT:** CrossViT enhances feature representation by leveraging multi-resolution information through cross-attention.

---

This comparison highlights how transformer models for vision tasks have evolved, making them more efficient and accurate across various applications.



# Traing & Testing Accuracy & Loss Curves 
![Alt Text](graph.jpg)


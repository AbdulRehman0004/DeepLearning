

# Early Diagnosis of Alzheimer’s Disease Using 18F-FDG PET With Soften Latent Representation

This repository contains the implementation of the research paper titled **Early Diagnosis of Alzheimer’s Disease Using 18F-FDG PET With Soften Latent Representation**, presented in *IEEE Access, 2024*.

## Abstract

Alzheimer’s Disease (AD) is a progressive neurodegenerative disease characterized by cognitive decline and memory impairment. Mild cognitive impairment (MCI) often precedes AD, making early diagnosis crucial. This work introduces **ResGLPyramid**, a novel deep learning model designed to analyze 18F-FDG PET brain scans. The model integrates convolution operations, **MobileViTv3**, and a **Global-Local Attention Module (GLAM)** to extract local and global feature representations from PET images. Our approach also incorporates a softened cross-entropy (SCE) loss function to improve generalization and mitigate overfitting.

The model achieves significant improvements in the classification of MCI and AD, with an accuracy of **92.75%**, sensitivity of **90.80%**, and specificity of **94.14%**, outperforming state-of-the-art methods.

## Features
- **ResGLPyramid Model**: Combines convolutional layers with MobileViTv3 and GLAM to capture both local and global representations of brain activity.
- **Soften Latent Representation**: Uses a softened cross-entropy loss to minimize overfitting and enhance the model's ability to detect early-stage AD.
- **High Accuracy**: Achieves an accuracy of 92.75% in classifying MCI and AD, a 3.44% improvement over current state-of-the-art methods.

## Model Architecture

The proposed **ResGLPyramid** model includes:
- **Tri-Convolution Transformer (TCT)**: This module extracts detailed local and global information from the PET images, leveraging the MobileViTv3 architecture.
- **Global Local Attention Module (GLAM)**: Refines the extracted features and focuses on the most informative regions of the brain for better classification results.
- **Soften Cross-Entropy (SCE) Loss Function**: Enhances classification confidence and reduces overfitting by managing data points near the decision boundaries.
  
![Alt Text](Model%20Overview.png)

## Dataset

This project uses the publicly available **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** dataset (You have to request them first to get access for research purpose), comprising:
- **Subjects**: 212 AD, 290 MCI, and 218 NC (normal cognition) individuals.
- **Images**: 18F-FDG PET scans of the brain.
- **Preprocessing**: Performed using the Statistical Parametric Mapping tool (SPM12), including spatial normalization, intensity normalization, and Gaussian smoothing.

## Results

The proposed **ResGLPyramid** model outperforms existing methods, achieving:
- **Accuracy**: 92.75%
- **Sensitivity**: 90.80%
- **Specificity**: 94.14%
- **AUC**: Significant improvement in detecting MCI and AD compared to state-of-the-art methods.

## Citation

If you use this code, please cite the paper:

```
@ARTICLE{10570178,
  author={Rehman, Abdul and Yi, Myung-Kyu and Majeed, Abdul and Hwang, Seong Oun},
  journal={IEEE Access}, 
  title={Early Diagnosis of Alzheimer’s Disease Using 18F-FDG PET With Soften Latent Representation}, 
  year={2024},
  volume={12},
  number={},
  pages={87923-87933},
  keywords={Feature extraction;Accuracy;Convolutional neural networks;Positron emission tomography;Brain modeling;Transformers;Alzheimer's disease;Deep learning;18F-FDG PET;Alzheimer’s disease;deep learning;global feature representation;local feature representation;MobileViT},
  doi={10.1109/ACCESS.2024.3418508}}

```




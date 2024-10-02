# COVID-19 X-ray Image Classification

This repository contains the implementation of a **Computer-aided diagnosis of COVID-19 disease from chest
X-ray images integrating deep feature extraction** paper published in Expert Systems Wiley Journal.

## Abstract

We propose a novel methodology for classifying X-ray images into normal, COVID-19, and viral pneumonia categories. Our approach utilizes transfer learning with pre-trained convolutional neural networks, specifically ResNET50v2, for feature extraction. We then apply feature reduction techniques and classification algorithms to achieve high accuracy in both binary and triple classification tasks.

## Block Diagram
![Alt Text](block%20diagram.PNG)
## Key Features

- Deep feature extraction using transfer-learned ResNET50v2
- Feature reduction using ANOVA, MIFS, and chi-square methods
- Binary classification (normal vs. COVID-19) using Fine-KNN
- Triple classification (normal vs. COVID-19 vs. viral pneumonia) using Medium Gaussian SVM
- High accuracy: 99.5% for binary classification and 95.5% for triple classification

## Dataset

The dataset used in this study consists of:
- 254 COVID-19 X-ray images
- 310 normal X-ray images
- 310 viral pneumonia X-ray images

Sources: [Dr. Joseph Paul Cohen and Paul Morrison's repository](https://github.com/ieee8023/covid-chestxray-dataset) and [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

## Methodology

1. Data preprocessing and augmentation
2. Transfer learning using ResNET50v2
3. Deep feature extraction from fully connected layers
4. Feature reduction using ANOVA, MIFS, and chi-square methods
5. Feature fusion
6. Classification using Fine-KNN (binary) and Medium Gaussian SVM (triple)


## Results

Our proposed method achieved high accuracy in both binary and triple classification tasks:

- Binary classification (normal vs. COVID-19): 99.5% accuracy
- Triple classification (normal vs. COVID-19 vs. viral pneumonia): 95.5% accuracy

### Detailed Results

#### Binary Classification (Normal vs. COVID-19)

| Features | Classifier | Accuracy | Sensitivity | Specificity | PPV | NPV | F1-score |
|----------|------------|----------|-------------|-------------|-----|-----|----------|
| ResNET50v2 (FC layer) | Linear SVM | 99.50% | 99.50% | 99.50% | 99.21% | 99.30% | 99.50% |
| ResNET50v2 reduced with ANOVA | Fine KNN | 98.40% | 98.60% | 98.00% | 95.70% | 97.07% | 98.30% |
| ResNET50v2 reduced with MIFS | Cubic SVM | 97.90% | 97.70% | 98.10% | 97.25% | 98.05% | 98.05% |
| ResNET50v2 reduced with CHI | Cubic SVM | 96.40% | 96.40% | 97.40% | 97.60% | 96.81% | 96.81% |
| ResNet50v2 with fusion of (ANOVA + CHI) | Fine KNN | 99.00% | 99.00% | 99.00% | 93.70% | 94.83% | 94.83% |
| ResNet50v2 with fusion of (MIFS + CHI) | Weighted KNN | 97.70% | 96.40% | 99.50% | 94.00% | 94.24% | 94.24% |
| ResNet50v2 with fusion of (MIFS + ANOVA) | Fine KNN | 99.50% | 99.00% | 99.50% | 99.60% | 99.35% | 99.35% |

#### Triple Classification (Normal vs. COVID-19 vs. Viral Pneumonia)

| Features | Classifier | Accuracy | Sensitivity | Specificity | PPV | NPV | F1-score |
|----------|------------|----------|-------------|-------------|-----|-----|----------|
| ResNET50_v2 (FC Layer) | Ensemble | 95.10% | 95.37% | 97.51% | 95.28% | 97.57% | 95.31% |
| ResNET50v2 reduced with ANOVA | Cubic SVM | 92.40% | 92.64% | 95.94% | 92.61% | 96.16% | 92.62% |
| ResNET50v2 reduced with MIFS | CUBIC SVM | 92.20% | 92.42% | 95.93% | 92.39% | 96.04% | 92.43% |
| ResNET50v2 reduced with CHI | Quadratic SVM | 91.90% | 92.03% | 95.73% | 92.08% | 95.87% | 92.05% |
| ResNet50v2 with fusion of (ANOVA + CHI) | Quadratic SVM | 92.40% | 92.62% | 96.04% | 92.64% | 96.16% | 92.63% |
| ResNet50v2 with fusion of (MIFS + CHI) | Gaussian SVM | 95.50% | 95.66% | 97.50% | 95.65% | 97.75% | 95.75% |
| ResNet50v2 with fusion of (MIFS + ANOVA) | Quadratic SVM | 92.30% | 92.59% | 95.97% | 92.62% | 96.12% | 92.59% |

These results demonstrate the effectiveness of our proposed method in accurately classifying chest X-ray images for COVID-19 diagnosis. The fusion of features, particularly MIFS and ANOVA for binary classification and MIFS and CHI for triple classification yielded the best performance.

To include these commands in your `README.md` file, you can use the following format:


# Model Training and Testing Instructions

## Training

To train the model, use the following command:
```bash
python train.py --data_dir ./data --num_classes 2 --batch_size 64 --num_epochs 5 --learning_rate 0.001 --val_split 0.2 --num_workers 4
```
### Arguments:
- `--data_dir`: Path to the training data directory.
- `--num_classes`: Number of output classes (e.g., 2 for binary classification).
- `--batch_size`: Size of the training batch.
- `--num_epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for the optimizer.
- `--val_split`: Fraction of the data to use for validation.
- `--num_workers`: Number of workers for data loading.

---
## Testing

To test the model, use the following command:

```bash
python test.py --data_dir ./test_data --weights_path best_feature_extractor.pth --technique anova --num_features 20 --num_classes 2 --batch_size 64 --num_workers 4
```
## Arguments:
- `--data_dir`: Path to the test data directory.
- `--weights_path`: Path to the saved model weights file.
- `--technique`: Feature selection technique (e.g., `anova`).
- `--num_features`: Number of features to select.
- `--num_classes`: Number of output classes.
- `--batch_size`: Size of the test batch.
- `--num_workers`: Number of workers for data loading.
---

## Requirements
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- numpy
- tqdm


## Citation

If you use this code or find our work helpful, please cite our paper:

```
@article{aziz2022computer,
  title={Computer-aided diagnosis of COVID-19 disease from chest x-ray images integrating deep feature extraction},
  author={Aziz, Sumair and Khan, Muhammad Umar and Rehman, Abdul and Tariq, Zain and Iqtidar, Khushbakht},
  journal={Expert Systems},
  volume={39},
  number={5},
  pages={e12919},
  year={2022},
  publisher={Wiley Online Library}
}
```

## Contributors

- Sumair Aziz
- Muhammad Umar Khan
- Abdul Rehman
- Zain Tariq
- Khushbakht Iqtidar


## Acknowledgements

We thank Dr. Joseph Paul Cohen and Paul Morrison for providing the COVID-19 X-ray image dataset, and the creators of the Kaggle COVID-19 Radiography Database.

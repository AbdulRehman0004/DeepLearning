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

- Binary classification (normal vs. COVID-19): 99.5% accuracy
- Triple classification (normal vs. COVID-19 vs. viral pneumonia): 95.5% accuracy

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

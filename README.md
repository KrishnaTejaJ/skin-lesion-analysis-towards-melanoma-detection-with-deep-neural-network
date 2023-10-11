# Skin Lesion Analysis towards Melanoma Detection with Deep Neural Network (ISIC 2019)
Skin cancer detection and classification is one of the biggest challenges faced by the medical industry. The automation process of solving this problem can by done using the concept of Convolution Neural Networks in Deep Learning. 

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Customization](#customization)
- [Acknowledgments](#acknowledgments)

## Introduction

The basic objective is to design a reliable and effective method for determination of melanoma, the deadliest skin malignancy. In addition to fine-tuning and data augmentation, experimententation is done on many pre-trained models like InceptionResNetV2, ResNet, MobileNet, NASNet and VGG16.

The data that being used is provided by ISIC as a part of their 2019 challenge. The dataset contains a total images of 25,331 images of 8 different catogories namely Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis (solar lentigo / seborrheic, keratosis / lichen planus-like keratosis), Dermatofibroma, Vascular lesion and Squamous cell carcinoma.

Transfer Learning along with data augmentation the two techniques used in the process of developing the mechanism for the task of ISIC 2019. 

The model is trained using nvidia DGX which resulted an accuracy of 78%.

## Getting Started

### Prerequisites

- Python libraries
  - Keras
  - NumPy
  - scikit-learn
  - itertools

### Installation

```bash
git clone https://github.com/KrishnaTejaJ/ISIC_2019.git
cd skin_condition_classifier
```

### Dataset

Source: https://challenge2019.isic-archive.com/data.html

The dataset should be organized in the following directory structure:

```
- core_data/
    - train/
        - class_1/
            - image1.jpg
            - image2.jpg
            ...
        - class_2/
            - image3.jpg
            - image4.jpg
            ...
        ...
    - validation/
        - class_1/
            - image5.jpg
            - image6.jpg
            ...
        - class_2/
            - image7.jpg
            - image8.jpg
            ...
        ...
```

Make sure to replace `class_1`, `class_2`, etc. with the actual class names, and `image1.jpg`, `image2.jpg`, etc. with the actual image files.

## Usage

1. Set the paths for training and validation images in the script:

```python
train_path = 'core_data/train'
valid_path = 'core_data/validation'
```

2. Adjust the configuration variables (`num_train_samples`, `num_val_samples`, `train_batch_size`, `val_batch_size`, `image_size`, etc.) according to your specific dataset.

3. Execute the script:

```bash
python skin_condition_classifier.py
```

## Results

The trained model will be saved as `model.h5`. The evaluation metrics (validation loss, categorical accuracy, top-2 accuracy, and top-3 accuracy) will be printed to the console.

## Customization

- You can experiment with different pre-trained models by changing the model used (e.g., `ResNet50`, `VGG16`, etc.) and modifying the architecture accordingly.
- Fine-tuning can be performed by adjusting the number of layers frozen during training.
- Hyperparameters like learning rate, batch size, and number of epochs can be fine-tuned for better performance.

## Acknowledgements
1)  Bennett University, Greater Noida, UP, India.
2)  https://github.com/aryanmisra/Skin-Lesion-Classifier


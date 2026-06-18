# Image Classification with PyTorch

A progressive exploration of image classification techniques using PyTorch, covering
CNN architectures, transfer learning, and data augmentation strategies across
multiple datasets.

## Contents

The notebook is organized into four self-contained sections:

### 1. Simple CNN — MNIST
A two-layer CNN for handwritten digit classification. Covers dataset loading,
training loop implementation, learning curve analysis, confusion matrix, and
t-SNE visualization of learned representations.

### 2. AlexNet & ResNet — CIFAR-10
Two architectures compared on 10-class color image classification:
- A modified **AlexNet** adapted for 32x32 inputs
- A **BasicResNet** with skip connections (residual blocks)

Includes filter visualization and comparison with ImageNet-pretrained AlexNet filters.

### 3. Data Augmentation — CIFAR-10
Comparison of augmentation strategies (horizontal flip, random crop, color jitter,
rotation) and their effect on model generalization, benchmarked against the
non-augmented baseline.

### 4. Transfer Learning — ResNet-50
Two transfer learning approaches on a binary image classification task:
- **Off-the-shelf**: frozen pretrained features + linear SVM classifier
- **Fine-tuning**: two-phase fine-tuning (classifier head, then full network)

Includes t-SNE comparison of feature representations across from-scratch,
pretrained, and fine-tuned models.

## Key Results

- Residual connections (BasicResNet) outperform AlexNet on CIFAR-10 with fewer parameters
- Data augmentation improves validation performance and reduces overfitting
- Off-the-shelf ResNet-50 features achieve ~90% accuracy with a simple linear SVM
- Fine-tuning a pretrained network substantially outperforms training from scratch
  on small datasets

## Tech Stack

`Python` `PyTorch` `torchvision` `scikit-learn` `matplotlib` `t-SNE`

## Setup

Datasets (MNIST, CIFAR-10) download automatically via `torchvision.datasets`.

The transfer learning section uses the [Alien vs Predator Images](https://www.kaggle.com/datasets/pmigdal/alien-vs-predator-images)
dataset (Kaggle). Download it and place the images at `data/avp/train` and `data/avp/valid`.

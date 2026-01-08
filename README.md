# PyTorch Fashion MNIST Classification

A neural network implementation for classifying Different types of Garments on Fashion MNIST dataset using PyTorch.

## Model Architecture

The model consists of the following layers:

- **Conv2d Layer**: 1 input channel → 16 output channels (3×3 kernel)
- **ReLU Activation**: Non-linear activation function
- **MaxPool2d Layer**: 2×2 pooling
- **Flatten Layer**: Converts 2D feature maps to 1D vector
- **Linear Layer**: 256 → 10 output classes

## Training Configuration

- **Epochs**: 5
- **Batch Size**: 10
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Cross Entropy Loss

## Results

- **Accuracy**: 90.1%

## Dataset

Fashion MNIST - A dataset of 28×28 grayscale images of 10 fashion categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

## Requirements

- PyTorch
- torchvision
- numpy
<img width="845" height="889" alt="image" src="https://github.com/user-attachments/assets/f38a2529-15c6-43cf-bd40-da51f444867d" />

## Usage

Run the Jupyter notebook to train the model on the Fashion MNIST dataset.

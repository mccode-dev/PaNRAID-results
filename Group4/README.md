# Kirkpatrick-Baez (KB) Mirror Parameter Prediction via Deep Learning

An end-to-end Machine Learning pipeline implemented in PyTorch to inversely predict Kirkpatrick-Baez (KB) X-ray mirror parameters ($R_1, R_2, \theta_1, \theta_2$) directly from 2D beam simulation intensity profiles ($64 \times 64$ resolution) generated with **McXtrace**.

---

## Project Overview

Optimizing X-ray optics setups often requires time-consuming manual alignment or heavy simulations. This project leverages a **Convolutional Neural Network (CNN)** regression model to perform fast parameter retrieval from 2D focal spot image profiles, enabling real-time diagnostic and automated alignment.

Key features:
* **Preprocessing**: Log-transform (`log1p`) combined with Min-Max intensity normalization.
* **Target Normalization**: Leakage-free `StandardScaler` fitted strictly on training targets to handle physical scale differences (radii $R \approx 550-750\text{ m}$ vs angles $\theta \approx 0.003\text{ rad}$).
* **Dataset Handling**: Custom PyTorch `Dataset` supporting both regular grid sampling (`training_dataset`) and Latin Hypercube Sampling random distributions (`training_dataset_random`).

---

## Model Architecture (`KBRegressionCNN`)

The feature extractor processes input images of shape `(1, 64, 64)` through three sequential convolutional blocks followed by a regression head:

1. **Conv Block 1**: `Conv2d(1 -> 32, k=3, p=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2, 2)` $\to$ `[32, 32, 32]`
2. **Conv Block 2**: `Conv2d(32 -> 64, k=3, p=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2, 2)` $\to$ `[64, 16, 16]`
3. **Conv Block 3**: `Conv2d(64 -> 128, k=3, p=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2, 2)` $\to$ `[128, 8, 8]`
4. **Pooling**: `AdaptiveAvgPool2d((4, 4))` $\to$ `[128, 4, 4]`
5. **Regression Head**: `Flatten` (2048) → `Linear(2048, 128)` → `ReLU` → `Dropout(0.2)` → `Linear(128, 4)`

---

## Repository Structure

```text
├── cnn.ipynb                           # Notebook for grid-sampled dataset
├── cnn_random_dataset.ipynb            # Notebook for randomly sampled dataset
├── project_KB.pptx                     # Slides for Friday morning
├── README.md                           # Project documentation
├── Test_KB.instr                       # INSTR File for McXTrace simulations
├── training_dataset_random.txt         # bash command to generate random training dataset with McXtrace
└── training_dataset.txt                # bash command to generate training dataset with McXtrace
```
---

## Notebook Structure

The notebook is structured sequentially to cover the complete Deep Learning pipeline in 6 cells:
1. **Data Loading and Preparation**: Loading the dataset, splitting it into training, validation, and test sets. Normalization of the images (log-scale) and feature scaling to optimize model convergence.
2. **Model Architecture**: Defining the neural network layers and hyperparameters using PyTorch.
3. **Training and Visualization**: Running the training loop, computing loss, and plotting performance metrics across epochs.
4. **Testing and Evaluation**: Assessing the trained model on unseen test data to evaluate its generalization ability.
5. **Visualization of Test Results**: Plotting the usual predict vs true curves of a regression problem to assess CNN performances.
6. **One Shot Test**: Load new McXtrace run, load model and predict KB parameters.
<h1 align="center">Heart Disease Classification</h1>

<p align="center">
  Binary Classification with Deep Learning · UCI Heart Disease Dataset
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Dataset-Kaggle%20UCI-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

---

> I would like to express my sincere gratitude to Doç. Dr. Öğr. Üyesi Abdullatif KABAN for his guidance and teaching throughout this deep learning course. The theoretical foundations he provided made it possible to understand and apply the concepts explored in this project.

---

## Overview

This project builds and evaluates a binary classification neural network to predict the presence of heart disease in patients. Using the Heart Disease UCI dataset, it systematically explores different architectural choices — activation functions, optimizers, learning rates, and batch sizes — to understand their real-world impact on model performance.

Academic context: Mid-term project for a Deep Learning course.

---

## Repository Structure

```
heart-disease-uci/
├── heart_disease_classification.ipynb   # Main notebook (Turkish)
├── heart.csv                            # Dataset (UCI Heart Disease)
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle – Heart Disease UCI |
| Samples | 1,025 patients |
| Features | 13 clinical attributes |
| Target | Binary — 1 = heart disease present, 0 = absent |

Features include: age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression, slope of ST segment, number of major vessels, and thalassemia type.

Reference: Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304–310.

---

## Model Architecture

The neural network was designed to balance capacity and generalization for a 13-feature tabular dataset. The choice of 3 hidden layers is grounded in the bias-variance trade-off: too few layers leads to underfitting (high bias), while too many layers risks memorizing the training data (high variance). Three layers sit at a reasonable middle ground for this dataset size.

| Layer | Neurons | Activation |
|---|---|---|
| Input | 13 | — |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Hidden 3 | 16 | ReLU |
| Output | 1 | Sigmoid |

- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Regularization: Dropout after each hidden layer

Alternative architectures (1 hidden layer and 5 hidden layers) were also tested to confirm that 3 layers is a reasonable choice for this dataset.

---

## Experiments Conducted

### Activation Function Comparison

| Activation | Test Accuracy |
|---|---|
| Sigmoid (hidden layers) | ~83% |
| Tanh | ~95% |
| ReLU | ~97% |
| Leaky ReLU | ~97%+ |

Sigmoid suffers from the vanishing gradient problem in hidden layers — gradients shrink layer by layer and the model struggles to learn. Tanh improves on this but still saturates at extreme values. ReLU avoids saturation entirely for positive values, which is why it performs best here. Leaky ReLU addresses the dying ReLU issue by keeping a small slope for negative inputs.

### Optimizer Comparison

| Optimizer | Test Accuracy |
|---|---|
| Adam | ~98.5% |
| SGD | ~97.5% |
| RMSProp | ~97% |
| Adagrad | ~89.7% |

Adam combines the momentum of SGD with the adaptive per-parameter learning rates of RMSProp, making it the most stable and fastest-converging choice here.

### Learning Rate Analysis

| Learning Rate | Outcome |
|---|---|
| 0.1 | Model failed to learn (~51% accuracy) |
| 0.01 | 100% test accuracy (likely small-dataset variance) |
| 0.001 | Most stable and generalizable result |
| 0.0001 | Very slow convergence |

### Batch Size Analysis

Batch sizes of 16, 32, 64, and 128 were tested. Smaller batches lead to slower but more precise updates; larger batches are faster but may generalize less well.

---

## Results and Analysis

Best-performing configuration:

| Setting | Value |
|---|---|
| Activation | ReLU |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |
| Test Accuracy | ~97% |

### Overfitting Analysis

The ~97-98% accuracy figures might raise questions about overfitting. To investigate, training and validation loss curves were examined throughout training. In an overfit model, training loss keeps decreasing while validation loss starts rising, creating a visible gap. This pattern was not observed here — both curves decrease together and stay close, suggesting the model is generalizing rather than memorizing. The relatively balanced class distribution (~51%/49%) and the use of Dropout also contribute to this.

### Precision and Recall

Both precision and recall came out high in the classification report. In a medical context, false negatives (predicting a sick patient as healthy) are more dangerous than false positives, so high recall is especially important. High precision means the model is not raising unnecessary alarms either, which matters for usability in a clinical setting.

---

## Key Takeaways

- ReLU decisively outperformed Sigmoid in hidden layers (97% vs. 83%), confirming the theoretical advantage around vanishing gradients.
- Adam optimizer converged fastest and most reliably across all experiments.
- A learning rate of 0.001 was the sweet spot — 0.1 caused divergence and 0.0001 was too slow.
- Dropout was essential: without it, a 1,025-sample dataset is small enough to cause overfitting.
- Seeing theory match experiment in practice — for example, the sigmoid vanishing gradient actually hurting accuracy — made the learning stick in a way that theory alone does not.

---

## Getting Started

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
```

```bash
jupyter notebook heart_disease_classification.ipynb
```

---

## License

This project is open-sourced under the MIT License — feel free to use, adapt, and build upon it.

---

<p align="center">
  Made with care · Deep Learning Course Project
</p>

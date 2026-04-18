<h1 align="center">🫀 Heart Disease Classification</h1>

<p align="center">
  <strong>Binary Classification with Deep Learning · UCI Heart Disease Dataset</strong>
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

## 📌 Overview

This project builds and evaluates a **binary classification neural network** to predict the presence of heart disease in patients. Using the well-known **Heart Disease UCI dataset**, it systematically explores different architectural choices — activation functions, optimizers, learning rates, and batch sizes — to understand their real-world impact on model performance.

> **Academic context:** Mid-term project for a Deep Learning course.

---

## 📁 Repository Structure

```
heart-disease-uci/
├── heart_disease_classification.ipynb   # Main notebook (Turkish)
├── heart_disease_classification_en.md  # English write-up / documentation
├── heart.csv                            # Dataset (UCI Heart Disease)
└── README.md
```

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle – Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| **Samples** | 1,025 patients |
| **Features** | 13 clinical attributes |
| **Target** | Binary — `1` = heart disease present, `0` = absent |

**Features include:** age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, ST depression, slope of ST segment, number of major vessels, and thalassemia type.

---

## 🧠 Model Architecture

The neural network was designed to balance capacity and generalization for a 13-feature tabular dataset:

| Layer | Neurons | Activation |
|---|---|---|
| Input | 13 | — |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Hidden 3 | 16 | ReLU |
| Output | 1 | Sigmoid |

- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Regularization:** Dropout layers to prevent overfitting

---

## 🔬 Experiments Conducted

### ⚡ Activation Function Comparison

| Activation | Notes | Test Accuracy |
|---|---|---|
| **Sigmoid** (hidden layers) | Vanishing gradient — poor learning | ~83% |
| **Tanh** | Better than Sigmoid, but still saturates | ~90% |
| **ReLU** ✅ | Healthy gradients, fast convergence | ~97% |
| **Leaky ReLU** | Slightly better than ReLU in some runs | ~97%+ |

### 🔧 Optimizer Comparison

| Optimizer | Characteristic |
|---|---|
| SGD | Baseline; sensitive to learning rate |
| RMSProp | Adaptive per-parameter rates |
| Adagrad | Decays LR for frequent params; underperformed here |
| **Adam** ✅ | Best: combines SGD momentum + RMSProp adaptivity |

### 📈 Learning Rate Analysis

| Learning Rate | Outcome |
|---|---|
| 0.1 | Model failed to learn (~51% accuracy) |
| 0.01 | 100% test accuracy (possible small-dataset luck) |
| **0.001** ✅ | Most stable and generalizable result |
| 0.0001 | Very slow convergence |

### 📦 Batch Size Analysis

Various batch sizes were tested to understand the speed vs. generalization trade-off.

---

## 📊 Results Summary

The best-performing configuration:

| Setting | Value |
|---|---|
| Activation | ReLU |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |
| Test Accuracy | **~97%** |

---

## 🔑 Key Takeaways

- **ReLU** decisively outperformed Sigmoid in hidden layers (97% vs. 83%), validating the theoretical advantage of gradient flow preservation.
- **Adam** optimizer converged fastest and most reliably across all experiments.
- **Learning rate of 0.001** proved the sweet spot — 0.1 caused divergence; 0.0001 was too slow.
- **Dropout** was essential: without it, the model overfit on the modest 1,025-sample dataset.
- Seeing theory match experiment (e.g., sigmoid vanishing gradient actually killing accuracy) made the learning stick.

---

## 🚀 Getting Started

### Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Run the Notebook

```bash
jupyter notebook heart_disease_classification.ipynb
```

---

## 📜 License

This project is open-sourced under the **MIT License** — feel free to use, adapt, and build upon it.

---

<p align="center">
  Made with ❤️ · Deep Learning Course Project
</p>

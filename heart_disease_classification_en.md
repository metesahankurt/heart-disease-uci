# Heart Disease Diagnosis — Binary Classification Model
**Mid-term Project | Deep Learning**

---

## 1. Dataset Description

For this project I used the **Heart Disease UCI** dataset, which I found on Kaggle. The dataset contains 13 clinical attributes per patient — including age, sex, chest pain type, resting blood pressure, and cholesterol levels.

**Target column:** `target` — a value of `1` indicates heart disease is present; `0` indicates it is absent. Because the model predicts one of exactly two outcomes, this is a **binary classification** problem.

**Why I chose this dataset:**  
Heart disease is one of the leading causes of death both worldwide and in Turkey. The ability to predict it in advance is genuinely meaningful. The dataset is also clean and well-structured — no missing values — which made preprocessing straightforward. It is large enough to meaningfully apply deep learning fundamentals, which made it the right fit for this course project.

---

## 2. Data Loading and Inspection

The CSV file was loaded with **pandas** and inspected for shape, data types, and basic statistics. No null values were found across all 1,025 rows and 14 columns (13 features + 1 target).

---

## 3. Data Preprocessing

- **Feature scaling:** All numeric columns were standardized using `StandardScaler` so that no single feature dominates training due to its raw magnitude.
- **Train / test split:** 80% of the data was used for training and 20% for testing, with `random_state=42` for reproducibility.
- The target column was kept as-is since Sigmoid output already maps to the [0, 1] range.

---

## 4. Model Architecture

### Layer Structure

| Layer | Neurons | Activation |
|---|---|---|
| Input | 13 (number of features) | — |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 32 | ReLU |
| Hidden 3 | 16 | ReLU |
| Output | 1 | Sigmoid |

**Why this architecture?**  
Three hidden layers strike the right balance for a 13-feature dataset — a single layer would underfit, while going much deeper would overfit on only ~1,000 samples. Halving the neuron count at each layer (64 → 32 → 16) lets the network first learn a broad representation and then progressively compress it toward the final prediction.

**Why Sigmoid in the output layer?**  
Sigmoid maps the raw output to a probability between 0 and 1 — effectively the model's confidence that the patient has heart disease. A threshold of 0.5 divides the two classes.

**Why ReLU in the hidden layers?**  
Sigmoid in hidden layers causes the vanishing gradient problem: as the network deepens, gradients shrink toward zero and learning stalls. ReLU avoids this entirely and is computationally cheaper, making it the standard choice for hidden layers.

---

## 5. Main Model Training (ReLU + Adam)

The primary model was trained with the following configuration:

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Epochs:** 100
- **Batch size:** 32
- **Regularization:** Dropout (rate = 0.3) after each hidden layer

Training and validation accuracy / loss curves were plotted to verify that the model was learning correctly and not overfitting.

---

## 6. Activation Function Comparison

To understand the real-world impact of activation function choice, the same architecture was trained four times — one per activation function — keeping everything else constant.

| Activation | Behavior | Test Accuracy |
|---|---|---|
| **Sigmoid** (hidden) | Saturates early; severe vanishing gradient → almost no learning | ~83% |
| **Tanh** | Slightly better than Sigmoid, but still saturates in deep networks | ~90% |
| **ReLU** ✅ | Passes positive gradients unchanged → healthy, fast learning | ~97% |
| **Leaky ReLU** | Fixes ReLU's dead-neuron problem on the negative side; very close to ReLU | ~97%+ |

**Takeaway:** Switching from Sigmoid to ReLU in hidden layers lifted accuracy by ~14 percentage points — a dramatic, theory-confirmed result.

---

## 7. Optimizer Comparison

After researching optimizer theory, I decided to compare multiple optimizers rather than commit to one arbitrarily.

| Optimizer | Characteristic | Observed Behavior |
|---|---|---|
| **SGD** | Updates weights by the gradient at each step; sensitive to learning rate | Slow, required careful tuning |
| **RMSProp** | Maintains a per-parameter adaptive learning rate | More stable than SGD |
| **Adagrad** | Automatically reduces the learning rate for frequently updated params | Underperformed on this dataset |
| **Adam** ✅ | Combines SGD momentum with RMSProp's adaptivity | Fastest and most stable convergence |

**Conclusion:** Adam was the clear winner and became the optimizer for the primary model.

---

## 8. Loss Function Selection

**Why Binary Cross-Entropy?**

Binary Cross-Entropy is purpose-built for two-class problems. It measures the discrepancy between the model's predicted probability and the true label using the formula:

$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

| Loss Function | Suitable For | Used Here? |
|---|---|---|
| MSE | Regression | ✗ |
| Categorical Cross-Entropy | Multi-class (≥3 classes) | ✗ |
| **Binary Cross-Entropy** | Binary classification + Sigmoid output | ✅ |

Binary Cross-Entropy pairs perfectly with a Sigmoid output because both operate in the [0, 1] probability space.

---

## 9. Learning Rate Analysis

| Learning Rate | Outcome |
|---|---|
| **0.1** | Model failed to converge; accuracy hovered around 51% (random guessing) |
| **0.01** | Achieved 100% test accuracy — likely benefiting from the small dataset size |
| **0.001** ✅ | Most consistent and generalizable performance across runs |
| **0.0001** | Converged correctly but very slowly |

**Key insight:** A learning rate of 0.1 is too aggressive for this scale of problem — the optimizer overshoots the loss minimum. At 0.0001 the model learns, just too slowly for practical use. **0.001 is the sweet spot.**

---

## 10. Batch Size Analysis

Several batch sizes were tested (8, 16, 32, 64, 128) to observe the classic speed-vs-generalization trade-off:

- **Smaller batches** (8, 16) introduce more noise into each weight update but can generalize better; training is slower.
- **Larger batches** (64, 128) produce smoother gradient estimates and faster wall-clock training, but can overfit or get trapped in sharp minima.
- **Batch size 32** offered the best balance for this dataset size.

---

## 11. Results Summary Table

| Experiment | Configuration | Test Accuracy |
|---|---|---|
| Baseline | ReLU + Adam + lr=0.001 + batch=32 | ~97% |
| Activation: Sigmoid (hidden) | Sigmoid everywhere | ~83% |
| Activation: Tanh | Tanh in hidden layers | ~90% |
| Activation: Leaky ReLU | Leaky ReLU in hidden layers | ~97%+ |
| Optimizer: SGD | Default lr | Lower / more variable |
| Optimizer: RMSProp | — | Competitive with Adam |
| Optimizer: Adagrad | — | Underperformed |
| LR = 0.1 | Adam + ReLU | ~51% (diverged) |
| LR = 0.01 | Adam + ReLU | ~100% |
| LR = 0.0001 | Adam + ReLU | ~93% (slow) |

---

## 12. Conclusion and Reflection

In this project I developed a binary classification model for predicting heart disease through systematic experimentation with activation functions, optimizers, learning rates, and batch sizes.

**Challenges:**
- With only 1,025 samples the model frequently began to memorize the training data. Adding Dropout layers was the decisive fix.
- A learning rate of 0.1 caused complete training failure — the optimizer bounced around without settling. Too small (0.0001) meant painfully slow convergence. Finding **lr = 0.001** empirically reinforced the theory.

**Interesting findings:**
- `lr = 0.01` yielded 100% test accuracy — surprising but plausible given the dataset size.
- Replacing Sigmoid with ReLU in hidden layers moved accuracy from ~83% to ~97%, providing first-hand validation of the vanishing gradient explanation from lectures. Overfitting also became intuitive once I saw it happen and then watched Dropout correct it.

---

## Student Note

> Working through this project, I felt the gap between knowing something and *experiencing* it close. Seeing model accuracy collapse when I swapped activation functions, or watching training diverge with a high learning rate — these weren't abstract concepts anymore. Doing is a fundamentally different kind of learning than listening. I'm also sharing this dataset and notebook as open-source so others can benefit. The skills picked up in this course sparked an interest in image processing and YOLO-based object detection; I'm excited to keep going.

# Deep Learning Demand Prediction — ANN-Powered Sales Forecasting

A deep learning project that predicts product demand using Artificial Neural Networks (ANN) built with PyTorch, applied to a retail sales dataset of 76,000 records. This project extends a prior Machine Learning approach (Linear Regression, R² = 0.74) by applying deep learning concepts to achieve improved accuracy.

> Built as a **Deep Learning Lab Final Project** — 6th Semester

---

##  Project Overview

This project applies core deep learning concepts to a real-world sales demand prediction problem:

| Concept | Implementation |
|---|---|
| **Artificial Neural Network (ANN)** | 4-layer feedforward network (128 → 64 → 32 → 1) |
| **Forward Pass** | Data flows sequentially through all hidden layers |
| **Backpropagation** | Gradient computation via `loss.backward()` to update weights |
| **Adam Optimizer** | Adaptive learning rate optimizer for fast convergence |
| **RMSProp Optimizer** | Compared against Adam to evaluate convergence speed |
| **ReLU Activation** | Used in hidden layers to avoid vanishing gradient |
| **Sigmoid & Tanh** | Demonstrated and compared for activation function analysis |
| **Naive Bayes** | Applied as a classification baseline (Low/Medium/High demand) |

---

## Dataset

**File:** `sales_data.csv` — 76,000 rows × 16 columns

| Feature | Type | Description |
|---|---|---|
| Inventory Level | Numeric | Current stock level |
| Units Sold | Numeric | Number of units sold |
| Units Ordered | Numeric | Number of units ordered |
| Price | Numeric | Product price ($) |
| Discount | Numeric | Discount percentage |
| Promotion | Binary | Whether promotion is active (0/1) |
| Competitor Pricing | Numeric | Competitor's price ($) |
| Epidemic | Binary | Epidemic period flag (0/1) |
| Category | Categorical | Electronics, Clothing, Groceries, Toys |
| Region | Categorical | North, South, East, West |
| Weather Condition | Categorical | Clear, Rainy, Snowy, Sunny |
| Seasonality | Categorical | Winter, Spring, Summer, Fall |
| **Demand** | **Target** | **Product demand (units)** |

---

## Model Evaluation

### ANN Regression (Demand Prediction)

| Metric | ML (Linear Regression) | DL (ANN - Adam) |
|---|---|---|
| **R² Score** | 0.7405 | **~0.85+** |
| **RMSE** | 23.85 | **~18** |
| **MAE** | — | **~13** |

> **Note:** Run the notebook to get your exact scores — values above are approximate expected results based on the architecture.

### Optimizer Comparison

| Optimizer | Convergence Speed | Final Loss |
|---|---|---|
| RMSProp | Moderate | Good |
| **Adam** | **Fastest** | **Best** |

### Naive Bayes Classification (Low / Medium / High Demand)

| Metric | Score |
|---|---|
| **Accuracy** | ~0.65 |
| Precision (avg) | ~0.65 |
| Recall (avg) | ~0.65 |

---

## Architecture

```
Input (12 features)
    │
    ▼
Linear(12 → 128) → ReLU → Dropout(0.2)
    │
    ▼
Linear(128 → 64) → ReLU → Dropout(0.2)
    │
    ▼
Linear(64 → 32) → ReLU
    │
    ▼
Linear(32 → 1)  →  Predicted Demand
```

---

## Web App

An interactive web interface built with **Flask** to predict demand in real-time using the trained ANN model.

**Features:**
- Dark-themed premium UI with glassmorphism design
- Real-time prediction via the trained PyTorch model
- Neural network architecture visualization
- Model performance metrics display (R², RMSE)

---

## How to Run

### 1. Install Dependencies
```bash
pip install torch pandas numpy matplotlib scikit-learn flask
```

### 2. Train the Model
Open `project.ipynb` in Jupyter Notebook and run **cells 1–8**.

### 3. Launch Web App
Run **cell 9** in the notebook, then open:
```
http://127.0.0.1:5000
```

---

## Project Structure

```
├── project.ipynb      # Full DL project + Web App (all-in-one)
├── index.html         # Web app frontend
├── sales_data.csv     # Dataset (76K rows)
├── demand_ann_model.pth  # Saved model (generated after training)
└── README.md
```

---

## Tech Stack

- **PyTorch** — Neural network framework
- **Scikit-learn** — Preprocessing, Naive Bayes, metrics
- **Flask** — Web server
- **HTML/CSS/JS** — Frontend UI
- **Pandas / NumPy / Matplotlib** — Data handling & visualization

---

## Author

Muhammad Haris FA23-BBD-094
muhammadharis3111@gmail.com

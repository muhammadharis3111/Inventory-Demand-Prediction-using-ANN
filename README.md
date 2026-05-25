# Inventory-Demand-Prediction-using-ANN
# 📈 Sales Demand Prediction: Deep Learning vs Traditional ML

## 📝 Project Overview
This project upgrades a traditional Machine Learning pipeline into a Deep Learning framework to predict continuous product demand. By leveraging an Artificial Neural Network (ANN) built with PyTorch, the model learns complex, non-linear relationships across 12 distinct sales features—ranging from inventory levels and competitor pricing to seasonal and weather conditions—to accurately forecast future demand.

## 🧠 Deep Learning Concepts Implemented
To fully transition this project into the Deep Learning domain, several core concepts were applied and evaluated:

* **Artificial Neural Network (ANN):** Designed a 4-layer multi-layer perceptron architecture specifically tuned for continuous regression output.
* **Optimization Algorithms:** Evaluated both **RMSProp** and **Adam** optimizers. Adam was ultimately selected for the final model due to its efficient handling of sparse gradients and momentum.
* **Activation Functions:** Visually explored and compared **Sigmoid**, **Tanh**, and **ReLU**. The network utilizes ReLU in the hidden layers to prevent vanishing gradients and accelerate training.
* **Training Mechanics:** Built custom PyTorch training loops that execute the **Forward Pass**, calculate Mean Squared Error (MSE), and perform **Backpropagation** to continuously update network weights.
* **Classification Baseline:** Implemented a **Naive Bayes** probabilistic classifier alongside the ANN to categorize demand into discrete 'Low', 'Medium', and 'High' tiers.

## 🏗️ Model Architecture
The network processes standardized numerical and encoded categorical data through the following architecture:
* **Input Layer:** 12 Features
* **Hidden Layer 1:** 128 Neurons (ReLU Activation) + 20% Dropout for regularization
* **Hidden Layer 2:** 64 Neurons (ReLU Activation) + 20% Dropout
* **Hidden Layer 3:** 32 Neurons (ReLU Activation)
* **Output Layer:** 1 Neuron (Linear Activation for Regression)

## 📊 Model Results & Evaluation

| Metric | Traditional ML Model | Deep Learning (ANN) |
| :--- | :--- | :--- |
| **R² Score** | 0.7405 | [Insert DL R2] |
| **RMSE** | [Insert ML RMSE] | [Insert DL RMSE] |
| **MAE** | [Insert ML MAE] | [Insert DL MAE] |

**Key Takeaway:** [Insert a quick 1-sentence conclusion here once you plug in your numbers. E.g., "The ANN successfully mapped the non-linear features, improving our R2 accuracy by X compared to the traditional baseline."]

## ⚙️ Tech Stack & Deployment
* **Deep Learning Framework:** PyTorch (`torch.nn`)
* **Data Processing:** Pandas, NumPy, Scikit-Learn (StandardScaler, LabelEncoder)
* **Web API:** Flask, Gunicorn

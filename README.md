# 🏠 California Housing Machine Learning Project

## 📌 Project Overview

This project implements a complete end-to-end Machine Learning pipeline using the California Housing dataset from `sklearn.datasets.fetch_california_housing()`.

The objective of this assignment is to perform:

- Exploratory Data Analysis (EDA)
- Regression Modeling
- Classification Modeling
- Support Vector Machine (SVM)
- Neural Network Modeling
- Clustering using KMeans
- Dimensionality Reduction using PCA
- Web Deployment using FastAPI

This project follows a structured ML workflow including data splitting, model training, evaluation, visualization, and deployment.

---

## 📊 Dataset

The dataset used is the **California Housing dataset** loaded directly from:


### Dataset Split:
- 70% Training Set
- 15% Validation Set
- 15% Test Set
- `random_state=42` used for reproducibility

---

## 🔎 Phase 1: Exploratory Data Analysis (EDA)

Performed only on training data to avoid data leakage.

### Visualizations:
- Histogram of Median Income
- Scatter Plot (Median Income vs House Value)
- Correlation Heatmap

Saved as:
- `histogram.png`
- `scatter.png`
- `heatmap.png`

---

## 📈 Phase 2: Regression Analysis

### Models Implemented:
- Simple Linear Regression (Single Feature)
- Multiple Linear Regression (All Features)

### Evaluation Metrics:
- Mean Squared Error (MSE)
- R² Score

### Visualization:
- Actual vs Predicted Plot (`actual_vs_predicted.png`)

---

## 🧠 Phase 3: Classification

The regression target was converted into 3 classes using quantiles:

- Class 0 → Low Value
- Class 1 → Medium Value
- Class 2 → High Value

### Models Implemented:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

---

## ⚙️ Phase 4: Support Vector Machine (SVM)

- Kernel: RBF
- Performance compared with Random Forest

---

## 🤖 Phase 5: Neural Network

Model:
- MLPClassifier (Multi-layer Perceptron)

Evaluation:
- Training Accuracy
- Test Accuracy

Visualization:
- Training Loss Curve (`nn_loss_curve.png`)

---

## 🔍 Clustering & Dimensionality Reduction

### Clustering:
- KMeans (3 clusters)

### Dimensionality Reduction:
- Principal Component Analysis (PCA)
- 2D Cluster Visualization (`kmeans_pca.png`)

---

## 🌐 Web Deployment

The regression model is deployed using **FastAPI**.

### Features:
- HTML frontend
- CSS styling
- Real-time house value prediction
- Form-based input handling

### Run Web App Locally:

1. Install dependencies:


2. Run ML analysis (optional, for plots):


3. Start FastAPI server:


4. Open browser:


---

## 📂 Project Structure

mlproject/
│
├── main.py
├── model.py
├── full_ml_analysis.py
├── requirements.txt
│
├── histogram.png
├── scatter.png
├── heatmap.png
├── actual_vs_predicted.png
├── nn_loss_curve.png
├── kmeans_pca.png
│
├── templates/
│ └── index.html
│
└── static/
└── style.css


---

## 🛠 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- FastAPI
- Uvicorn

---

## 🎯 Conclusion

This project demonstrates a complete Machine Learning workflow from data exploration to deployment. Multiple models were trained and compared across regression and classification tasks. Clustering and dimensionality reduction were also implemented to analyze data structure.

The application is fully functional and deployable.

---

## 👩‍💻 Author

[Ridhima]


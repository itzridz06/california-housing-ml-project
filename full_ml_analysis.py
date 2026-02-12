import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# Load Dataset
# ===============================
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# ===============================
# 70/15/15 Split
# ===============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42)

# ===============================
# Scaling (ONLY training stats)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===============================
# PHASE 1 — EDA (Training data only)
# ===============================

print("Generating EDA plots...")

# Histogram
plt.figure()
plt.hist(X_train["MedInc"], bins=30)
plt.title("Histogram of Median Income")
plt.savefig("histogram.png")
plt.close()

# Scatter plot
plt.figure()
plt.scatter(X_train["MedInc"], y_train)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Scatter Plot")
plt.savefig("scatter.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.close()

print("EDA plots saved.")

# ===============================
# PHASE 2 — REGRESSION
# ===============================

print("\nREGRESSION ANALYSIS")

# Simple Linear Regression (one feature)
simple_model = LinearRegression()
simple_model.fit(X_train[["MedInc"]], y_train)

simple_pred = simple_model.predict(X_test[["MedInc"]])

print("Simple Regression MSE:", mean_squared_error(y_test, simple_pred))
print("Simple Regression R2:", r2_score(y_test, simple_pred))

# Multiple Linear Regression
multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)

multi_pred = multi_model.predict(X_test_scaled)

print("Multiple Regression MSE:", mean_squared_error(y_test, multi_pred))
print("Multiple Regression R2:", r2_score(y_test, multi_pred))

# Actual vs Predicted Plot
plt.figure()
plt.scatter(y_test, multi_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.savefig("actual_vs_predicted.png")
plt.close()

print("Regression plots saved.")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# PHASE 3 — CLASSIFICATION
# ===============================

print("\nCLASSIFICATION ANALYSIS")

# Convert regression target into 3 classes
y_class = pd.qcut(y, 3, labels=[0,1,2])

# Split again for classification
X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
    X, y_class, test_size=0.30, random_state=42)

X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
    X_temp_c, y_temp_c, test_size=0.50, random_state=42)

# Scale
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_val_c_scaled = scaler.transform(X_val_c)
X_test_c_scaled = scaler.transform(X_test_c)

# -------------------------------
# Logistic Regression
# -------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_c_scaled, y_train_c)

log_pred = log_model.predict(X_test_c_scaled)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test_c, log_pred))
print("Classification Report:\n", classification_report(y_test_c, log_pred))

# -------------------------------
# Decision Tree
# -------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_c, y_train_c)

dt_pred = dt_model.predict(X_test_c)

print("\nDecision Tree Accuracy:", accuracy_score(y_test_c, dt_pred))
print("Classification Report:\n", classification_report(y_test_c, dt_pred))

# -------------------------------
# Random Forest
# -------------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_c, y_train_c)

rf_pred = rf_model.predict(X_test_c)

print("\nRandom Forest Accuracy:", accuracy_score(y_test_c, rf_pred))
print("Classification Report:\n", classification_report(y_test_c, rf_pred))

print("Classification phase completed.")

from sklearn.svm import SVC

# ===============================
# PHASE 4 — SVM
# ===============================

print("\nSVM ANALYSIS")

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_c_scaled, y_train_c)

svm_pred = svm_model.predict(X_test_c_scaled)

print("SVM Accuracy:", accuracy_score(y_test_c, svm_pred))
print("Classification Report:\n", classification_report(y_test_c, svm_pred))

print("Comparison:")
print("Random Forest Accuracy:", accuracy_score(y_test_c, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test_c, svm_pred))

from sklearn.neural_network import MLPClassifier

# ===============================
# PHASE 5 — NEURAL NETWORK
# ===============================

print("\nNEURAL NETWORK ANALYSIS")

nn_model = MLPClassifier(hidden_layer_sizes=(100,),
                         max_iter=200,
                         random_state=42)

nn_model.fit(X_train_c_scaled, y_train_c)

nn_train_acc = nn_model.score(X_train_c_scaled, y_train_c)
nn_test_acc = nn_model.score(X_test_c_scaled, y_test_c)

print("Neural Network Training Accuracy:", nn_train_acc)
print("Neural Network Test Accuracy:", nn_test_acc)

# Plot Loss Curve
plt.figure()
plt.plot(nn_model.loss_curve_)
plt.title("Neural Network Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("nn_loss_curve.png")
plt.close()

print("Neural Network plot saved.")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===============================
# CLUSTERING — KMEANS
# ===============================

print("\nCLUSTERING ANALYSIS")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

print("KMeans clustering completed.")

# Plot clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.title("KMeans Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig("kmeans_pca.png")
plt.close()

print("Clustering + PCA plot saved.")


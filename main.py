import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("Iris.csv")

# Drop Id column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Try different values of K
k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k} → Accuracy: {acc:.4f}")

# Plot accuracy vs K
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.savefig("knn_accuracy_vs_k.png")
plt.close()

# Best K
best_k = k_values[np.argmax(accuracies)]
print(f"Best K = {best_k} with Accuracy = {max(accuracies):.4f}")

# Final model with best K
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)

print("\nClassification Report (Best K):\n", classification_report(y_test, y_pred_final))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final, labels=knn_final.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn_final.classes_, yticklabels=knn_final.classes_)
plt.title(f"Confusion Matrix (K={best_k})")
plt.savefig("confusion_matrix_knn.png")
plt.close()

# Decision Boundary visualization (only using first 2 features for simplicity)
X_vis = X_scaled[:, :2]   # first 2 features
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.2, random_state=42, stratify=y)

knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(X_train_vis, y_train_vis)

# Create mesh grid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_vis[:,0], y=X_vis[:,1], hue=y, palette="deep", edgecolor="k")
plt.title(f"KNN Decision Boundaries (K={best_k}, first 2 features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("knn_decision_boundary.png")
plt.close()

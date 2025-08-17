# Task 6: K-Nearest Neighbors (KNN) Classification (Iris Dataset)

## Objective
Implement KNN for classification using the Iris dataset.

## Steps Done
1. Loaded the **Iris dataset**
2. Normalized features using `StandardScaler`
3. Trained KNN with different `k` values
4. Plotted **Accuracy vs K** (`knn_accuracy_vs_k.png`)
5. Selected the best `k` and evaluated with **Confusion Matrix**
6. Visualized **Decision Boundaries** using first 2 features (`knn_decision_boundary.png`)

## Results
- Accuracy depends on the choice of `k`
- Best `k` is selected based on test accuracy
- Confusion matrix shows classification performance
- Decision boundary visualization shows separation of Iris species

## Files
- `main.py` → Main code
- `Iris.csv` → Dataset
- `knn_accuracy_vs_k.png` → Accuracy vs K plot
- `confusion_matrix_knn.png` → Confusion matrix
- `knn_decision_boundary.png` → Decision boundaries
- `requirements.txt` → Dependencies

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM with Linear Kernel
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)

# SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)

# Hyperparameter tuning using GridSearchCV for RBF kernel
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_svm_rbf = grid_search.best_estimator_

# Predictions
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = best_svm_rbf.predict(X_test)

# Evaluate function
def evaluate_model(y_test, y_pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return precision, recall, f1, accuracy

# Evaluation for both models
precision_linear, recall_linear, f1_linear, accuracy_linear = evaluate_model(y_test, y_pred_linear)
precision_rbf, recall_rbf, f1_rbf, accuracy_rbf = evaluate_model(y_test, y_pred_rbf)

# Print results
print("Linear Kernel SVM:")
print(f"Precision: {precision_linear:.2f}, Recall: {recall_linear:.2f}, F1-Score: {f1_linear:.2f}, Accuracy: {accuracy_linear:.2f}")
print("\nRBF Kernel SVM (with Hyperparameter Tuning):")
print(f"Precision: {precision_rbf:.2f}, Recall: {recall_rbf:.2f}, F1-Score: {f1_rbf:.2f}, Accuracy: {accuracy_rbf:.2f}")

# Confusion Matrix
cm_linear = confusion_matrix(y_test, y_pred_linear)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

# ROC Curve
fpr_linear, tpr_linear, _ = roc_curve(y_test, svm_linear.decision_function(X_test))
fpr_rbf, tpr_rbf, _ = roc_curve(y_test, best_svm_rbf.decision_function(X_test))
roc_auc_linear = auc(fpr_linear, tpr_linear)
roc_auc_rbf = auc(fpr_rbf, tpr_rbf)

# Visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

plt.figure(figsize=(12, 12))

# SVM Linear Kernel Plot
plt.subplot(2, 2, 1)
plt.title("SVM with Linear Kernel")
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred_linear, cmap='coolwarm', marker='o', edgecolor='k', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# SVM RBF Kernel Plot
plt.subplot(2, 2, 2)
plt.title("SVM with RBF Kernel (Tuned)")
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred_rbf, cmap='coolwarm', marker='o', edgecolor='k', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Confusion Matrix Heatmap
plt.subplot(2, 2, 3)
plt.title("Confusion Matrix - Linear Kernel")
plt.imshow(cm_linear, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Linear Kernel SVM Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.yticks([0, 1], ['Malignant', 'Benign'])

# Confusion Matrix Heatmap for RBF
plt.subplot(2, 2, 4)
plt.title("Confusion Matrix - RBF Kernel")
plt.imshow(cm_rbf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('RBF Kernel SVM Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.yticks([0, 1], ['Malignant', 'Benign'])

plt.tight_layout()
plt.show()

# ROC Curve Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_linear, tpr_linear, color='blue', lw=2, label=f'Linear Kernel (AUC = {roc_auc_linear:.2f})')
plt.plot(fpr_rbf, tpr_rbf, color='red', lw=2, label=f'RBF Kernel (AUC = {roc_auc_rbf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

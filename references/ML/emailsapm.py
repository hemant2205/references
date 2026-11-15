#email sapm
# 1. Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, precision_score,
    recall_score, f1_score
)



# 2. Load the dataset

df = pd.read_csv("emails.csv")

print("\n========== DATASET INFORMATION ==========")
print("First five records:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nTotal missing values:")
print(df.isnull().sum().sum())


# 3. Preprocess the dataset


# Remove email ID column
df = df.drop("Email No.", axis=1)

# Remove duplicate and missing records
df = df.drop_duplicates()
df = df.dropna()

print("Dataset after preprocessing:")
print(df.shape)

# Separate features and target
X = df.drop("Prediction", axis=1)
y = df["Prediction"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# 4. Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# 5. Feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 6. KNN Model


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("KNN model trained successfully!")
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))


# 7. SVM Model

print("\n========== SVM CLASSIFICATION ==========")

svm = SVC()

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("SVM model trained successfully!")
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


# 8. Classification reports

print("\n========== KNN CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred_knn))

print("\n========== SVM CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred_svm))


# 9. Confusion matrices

cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)

print("\nKNN Confusion Matrix:")
print(cm_knn)

print("\nSVM Confusion Matrix:")
print(cm_svm)


# 10. Visualize confusion matrices

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.subplot(1, 2, 2)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.tight_layout()
plt.show()

print("\nConfusion matrix graphs show correct and incorrect predictions.")


# 11. Compare model performance

results = pd.DataFrame({
    "Model": ["KNN", "SVM"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_svm)
    ],
    "Precision": [
        precision_score(y_test, y_pred_knn, zero_division=0),
        precision_score(y_test, y_pred_svm, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred_knn, zero_division=0),
        recall_score(y_test, y_pred_svm, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_knn, zero_division=0),
        f1_score(y_test, y_pred_svm, zero_division=0)
    ]
})

print("\n========== MODEL PERFORMANCE COMPARISON ==========")
print(results.to_string(index=False))


# 12. Performance comparison graph

results.set_index("Model").plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("KNN vs SVM Performance Comparison")
plt.xlabel("Classification Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print("\nThe bar graph compares Accuracy, Precision, Recall and F1 Score.")


# 13. Predict an email from the test dataset

print("\n========== EMAIL PREDICTION ==========")

email = X_test[[0]]

prediction = svm.predict(email)

if prediction[0] == 1:
    print("Predicted Result: Email is SPAM")
else:
    print("Predicted Result: Email is NOT SPAM")

print("Actual Email Category:", y_test.iloc[0])
print("Predicted Email Category:", prediction[0])

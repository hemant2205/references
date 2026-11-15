#diabetics
# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

print("All libraries imported successfully!")


# 2. Load and Explore Dataset

df = pd.read_csv("diabetes.csv")

print("\n========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nStatistical description:")
print(df.describe())


# 3. Data Cleaning

print("\n========== DATA CLEANING ==========")

# Remove duplicate records
df = df.drop_duplicates()

# Replace invalid zero values with missing values
cols = ["Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, np.nan)

print("Invalid zero values replaced with missing values.")

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fill missing values using training data medians
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print("Missing values handled successfully!")


# 4. Display Features and Target

print("\n========== FEATURES AND TARGET ==========")
print("Features shape:", X_train.shape)
print("Target shape:", y_train.shape)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# 5. Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature scaling completed successfully!")


# 6. Train KNN Classifier

print("\n========== KNN MODEL TRAINING ==========")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

print("KNN model trained successfully!")
print("Number of neighbours (K): 5")


# 7. Predict Test Data

y_pred = knn.predict(X_test)

print("\nPredictions completed successfully!")


# 8. Calculate Model Performance

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Error Rate:", round(error_rate * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))


# 9. Display Confusion Matrix

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\nRows represent actual values.")
print("Columns represent predicted values.")
print("0 = No Diabetes, 1 = Diabetes.")


# 10. Visualize Confusion Matrix

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.title("Diabetes Prediction - KNN Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()
plt.show()

print("\nConfusion matrix graph displayed successfully!")


# 11. Sample Prediction

print("\n========== SAMPLE PREDICTION ==========")

sample = X_test[0].reshape(1, -1)

prediction = knn.predict(sample)

if prediction[0] == 1:
    print("Prediction: Diabetes indicated")
else:
    print("Prediction: No diabetes indicated")

print("Actual Outcome:", y_test.iloc[0])
print("Predicted Outcome:", prediction[0])


# 12. Display First 10 Predictions

print("\n========== FIRST 10 PREDICTIONS ==========")

for i in range(min(10, len(X_test))):
    prediction = knn.predict(X_test[i].reshape(1, -1))

    print(
        "Person", i + 1,
        "| Actual:", y_test.iloc[i],
        "| Predicted:", prediction[0]
    )

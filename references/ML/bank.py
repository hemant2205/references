#bank customer
%pip install tensorflow
# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

print("All libraries imported successfully!")


# 2. Load Dataset

df = pd.read_csv("Churn_Modelling.csv")

print("\n========== DATASET INFORMATION ==========")
print("Dataset shape:", df.shape)

print("\nFirst five customer records:")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())


# 3. Data Preprocessing

print("\n========== DATA PREPROCESSING ==========")

# Remove duplicate records
df = df.drop_duplicates()

# Remove unnecessary columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    dtype=int
)

print("Dataset after preprocessing:")
print(df.head())

print("\nPreprocessed dataset shape:", df.shape)


# 4. Separate Features and Target

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\n========== FEATURES AND TARGET ==========")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nCustomer churn distribution:")
print(y.value_counts())


# 5. Split Dataset into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# 6. Normalize the Data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature normalization completed successfully!")


# 7. Build ANN Model

print("\n========== BUILDING ANN MODEL ==========")

model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

print("Neural network model created successfully!")


# 8. Compile Model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()


# 9. Early Stopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nEarly stopping configured successfully!")


# 10. Train Model

print("\n========== MODEL TRAINING ==========")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\nModel training completed!")


# 11. Predict on Test Data

print("\n========== CUSTOMER CHURN PREDICTION ==========")

y_pred = model.predict(X_test)

# Convert probabilities into 0 or 1
y_pred = (y_pred >= 0.5).astype(int).ravel()


# 12. Calculate Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")


# 13. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\nRows represent actual values.")
print("Columns represent predicted values.")
print("0 = Customer Stayed, 1 = Customer Exited.")


# 14. Visualize Confusion Matrix

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Exited"],
    yticklabels=["Stayed", "Exited"]
)

plt.title("Bank Customer Churn Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()
plt.show()

print("\nConfusion matrix graph displayed successfully!")


# 15. Classification Report

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Churn", "Churn"],
    zero_division=0
))

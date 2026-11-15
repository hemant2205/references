#uber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Dataset

df = pd.read_csv("uber.csv")

print("\nFirst 5 rows of the Uber dataset:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

# 2. Data Preprocessing

df = df.drop_duplicates()
df = df.dropna()
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")

df = df[df["fare_amount"] > 0]
df = df[df["passenger_count"] > 0]


# 3. Calculate Distance using Haversine Formula

lat1 = np.radians(df["pickup_latitude"])
lon1 = np.radians(df["pickup_longitude"])
lat2 = np.radians(df["dropoff_latitude"])
lon2 = np.radians(df["dropoff_longitude"])

a = (np.sin((lat2-lat1)/2)**2 +
     np.cos(lat1)*np.cos(lat2)*
     np.sin((lon2-lon1)/2)**2)

df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

df = df[df["distance_km"] > 0]

print("\nDataset with calculated distance:")
print(df[["pickup_latitude", "pickup_longitude",
          "dropoff_latitude", "dropoff_longitude",
          "distance_km"]].head())

# 4. Outlier Detection and Removal
print("\n--- OUTLIER DETECTION ---")
print("Boxplot showing outliers in fare amount:")

sns.boxplot(x=df["fare_amount"])
plt.title("Outliers in Uber Fare Amount")
plt.show()

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

df = df[df["fare_amount"].between(
    Q1 - 1.5 * IQR,
    Q3 + 1.5 * IQR
)]


# 5. Correlation Analysis

corr = df[["distance_km", "fare_amount"]].corr()

print("Correlation between distance and fare amount:")
print(corr)

print("\nHeatmap showing correlation between variables:")
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 6. Select Features and Target

X = df[["distance_km", "passenger_count"]]
y = df["fare_amount"]


# 7. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8. Linear Regression Model

lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)


# 9. Random Forest Regression Model

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("Random Forest model trained successfully.")

# 10. Model Evaluation and Comparison
print("\n--- MODEL COMPARISON TABLE ---")

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [
        r2_score(y_test, pred_lr),
        r2_score(y_test, pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, pred_lr)),
        np.sqrt(mean_squared_error(y_test, pred_rf))
    ]
})

print("This table compares the performance of both models.")
print("Higher R2 and lower RMSE indicate better performance.")
print(results)

# 11. Actual vs Predicted Graph
print("\n--- ACTUAL VS PREDICTED FARE ---")

plt.scatter(y_test, pred_lr)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

plt.scatter(y_test, pred_rf)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Random Forest: Actual vs Predicted")
plt.show()

# 12. Predict Fare for New Ride
print("\n--- NEW FARE PREDICTION ---")

distance = float(input("Enter distance in km: "))
passengers = int(input("Enter passenger count: "))

new_ride = pd.DataFrame({
    "distance_km": [distance],
    "passenger_count": [passengers]
})

print("\nPredicted fare using Linear Regression:")
print(lr.predict(new_ride)[0])

print("\nPredicted fare using Random Forest:")
print(rf.predict(new_ride)[0])

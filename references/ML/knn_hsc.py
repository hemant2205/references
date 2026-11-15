
# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

print("Libraries imported successfully!")


# 2. Load and Explore Dataset

df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

print("\n========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


# 3. Data Cleaning and Preprocessing

print("\n========== DATA PREPROCESSING ==========")

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing values in selected columns
df = df.dropna(
    subset=["QUANTITYORDERED", "PRICEEACH", "SALES"]
)

print("Dataset shape after cleaning:", df.shape)

# Select numerical features
X = df[["QUANTITYORDERED", "PRICEEACH", "SALES"]]

print("\nSelected features:")
print(X.head())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed successfully!")


# 4. K-Means: Elbow Method

print("\n========== ELBOW METHOD ==========")

wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

print("WCSS values for K = 1 to 10:")
print(wcss)

# Plot Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

print("\nElbow graph displayed.")
print("Choose K at the point where WCSS starts decreasing slowly.")


# 5. Apply K-Means Clustering

print("\n========== K-MEANS CLUSTERING ==========")

# Select K based on the elbow graph
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means clustering completed successfully!")

print("\nFirst five customer records with K-Means clusters:")
print(df[
    ["QUANTITYORDERED", "PRICEEACH", "SALES", "KMeans_Cluster"]
].head())


# 6. Visualize K-Means Clusters

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=df["QUANTITYORDERED"],
    y=df["SALES"],
    hue=df["KMeans_Cluster"],
    palette="viridis"
)

plt.title("K-Means Customer Segmentation")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.legend(title="Cluster")
plt.show()

print("\nK-Means cluster graph displayed!")


# 7. Hierarchical Clustering: Dendrogram

print("\n========== HIERARCHICAL CLUSTERING ==========")

# Use first 200 records for readable dendrogram
linked = linkage(X_scaled[:200], method="ward")

plt.figure(figsize=(10, 6))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

print("\nDendrogram displayed successfully!")


# 8. Apply Agglomerative Clustering

hc = AgglomerativeClustering(n_clusters=3)

df["Hierarchical_Cluster"] = hc.fit_predict(X_scaled)

print("Hierarchical clustering completed successfully!")

print("\nFirst five records with Hierarchical clusters:")
print(df[
    ["QUANTITYORDERED", "PRICEEACH", "SALES",
     "Hierarchical_Cluster"]
].head())


# 9. Visualize Hierarchical Clusters

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=df["QUANTITYORDERED"],
    y=df["SALES"],
    hue=df["Hierarchical_Cluster"],
    palette="deep"
)

plt.title("Hierarchical Customer Segmentation")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.legend(title="Cluster")
plt.show()

print("\nHierarchical clustering graph displayed!")


# 10. Compare Cluster Distributions

print("\n========== CLUSTER COMPARISON ==========")

print("\nK-Means Cluster Counts:")
print(df["KMeans_Cluster"].value_counts().sort_index())

print("\nHierarchical Cluster Counts:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())

print("\nCustomer segmentation completed successfully!")

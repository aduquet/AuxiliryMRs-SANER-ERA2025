import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Step 1: Load the MNIST data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X = X.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

# # Step 2: Standardize the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Step 3: Apply K-Means clustering
# kmeans = KMeans(n_clusters=10, random_state=42)
# kmeans.fit(X_scaled)
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_

# # Step 4: Use t-SNE to reduce the dimensionality to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X_scaled)

# # Step 5: Plot the clusters with t-SNE and show cluster centroids
# plt.figure(figsize=(10, 7))

# # Plot the t-SNE reduced data points with their cluster labels
# scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=2)

# # Add a colorbar
# plt.colorbar(scatter, label="Cluster")

# # Plot the cluster centroids
# centroids_2d = tsne.fit_transform(centroids)
# plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroids')

# plt.title("K-Means Clusters of MNIST Data (Visualized with t-SNE)")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend()
# plt.show()
# Take a subset of the data (e.g., 5000 samples)
subset_size = 100
X_subset = X_scaled[:subset_size]
y_subset = y[:subset_size]

# Apply K-Means on the subset
kmeans_subset = KMeans(n_clusters=10, random_state=42)
kmeans_subset.fit(X_subset)
labels_subset = kmeans_subset.labels_
centroids_subset = kmeans_subset.cluster_centers_

# Use t-SNE on the smaller subset
X_2d_subset = tsne.fit_transform(X_subset)

# Plotting
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_2d_subset[:, 0], X_2d_subset[:, 1], c=labels_subset, cmap='tab10', s=2)
plt.colorbar(scatter, label="Cluster")
plt.title("K-Means Clusters of MNIST Data (Visualized with t-SNE, Subset)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
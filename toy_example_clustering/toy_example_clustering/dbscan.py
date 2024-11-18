import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Create synthetic data
n_samples = 1500
random_state = 42

# Create an elongated cluster
elongated_data, _ = make_blobs(n_samples=n_samples//2, centers=[[2, 2]], cluster_std=[0.4], random_state=random_state)
elongated_data[:, 1] *= 5  # Stretch the cluster in the y direction

# Create a compact circular cluster
compact_data, _ = make_blobs(n_samples=n_samples//2, centers=[[8, 8]], cluster_std=0.4, random_state=random_state)

# Combine the two clusters
X = np.vstack((elongated_data, compact_data))

# Add random noise
noise = np.random.uniform(low=[-2, -2], high=[10, 12], size=(int(0.1 * n_samples), 2))
X = np.vstack((X, noise))

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.title('DBSCAN Clustering Results')
plt.show()

# Calculate centroids for each cluster (ignoring noise, label -1)
unique_labels = set(labels)
centroids = []

for label in unique_labels:
    if label != -1:  # Ignore noise
        points_in_cluster = X[labels == label]
        centroid = points_in_cluster.mean(axis=0)
        centroids.append(centroid)

centroids = np.array(centroids)

# Plot clusters with centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('DBSCAN Clusters with Centroids')
plt.legend()
plt.show()

from scipy.spatial.distance import cdist

medoids = []

for label in unique_labels:
    if label != -1:  # Ignore noise
        points_in_cluster = X[labels == label]
        # Calculate the medoid: the point with the smallest sum of distances to other points
        distances = cdist(points_in_cluster, points_in_cluster)
        medoid_index = np.argmin(distances.sum(axis=1))
        medoids.append(points_in_cluster[medoid_index])

medoids = np.array(medoids)

# Plot clusters with medoids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.scatter(medoids[:, 0], medoids[:, 1], c='blue', marker='o', s=100, label='Medoids')
plt.title('DBSCAN Clusters with Centroids and Medoids')
plt.legend()
plt.show()


# Visualize the dataset
# plt.scatter(X[:, 0], X[:, 1], s=10, color='gray')
# plt.title('Synthetic Data with Elongated and Compact Clusters')
# plt.show()

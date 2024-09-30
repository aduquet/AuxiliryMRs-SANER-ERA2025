import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.datasets import mnist
import random

# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatten the 28x28 images into vectors of size 784
train_X = train_X.reshape(-1, 784)

# Limit the data size for faster computation (optional)
train_X = train_X[:5000]  # Use 5,000 samples for training

# Train K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42).fit(train_X)
train_cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Calculate intra-cluster distance per cluster
def calculate_intra_cluster_distances_per_cluster(X, labels, centers):
    intra_distances = {}
    for i, center in enumerate(centers):
        points_in_cluster = X[labels == i]
        distances = np.linalg.norm(points_in_cluster - center, axis=1)
        intra_distances[i] = np.mean(distances) if len(points_in_cluster) > 0 else 0
    return intra_distances

# Step 1: Get initial intra-cluster distances for each cluster
initial_intra_distances = calculate_intra_cluster_distances_per_cluster(train_X, train_cluster_labels, cluster_centers)
print("Initial Intra-cluster Distances:", initial_intra_distances)

# Step 2: Swap a point between two clusters
def swap_points_between_clusters(X, labels, cluster_1, cluster_2):
    # Find indices of points in cluster_1 and cluster_2
    indices_cluster_1 = np.where(labels == cluster_1)[0]
    indices_cluster_2 = np.where(labels == cluster_2)[0]
    
    # Randomly select one point from each cluster
    if len(indices_cluster_1) == 0 or len(indices_cluster_2) == 0:
        return X, labels  # Skip if either cluster is empty

    point_idx_1 = random.choice(indices_cluster_1)
    point_idx_2 = random.choice(indices_cluster_2)

    # Swap the labels of the selected points
    labels[point_idx_1], labels[point_idx_2] = labels[point_idx_2], labels[point_idx_1]

    return X, labels

# Choose two random clusters to swap points
cluster_1 = random.choice(range(10))
cluster_2 = random.choice(range(10))
print(f"Swapping points between Cluster {cluster_1} and Cluster {cluster_2}")

# Perform the swap
train_X_swapped, train_cluster_labels_swapped = swap_points_between_clusters(train_X, train_cluster_labels, cluster_1, cluster_2)

# Step 3: Recalculate intra-cluster distances after the swap
post_swap_intra_distances = calculate_intra_cluster_distances_per_cluster(train_X_swapped, train_cluster_labels_swapped, cluster_centers)
print("Post-swap Intra-cluster Distances:", post_swap_intra_distances)

# Step 4: Check if the intra-cluster distances increased after the swap
for cluster in [cluster_1, cluster_2]:
    if post_swap_intra_distances[cluster] > initial_intra_distances[cluster]:
        print(f"Cluster {cluster}: Intra-cluster distance increased after the swap.")
    else:
        print(f"Cluster {cluster}: Intra-cluster distance did not increase after the swap.")

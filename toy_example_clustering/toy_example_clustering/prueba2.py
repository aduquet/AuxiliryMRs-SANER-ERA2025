import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tensorflow.keras.datasets import mnist

# Load MNIST data
(train_X, _), (test_X, _) = mnist.load_data()

# Flatten the 28x28 images into vectors of size 784
train_X = train_X.reshape(-1, 784)

# Limit the data size for faster computation
train_X = train_X[:5000]  # Use 5000 samples for training

# Train K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42).fit(train_X)
train_cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Function to calculate intra-cluster distances
def calculate_intra_cluster_distances_per_cluster(X, labels, centers):
    intra_distances = {}
    for i, center in enumerate(centers):
        points_in_cluster = X[labels == i]
        distances = np.linalg.norm(points_in_cluster - center, axis=1)
        intra_distances[i] = np.mean(distances) if len(points_in_cluster) > 0 else 0
    return intra_distances

# Initial intra-cluster distances for reference
initial_intra_distances = calculate_intra_cluster_distances_per_cluster(train_X, train_cluster_labels, cluster_centers)
print("Initial Intra-cluster Distances:", initial_intra_distances)

# Function to swap cluster centers
def swap_cluster_centers(centers, cluster_1, cluster_2):
    new_centers = centers.copy()
    new_centers[cluster_1], new_centers[cluster_2] = new_centers[cluster_2], new_centers[cluster_1]
    return new_centers

# Function to reassign points to the new centers
def reassign_clusters(X, new_centers):
    new_labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - new_centers, axis=2), axis=1)
    return new_labels

# List to store all results for the table
swap_results = []

# Loop through each cluster and swap it with every other cluster
for cluster in range(10):
    for swap_with in range(10):
        if cluster != swap_with:
            # Swap the centers of 'cluster' and 'swap_with'
            swapped_centers = swap_cluster_centers(cluster_centers, cluster, swap_with)
            
            # Reassign points based on swapped centers
            new_cluster_labels = reassign_clusters(train_X, swapped_centers)
            
            # Recalculate intra-cluster distances after the swap
            post_swap_intra_distances = calculate_intra_cluster_distances_per_cluster(train_X, new_cluster_labels, swapped_centers)
            
            # Store the pre-swap and post-swap distances for the clusters involved
            swap_results.append({
                'Cluster': cluster,
                'Swapped With': swap_with,
                'Initial Distance (Cluster {})'.format(cluster): initial_intra_distances[cluster],
                'Post-swap Distance (Cluster {})'.format(cluster): post_swap_intra_distances[cluster],
                'Initial Distance (Cluster {})'.format(swap_with): initial_intra_distances[swap_with],
                'Post-swap Distance (Cluster {})'.format(swap_with): post_swap_intra_distances[swap_with]
            })

# Convert the results to a Pandas DataFrame to display as a table
swap_results_df = pd.DataFrame(swap_results)

# Display the table
print(swap_results_df)

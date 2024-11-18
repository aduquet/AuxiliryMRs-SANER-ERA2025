# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans

# # Generate synthetic data with 3 clusters and some overlap
# X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)

# # Fit K-Means clustering (with bad initialization)
# kmeans = KMeans(n_clusters=3, init='random', n_init=1, random_state=42)
# kmeans.fit(X)

# # Get the centroids and labels
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# # Plot the data points and centroids
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')  # Plot centroids
# plt.title("Suboptimal Centroid Placement Example")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic data with 3 clusters and some overlap
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)

# Step 2: Perform K-Means clustering
kmeans = KMeans(n_clusters=4, init='random',random_state=42)
kmeans.fit(X)
original_centroids = kmeans.cluster_centers_
original_labels = kmeans.labels_

# Step 3: Calculate initial intra-cluster distances
def intra_cluster_distance(X, labels, centroids):
    distances = {}
    for i, center in enumerate(centroids):
        points_in_cluster = X[labels == i]
        if len(points_in_cluster) > 0:
            distances[i] = np.mean(np.linalg.norm(points_in_cluster - center, axis=1))
        else:
            distances[i] = None  # Handle case where no points are in the cluster
    return distances

initial_intra_distances = intra_cluster_distance(X, original_labels, original_centroids)
print(f"Initial Intra-cluster Distances: {initial_intra_distances}")

# Step 4: Perturb the centroids (small random shift)
def perturb_centroids(centroids, noise_level=0.2):
    perturbed_centroids = centroids + np.random.normal(0, noise_level, centroids.shape)
    return perturbed_centroids

perturbed_centroids = perturb_centroids(original_centroids, noise_level=0.5)

# Step 5: Reassign points based on the perturbed centroids
def reassign_points(X, new_centroids):
    new_labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - new_centroids, axis=2), axis=1)
    return new_labels

perturbed_labels = reassign_points(X, perturbed_centroids)

# Step 6: Calculate new intra-cluster distances after perturbation
perturbed_intra_distances = intra_cluster_distance(X, perturbed_labels, perturbed_centroids)
print(f"Post-perturbation Intra-cluster Distances: {perturbed_intra_distances}")

# Step 7: Compare the original and perturbed results (AMR validation)
for cluster_id in initial_intra_distances.keys():
    initial_distance = initial_intra_distances[cluster_id]
    perturbed_distance = perturbed_intra_distances[cluster_id]
    
    if initial_distance is not None and perturbed_distance is not None:
        print(f"Cluster {cluster_id}: Initial Distance = {initial_distance:.4f}, Post-perturbation Distance = {perturbed_distance:.4f}")
        if perturbed_distance > initial_distance:
            print(f"Cluster {cluster_id} validation passed: Intra-cluster distance increased (possible suboptimal centroid).")
        else:
            print(f"Cluster {cluster_id} validation failed: Intra-cluster distance decreased (centroid was optimal).")
    else:
        print(f"Cluster {cluster_id}: No points assigned to this cluster.")

# import numpy as np
# from sklearn.cluster import KMeans
# from tensorflow.keras.datasets import mnist
# from scipy.stats import mode
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# # Load MNIST data
# (train_X, train_y), (test_X, test_y) = mnist.load_data()

# # Flatten the 28x28 images into vectors of size 784
# train_X = train_X.reshape(-1, 784)
# test_X = test_X.reshape(-1, 784)

# # Normalize the data
# train_X = train_X / 255.0
# test_X = test_X / 255.0

# # Train K-Means clustering on clean data
# kmeans = KMeans(n_clusters=10, init='random', random_state=42).fit(train_X)
# train_cluster_labels = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_

# # Function to calculate intra-cluster distances
# def intra_cluster_distance(X, labels, cluster_center, cluster_id):
#     points_in_cluster = X[labels == cluster_id]
#     if len(points_in_cluster) == 0:
#         return None
#     distances = np.linalg.norm(points_in_cluster - cluster_center, axis=1)
#     return np.mean(distances)

# # Get initial intra-cluster distances
# initial_intra_distances = {i: intra_cluster_distance(train_X, train_cluster_labels, cluster_centers[i], i) for i in range(10)}

# print("Initial Intra-cluster Distances:", initial_intra_distances)

# # Perturb cluster centers (simulate output transformation)
# def perturb_cluster_centers(centers, noise_level=0.5):
#     return centers + np.random.normal(0, noise_level, centers.shape)

# # Step 2: Perturb the cluster centers
# perturbed_centers = perturb_cluster_centers(cluster_centers, noise_level=1.0)

# # Step 3: Reassign points to new cluster centers after perturbation
# def reassign_clusters(X, new_centers):
#     # Assign points to the closest perturbed center
#     return np.argmin(np.linalg.norm(X[:, np.newaxis] - new_centers, axis=2), axis=1)

# # Get new cluster assignments based on perturbed centers
# new_cluster_labels = reassign_clusters(train_X, perturbed_centers)

# # Step 4: Calculate intra-cluster distances after perturbation
# post_perturbation_intra_distances = {i: intra_cluster_distance(train_X, new_cluster_labels, perturbed_centers[i], i) for i in range(10)}

# print("Post-perturbation Intra-cluster Distances:", post_perturbation_intra_distances)

# # Step 5: Validate by comparing initial and post-perturbation intra-cluster distances
# for cluster in initial_intra_distances.keys():
#     initial_distance = initial_intra_distances[cluster]
#     perturbed_distance = post_perturbation_intra_distances[cluster]
    
#     # Check if the perturbed distance is None (no points in the cluster after reassignment)
#     if perturbed_distance is None:
#         print(f"Cluster {cluster}: No points in the cluster after perturbation. Skipping validation.")
#         continue
    
#     # Compare the distances only if both are valid
#     if perturbed_distance > initial_distance:
#         status = 'validation passed'
#     else:
#         status = 'validation failed: distance did not increase'
    
#     print(f"Cluster {cluster}: Initial Distance = {initial_distance}, Post-perturbation Distance = {perturbed_distance}, Status: {status}")

import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.datasets import mnist
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatten the 28x28 images into vectors of size 784
train_X = train_X.reshape(-1, 784)
test_X = test_X.reshape(-1, 784)

# Normalize the data
train_X = train_X / 255.0
test_X = test_X / 255.0

# Train K-Means clustering on clean data
kmeans = KMeans(n_clusters=10, init='random', random_state=42).fit(train_X)
train_cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Function to calculate intra-cluster distances
def intra_cluster_distance(X, labels, cluster_center, cluster_id):
    points_in_cluster = X[labels == cluster_id]
    if len(points_in_cluster) == 0:
        return None
    distances = np.linalg.norm(points_in_cluster - cluster_center, axis=1)
    return np.mean(distances)

# Get initial intra-cluster distances
initial_intra_distances = {i: intra_cluster_distance(train_X, train_cluster_labels, cluster_centers[i], i) for i in range(10)}

print("Initial Intra-cluster Distances:", initial_intra_distances)

# Perturb cluster centers by moving them toward the mean of the cluster points
def perturb_cluster_centers_to_mean(X, labels, centers):
    perturbed_centers = []
    for i, center in enumerate(centers):
        points_in_cluster = X[labels == i]
        if len(points_in_cluster) > 0:
            cluster_mean = np.mean(points_in_cluster, axis=0)
            # Move center closer to the mean of points in the cluster
            new_center = (center + cluster_mean) / 2
            perturbed_centers.append(new_center)
        else:
            perturbed_centers.append(center)  # No points in the cluster, no perturbation
    return np.array(perturbed_centers)

# Step 2: Perturb the cluster centers by moving them toward the mean of the points in each cluster
perturbed_centers = perturb_cluster_centers_to_mean(train_X, train_cluster_labels, cluster_centers)

# Step 3: Reassign points to new cluster centers after perturbation
def reassign_clusters(X, new_centers):
    # Assign points to the closest perturbed center
    return np.argmin(np.linalg.norm(X[:, np.newaxis] - new_centers, axis=2), axis=1)

# Get new cluster assignments based on perturbed centers
new_cluster_labels = reassign_clusters(train_X, perturbed_centers)

# Step 4: Calculate intra-cluster distances after perturbation
post_perturbation_intra_distances = {i: intra_cluster_distance(train_X, new_cluster_labels, perturbed_centers[i], i) for i in range(10)}

print("Post-perturbation Intra-cluster Distances:", post_perturbation_intra_distances)

# Step 5: Validate by comparing initial and post-perturbation intra-cluster distances
for cluster in initial_intra_distances.keys():
    initial_distance = initial_intra_distances[cluster]
    perturbed_distance = post_perturbation_intra_distances[cluster]
    
    # Check if the perturbed distance is None (no points in the cluster after reassignment)
    if perturbed_distance is None:
        print(f"Cluster {cluster}: No points in the cluster after perturbation. Skipping validation.")
        continue
    
    # Compare the distances only if both are valid
    if perturbed_distance > initial_distance:
        status = 'validation passed'
    else:
        status = 'validation failed: distance did not increase'
    
    print(f"Cluster {cluster}: Initial Distance = {initial_distance}, Post-perturbation Distance = {perturbed_distance}, Status: {status}")

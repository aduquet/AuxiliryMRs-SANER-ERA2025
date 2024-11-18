import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import make_blobs
from scipy.stats import mode

# Step 1: Generate a synthetic dataset with overlapping clusters
X, y_true = make_blobs(n_samples=1000, centers=3, cluster_std=2.5, random_state=42)  # Increased cluster_std to create overlap

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
plt.title("Dataset with Overlapping Clusters")
plt.show()

# Step 2: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, init='random', random_state=42).fit(X)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Step 3: Evaluate Performance (Mapping clusters to labels)

def map_clusters_to_labels(cluster_labels, true_labels):
    label_mapping = {}
    for cluster in np.unique(cluster_labels):
        # Get the true labels for points in this cluster
        cluster_points = true_labels[cluster_labels == cluster]

        # Check if the cluster has points, otherwise skip
        if len(cluster_points) == 0:
            print(f"Warning: Cluster {cluster} is empty. Skipping.")
            continue
        
        # Calculate the mode (most frequent label) of the points in the cluster
        mode_result = mode(cluster_points)

        # Check if the mode result is valid
        if isinstance(mode_result.mode, np.ndarray) and len(mode_result.mode) > 0:
            true_label_for_cluster = mode_result.mode[0]  # Safely access the first element
        else:
            true_label_for_cluster = mode_result.mode  # Directly access the mode if it's a scalar

        label_mapping[cluster] = true_label_for_cluster

    return label_mapping

# Map the clusters to actual labels
label_mapping = map_clusters_to_labels(cluster_labels, y_true)
mapped_test_labels = np.array([label_mapping[cluster] for cluster in cluster_labels])

# Step 4: Calculate performance metrics
conf_matrix = confusion_matrix(y_true, mapped_test_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Print Classification Report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_true, mapped_test_labels))

# Overall Accuracy
accuracy = accuracy_score(y_true, mapped_test_labels)
print(f"Accuracy: {accuracy:.4f}")

# Step 5: Apply Perturbation (Shift the cluster centers)
def perturb_cluster_centers(centers, noise_level=1.0):
    return centers + np.random.normal(0, noise_level, centers.shape)

perturbed_centers = perturb_cluster_centers(cluster_centers, noise_level=1.0)

# Step 6: Reassign points to new cluster centers after perturbation
def reassign_clusters(X, new_centers):
    return np.argmin(np.linalg.norm(X[:, np.newaxis] - new_centers, axis=2), axis=1)

new_cluster_labels = reassign_clusters(X, perturbed_centers)

# Step 7: Calculate Intra-cluster Distances Before and After Perturbation
def intra_cluster_distance(X, labels, cluster_center, cluster_id):
    points_in_cluster = X[labels == cluster_id]
    if len(points_in_cluster) == 0:
        return None
    distances = np.linalg.norm(points_in_cluster - cluster_center, axis=1)
    return np.mean(distances)

# Calculate intra-cluster distances before perturbation
initial_intra_distances = {i: intra_cluster_distance(X, cluster_labels, cluster_centers[i], i) for i in range(3)}
# Calculate intra-cluster distances after perturbation
post_perturbation_intra_distances = {i: intra_cluster_distance(X, new_cluster_labels, perturbed_centers[i], i) for i in range(3)}

# Step 8: Validate using AMRs (compare initial and post-perturbation intra-cluster distances)
print("\nAuxiliary Metamorphic Validation:")
for cluster in initial_intra_distances.keys():
    initial_distance = initial_intra_distances[cluster]
    perturbed_distance = post_perturbation_intra_distances[cluster]
    
    if perturbed_distance is None:
        print(f"Cluster {cluster}: No points after perturbation. Skipping validation.")
        continue
    
    if perturbed_distance > initial_distance:
        status = 'validation passed'
    else:
        status = 'validation failed: intra-cluster distance did not increase'
    
    print(f"Cluster {cluster}: Initial Distance = {initial_distance:.4f}, Post-perturbation Distance = {perturbed_distance:.4f}, Status: {status}")

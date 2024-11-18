import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import make_blobs
from scipy.stats import mode

# Step 1: Generate a synthetic dataset with overlapping clusters
X, y_true = make_blobs(n_samples=1000, centers=5, cluster_std=2.5, random_state=42)  # Increased cluster_std to create overlap

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
plt.title("Dataset with Overlapping Clusters")
plt.show()

# Step 2: Train a Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(X)
gmm_labels = gmm.predict(X)
gmm_means = gmm.means_
gmm_covariances = gmm.covariances_

# Step 3: Map GMM clusters to true labels
def map_clusters_to_labels(cluster_labels, true_labels):
    label_mapping = {}
    for cluster in np.unique(cluster_labels):
        cluster_points = true_labels[cluster_labels == cluster]
        if len(cluster_points) == 0:
            continue
        mode_result = mode(cluster_points)
        if isinstance(mode_result.mode, np.ndarray) and len(mode_result.mode) > 0:
            true_label_for_cluster = mode_result.mode[0]
        else:
            true_label_for_cluster = mode_result.mode
        label_mapping[cluster] = true_label_for_cluster
    return label_mapping

label_mapping = map_clusters_to_labels(gmm_labels, y_true)
mapped_labels = np.array([label_mapping[cluster] for cluster in gmm_labels])

# Step 4: Evaluate traditional metrics (Precision, Recall, F1-score)
conf_matrix = confusion_matrix(y_true, mapped_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_true, mapped_labels))

# Step 5: Apply perturbation to GMM (Perturb the means and covariances)
def perturb_gmm_means(means, noise_level=1.0):
    return means + np.random.normal(0, noise_level, means.shape)

def perturb_gmm_covariances(covariances, factor=1.5):
    return covariances * factor

perturbed_means = perturb_gmm_means(gmm_means, noise_level=2.0)  # Larger noise for more perturbation
perturbed_covariances = perturb_gmm_covariances(gmm_covariances, factor=1.5)

# Step 6: Create a new GMM with perturbed means and covariances
perturbed_gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
perturbed_gmm.means_ = perturbed_means
perturbed_gmm.covariances_ = perturbed_covariances
perturbed_gmm.weights_ = gmm.weights_
perturbed_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(perturbed_covariances))

# Predict labels with the perturbed GMM
perturbed_labels = perturbed_gmm.predict(X)

# Step 7: Calculate intra-cluster distances before and after perturbation
def intra_cluster_distance(X, labels, means, cluster_id):
    points_in_cluster = X[labels == cluster_id]
    if len(points_in_cluster) == 0:
        return None
    distances = np.linalg.norm(points_in_cluster - means[cluster_id], axis=1)
    return np.mean(distances)

# Calculate intra-cluster distances before perturbation
initial_intra_distances = {i: intra_cluster_distance(X, gmm_labels, gmm_means, i) for i in range(3)}
# Calculate intra-cluster distances after perturbation
post_perturbation_intra_distances = {i: intra_cluster_distance(X, perturbed_labels, perturbed_means, i) for i in range(3)}


# Step 8: Validate using AMRs
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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import mode
from sklearn.datasets import make_blobs
from sklearn.datasets import load_wine

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import mode

# Step 1: Load the Wine dataset
# Load Digits dataset
# Load Iris dataset
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y_true = iris.target  

# Step 2: Apply K-Means clustering

n_clusters = 20

kmeans = KMeans(n_clusters=n_clusters, init = 'random',random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Step 3: Map clusters to true labels
def map_clusters_to_labels(cluster_labels, true_labels):
    label_mapping = {}
    for cluster in np.unique(cluster_labels):
        # Get the true labels for points in this cluster
        cluster_points = true_labels[cluster_labels == cluster]

        # Debugging print to see if any clusters are empty or unexpected
        print(f"Cluster {cluster}: {len(cluster_points)} points")
        
        # Check if the cluster has points, otherwise skip
        if len(cluster_points) == 0:
            print(f"Warning: Cluster {cluster} is empty.")
            continue
        
        # Safeguard around mode calculation
        mode_result = mode(cluster_points)
        
        # Since mode_result.mode is scalar, no need to use len()
        true_label_for_cluster = mode_result.mode.item()  # Use .item() to extract the scalar value
        
        label_mapping[cluster] = true_label_for_cluster
    
    return label_mapping


label_mapping = map_clusters_to_labels(cluster_labels, y_true)
mapped_labels = np.array([label_mapping[cluster] for cluster in cluster_labels])

# Step 4: Evaluate performance metrics
conf_matrix = confusion_matrix(y_true, mapped_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Print Classification Report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_true, mapped_labels))

# Overall Accuracy
accuracy = accuracy_score(y_true, mapped_labels)
print(f"Accuracy: {accuracy:.4f}")

# Step 5: Visualize clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


def intra_cluster_distance(X, labels, cluster_center, cluster_id):
    points_in_cluster = X[labels == cluster_id]
    if len(points_in_cluster) == 0:
        return None
    distances = np.linalg.norm(points_in_cluster - cluster_center, axis=1)
    return np.mean(distances)

# Function to calculate the distance between one cluster and another cluster's center
def distance_to_other_cluster_center(X, labels, center, cluster_id):
    points_in_cluster = X[labels == cluster_id]
    if len(points_in_cluster) == 0:
        return None
    distances = np.linalg.norm(points_in_cluster - center, axis=1)
    return np.mean(distances)

# Store the distances between all clusters dynamically in a dictionary
distance_matrix = {}
initial_intra_distances = {i: intra_cluster_distance(X, cluster_labels, cluster_centers[i], i) for i in range(n_clusters)}
print(initial_intra_distances)
# Loop through all clusters
for i in range(n_clusters):  # Assuming 10 clusters
    for j in range(n_clusters):
        if i != j:
            # Calculate distance from cluster i to center of cluster j
            distance_matrix[(i, j)] = distance_to_other_cluster_center(X, cluster_labels, cluster_centers[j], cluster_id=i)

# Now, for each cluster pair, compare the distances and print results
for i in range(n_clusters):
    for j in range(n_clusters):
        if i != j:
            distance_i_to_j = distance_matrix[(i, j)]  # Distance from cluster i to center of cluster j
            
            # Compare with intra-cluster distance (initial distance of cluster i)
            if distance_i_to_j > initial_intra_distances[i]:
                status = 'pass'
            else:
                status = 'fail'
            
            print(f"Mean distance between points in cluster {i} and the center of cluster {j}: {distance_i_to_j:.4f} *** {status}")



plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title("K-Means Clustering on Wine Dataset (PCA Projection)")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from scipy.stats import mode
import numpy as np
# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatten the 28x28 images into vectors of size 784
train_X = train_X.reshape(-1, 784)

# Flatten the 28x28 images into vectors of size 784
train_X = train_X.reshape(-1, 784)
test_X = test_X.reshape(-1, 784)

# Limit the data size for faster computation (optional)
train_X = train_X[:100]  # Use 10,000 samples for training
train_y = train_y[:100]
test_X = test_X[:2000]     # Use 2,000 samples for testing
test_y = test_y[:2000]

# Normalize the data by scaling pixel values between 0 and 1 (or you can use StandardScaler)
train_X = train_X / 255.0
test_X = test_X / 255.0

# Option 2: Add noise to the data
noise = np.random.normal(0, 0.5, train_X.shape)
train_X_noisy = train_X + noise + noise

# Train K-Means clustering
kmeans = KMeans(n_clusters=10,init='random', random_state=42).fit(train_X)
train_cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

test_cluster_labels = kmeans.predict(test_X)



def intra_cluster_distance(X, labels, cluster_center, cluster_id):
    """
    Calculate the intra-cluster distance for a specific cluster.
    
    Parameters:
    - X: Data points (numpy array of shape [n_samples, n_features])
    - labels: Cluster labels for each data point (numpy array of shape [n_samples])
    - cluster_center: The center of the cluster (numpy array of shape [n_features])
    - cluster_id: The ID of the cluster for which to calculate the intra-cluster distance
    
    Returns:
    - The mean Euclidean distance between points in the cluster and the cluster center.
      If the cluster has no points, returns None.
    """
    # Get all points in the given cluster
    points_in_cluster = X[labels == cluster_id]
    
    # Check if there are any points in the cluster
    if len(points_in_cluster) == 0:
        return None  # No points in this cluster
    
    # Calculate the Euclidean distances between points and the cluster center
    distances = np.linalg.norm(points_in_cluster - cluster_center, axis=1)
    
    # Return the mean distance (intra-cluster distance)
    return np.mean(distances)




# Initial intra-cluster distances for reference (only for each cluster's points)
initial_intra_distances = {i: intra_cluster_distance(test_X, test_cluster_labels, cluster_centers[i], i) for i in range(10)}


print(initial_intra_distances)

# Function to calculate intra-cluster distance
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

# Loop through all clusters
for i in range(10):  # Assuming 10 clusters
    for j in range(10):
        if i != j:
            # Calculate distance from cluster i to center of cluster j
            distance_matrix[(i, j)] = distance_to_other_cluster_center(train_X, train_cluster_labels, cluster_centers[j], cluster_id=i)

# Now, for each cluster pair, compare the distances and print results
for i in range(10):
    for j in range(10):
        if i != j:
            distance_i_to_j = distance_matrix[(i, j)]  # Distance from cluster i to center of cluster j
            
            # Compare with intra-cluster distance (initial distance of cluster i)
            if distance_i_to_j > initial_intra_distances[i]:
                status = 'pass'
            else:
                status = 'fail'
            
            print(f"Mean distance between points in cluster {i} and the center of cluster {j}: {distance_i_to_j:.4f} *** {status}")

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

# Create a mapping from clusters to actual labels
label_mapping = map_clusters_to_labels(train_cluster_labels, train_y)

# Map the predicted clusters to the actual labels using the mapping
mapped_test_labels = np.array([label_mapping[cluster] for cluster in test_cluster_labels])

# Step 4: Generate Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(test_y, mapped_test_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy, Precision, Recall, F1-Score
print("\nClassification Report:")
print(classification_report(test_y, mapped_test_labels))

# Overall Accuracy
accuracy = accuracy_score(test_y, mapped_test_labels)
print(f"Accuracy: {accuracy:.4f}")

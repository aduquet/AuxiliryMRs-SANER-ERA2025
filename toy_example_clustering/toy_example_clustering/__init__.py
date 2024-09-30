from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from scipy.stats import mode
import numpy as np
import joblib

# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatten the 28x28 images into vectors of size 784
train_X = train_X.reshape(-1, 784)
test_X = test_X.reshape(-1, 784)

# Limit the data size for faster computation (optional)
train_X = train_X[:10000]  # Use 10,000 samples for training
train_y = train_y[:10000]
test_X = test_X[:2000]     # Use 2,000 samples for testing
test_y = test_y[:2000]

# Step 1: Train K-means clustering on training data
kmeans = KMeans(n_clusters=10, random_state=42).fit(train_X)
train_cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Step 2: Predict cluster labels on the test data using the trained model
test_cluster_labels = kmeans.predict(test_X)

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

joblib.dump(kmeans, 'kmeans_mnist_model_v1.joblib')
print("Model saved successfully.")
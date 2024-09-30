import joblib
from tensorflow.keras.datasets import mnist

# Load the saved K-Means model
kmeans_loaded = joblib.load('kmeans_mnist_model_v1.joblib')
print("Model loaded successfully.")

# Load new MNIST data
(_, _), (test_X, _) = mnist.load_data()

# Flatten the 28x28 images into vectors of size 784
test_X = test_X.reshape(-1, 784)
test_X = test_X[:2000]  # Use 2,000 samples for testing

# Use the loaded K-Means model to predict cluster labels on new data
test_cluster_labels = kmeans_loaded.predict(test_X)

# Print some predicted cluster labels
print("Predicted cluster labels:", test_cluster_labels[:10])

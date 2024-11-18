import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import random

# Step 1: Load the image
image_path = 'what-is-image-compression-and-how-does-it-work.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Compress and decompress the image (using OpenCV, simulate this as JPEG compression)
compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Set JPEG compression quality
_, compressed_image = cv2.imencode('.jpg', original_image, compression_params)
decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_GRAYSCALE)

# Step 3: Add noise to the decompressed image (Output Transformation)
def add_noise_to_image(image, noise_factor=0.02):
    noisy_image = image.copy()
    h, w = noisy_image.shape
    num_noise_pixels = int(noise_factor * h * w)  # Noise factor controls how many pixels are changed

    # Randomly choose 'num_noise_pixels' to alter
    for _ in range(num_noise_pixels):
        x, y = random.randint(0, h - 1), random.randint(0, w - 1)
        noisy_image[x, y] = random.randint(0, 255)  # Set random pixel value

    return noisy_image

noisy_decompressed_image = add_noise_to_image(decompressed_image, noise_factor=0.02)

# Step 4: Calculate SSIM and MSE (Auxiliary Function)
def calculate_ssim(image1, image2):
    return ssim(image1, image2)

def calculate_mse(image1, image2):
    return mean_squared_error(image1, image2)

# Initial SSIM and MSE between original and decompressed images
ssim_before_noise = calculate_ssim(original_image, decompressed_image)
mse_before_noise = calculate_mse(original_image, decompressed_image)

# SSIM and MSE between original image and noisy decompressed image
ssim_after_noise = calculate_ssim(original_image, noisy_decompressed_image)
mse_after_noise = calculate_mse(original_image, noisy_decompressed_image)

# Step 5: Output the results
print(f"SSIM before noise: {ssim_before_noise:.4f}")
print(f"MSE before noise: {mse_before_noise:.4f}")

print(f"SSIM after noise: {ssim_after_noise:.4f}")
print(f"MSE after noise: {mse_after_noise:.4f}")

# Step 6: Validation based on MR
if ssim_after_noise < ssim_before_noise:
    print("Validation passed: SSIM decreased after noise.")
else:
    print("Validation failed: SSIM did not decrease as expected.")

if mse_after_noise > mse_before_noise:
    print("Validation passed: MSE increased after noise.")
else:
    print("Validation failed: MSE did not increase as expected.")

# ******************************** 
print('failing example')

# Step 1: Load the original image
 # Replace with your image path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Compress the image with aggressive settings
compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 1]  # Very low quality JPEG compression (high loss)
_, compressed_image = cv2.imencode('.jpg', original_image, compression_params)
decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_GRAYSCALE)

# Step 3: Add minor noise to the decompressed image (Output Transformation)
def add_noise_to_image(image, noise_factor=0.01):
    noisy_image = image.copy()
    h, w = noisy_image.shape
    num_noise_pixels = int(noise_factor * h * w)  # Noise factor controls how many pixels to change

    # Randomly modify a small number of pixel values
    for _ in range(num_noise_pixels):
        x, y = random.randint(0, h - 1), random.randint(0, w - 1)
        noisy_image[x, y] = random.randint(0, 255)

    return noisy_image

noisy_decompressed_image = add_noise_to_image(decompressed_image, noise_factor=0.01)

# Step 4: Calculate SSIM and MSE between original and decompressed images
def calculate_ssim(image1, image2):
    return ssim(image1, image2)

def calculate_mse(image1, image2):
    return mean_squared_error(image1, image2)

# SSIM and MSE between original and decompressed image
ssim_before_noise = calculate_ssim(original_image, decompressed_image)
mse_before_noise = calculate_mse(original_image, decompressed_image)

# SSIM and MSE between original image and noisy decompressed image
ssim_after_noise = calculate_ssim(original_image, noisy_decompressed_image)
mse_after_noise = calculate_mse(original_image, noisy_decompressed_image)

# Step 5: Output the results
print(f"SSIM before noise: {ssim_before_noise:.4f}")
print(f"MSE before noise: {mse_before_noise:.4f}")

print(f"SSIM after noise: {ssim_after_noise:.4f}")
print(f"MSE after noise: {mse_after_noise:.4f}")

# Step 6: Validation with AMR
if ssim_after_noise < ssim_before_noise:
    print("Validation passed: SSIM decreased after noise.")
else:
    print("Validation failed: SSIM did not decrease as expected.")

if mse_after_noise > mse_before_noise:
    print("Validation passed: MSE increased after noise.")
else:
    print("Validation failed: MSE did not increase as expected.")

# ****
# 
# import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the original image
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to progressively compress an image at decreasing quality
def compress_image_progressively(image, quality_levels):
    compressed_images = []
    for quality in quality_levels:
        compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, compressed = cv2.imencode('.jpg', image, compression_params)
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)
        compressed_images.append(decompressed)
    return compressed_images

# Define quality levels for compression
quality_levels = [90, 70, 50, 30, 10, 5, 2]  # Gradual degradation of quality

# Get progressively compressed images
compressed_images = compress_image_progressively(original_image, quality_levels)

# Function to calculate SSIM and MSE for each decompressed image
def evaluate_images(original, compressed_images):
    ssim_scores = []
    mse_scores = []
    for img in compressed_images:
        ssim_score = ssim(original, img)
        mse_score = mean_squared_error(original, img)
        ssim_scores.append(ssim_score)
        mse_scores.append(mse_score)
    return ssim_scores, mse_scores

# Evaluate all compressed images
ssim_scores, mse_scores = evaluate_images(original_image, compressed_images)

# Output results
for i, quality in enumerate(quality_levels):
    print(f"Quality {quality}: SSIM = {ssim_scores[i]:.4f}, MSE = {mse_scores[i]:.4f}")

# Plot SSIM and MSE over quality levels
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(quality_levels, ssim_scores, marker='o')
plt.title('SSIM vs. Compression Quality')
plt.xlabel('JPEG Quality')
plt.ylabel('SSIM')

plt.subplot(1, 2, 2)
plt.plot(quality_levels, mse_scores, marker='o', color='r')
plt.title('MSE vs. Compression Quality')
plt.xlabel('JPEG Quality')
plt.ylabel('MSE')

plt.tight_layout()
plt.show()

# e with your high-detail image path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Compress the image with extreme low quality
compression_params = [int(cv2.IMWRITE_JPEG_QUALITY), 2]  # Very low quality
_, compressed_image = cv2.imencode('.jpg', original_image, compression_params)
decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_GRAYSCALE)

# Step 3: Calculate SSIM and MSE between original and decompressed images
ssim_score = ssim(original_image, decompressed_image)
mse_score = mean_squared_error(original_image, decompressed_image)

# Output the SSIM and MSE results
print(f"SSIM: {ssim_score:.4f}")
print(f"MSE: {mse_score:.4f}")

# Step 4: Visual comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(decompressed_image, cmap='gray')
plt.title(f'Decompressed Image (Quality = 2)')
plt.show()

 


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
import pickle

#Load and Preprocess Images

# Define paths
data_dir = "Segmented Medicinal Leaf Images"
classes = os.listdir(data_dir)  # List of plant species folders
img_size = 224

# Load images and labels
images = []
labels = []

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))  # Resize
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)
        labels.append(class_idx)  # Assign numeric label

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")



def extract_features(image):
    # Convert float64 image back to uint8 (0-255)
    image_uint8 = (image * 255).astype('uint8')  # Critical fix!
    
    # Color: 3D HSV histogram
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Texture: GLCM Contrast
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    # Shape: Hu Moments + Aspect Ratio
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Handle empty contours (edge case)
    if len(contours) == 0:
        hu = np.zeros(7)  # Default Hu moments
        aspect_ratio = 1.0  # Default ratio
    else:
        cnt = max(contours, key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1e-5)  # Avoid division by zero
    
    return np.hstack([hist, [contrast], hu, [aspect_ratio]])

# Extract features for all images
X_train_dt = np.array([extract_features(img) for img in X_train])
X_val_dt = np.array([extract_features(img) for img in X_val])
X_test_dt = np.array([extract_features(img) for img in X_test])


# Save CNN-ready data
np.savez("cnn_data.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

# Save Decision Tree features
with open("dt_features.pkl", "wb") as f:
    pickle.dump({"X_train_dt": X_train_dt, "y_train": y_train,
                 "X_test_dt": X_test_dt, "y_test": y_test}, f)
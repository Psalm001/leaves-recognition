import cv2
import numpy as np
import joblib
import os
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import tensorflow as tf # Import TensorFlow for CNN model loading
from skimage.feature import graycomatrix, graycoprops # Needed for GLCM

# --- Feature Extraction Functions (MUST match training scripts) ---

# 1. Feature extraction for the pure Decision Tree model (36 features)
def extract_features_for_dt(image):
    """
    Extracts color, texture, and shape features from an image.
    This function is IDENTICAL to the 'extract_features' function in your
    decision_tree.py script (which produces 36 features).
    """
    image_uint8 = (image * 255).astype('uint8')

    # Color Features: H and S histograms
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()

    # Texture Features: GLCM for contrast, homogeneity, ASM at distances 1 and 3
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    texture_features = []
    for distance in [1, 3]:
        glcm = graycomatrix(gray, distances=[distance], angles=[0], levels=256, symmetric=True)
        props = ['contrast', 'homogeneity', 'ASM']
        texture_features.extend([graycoprops(glcm, prop)[0, 0] for prop in props])

    # Shape Features: Hu Moments, Aspect Ratio, Circularity, Log Area
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1e-5)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        shape_features = np.concatenate([
            hu_moments,
            [aspect_ratio, circularity, np.log(area + 1e-5)]
        ])
    else:
        shape_features = np.zeros(10) # 7 Hu moments + 3 custom shape features

    return np.concatenate([hist_h, hist_s, texture_features, shape_features])

# 2. Feature extraction for the CNN-Random Forest hybrid model (77 features)
def extract_hybrid_features_for_cnn_rf(image, cnn_feature_extractor_model):
    """
    Extracts hybrid features: traditional (color, texture) + CNN features.
    This function is IDENTICAL to the 'extract_hybrid_features' function
    in your cnn_alg.py script.
    """
    image_uint8 = (image * 255).astype('uint8')
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)

    # Traditional features
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 256]).flatten()

    try:
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
    except Exception: # Broad exception for GLCM issues as in cnn_alg.py
        contrast = 0.0

    # CNN features (flatten last conv layer output)
    # Ensure image has batch dimension (1, H, W, C) for CNN prediction
    # Assuming cnn_feature_extractor_model expects 224x224x3 based on data_prep.py
    # but the cnn_alg.py defines it as 128x128x3. This needs to be consistent.
    # For now, let's assume the input image (preprocessed) is resized to what CNN expects.
    # If the CNN feature extractor was trained on 128x128, the 'image' passed here needs to be 128x128.
    cnn_input_image = cv2.resize((image * 255).astype('uint8'), (128, 128)) / 255.0
    cnn_features = cnn_feature_extractor_model.predict(cnn_input_image[np.newaxis, ...], verbose=0).flatten()

    return np.concatenate([hist_h, [contrast], cnn_features])

# --- Load Trained Models ---
def load_decision_tree_model(model_path):
    """Loads the trained Decision Tree pipeline."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Decision Tree model not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading Decision Tree model: {e}")
        return None

def load_cnn_rf_hybrid_model(cnn_feature_extractor_path, rf_classifier_path):
    """
    Loads both the CNN feature extractor and the Random Forest classifier.
    """
    cnn_fe = None
    rf_clf = None

    try:
        cnn_fe = tf.keras.models.load_model(cnn_feature_extractor_path, compile=False)
        print(f"Loaded CNN Feature Extractor: {cnn_feature_extractor_path}")
    except FileNotFoundError:
        print(f"Error: CNN Feature Extractor not found at {cnn_feature_extractor_path}")
    except Exception as e:
        print(f"Error loading CNN Feature Extractor: {e}")

    try:
        rf_clf = joblib.load(rf_classifier_path)
        print(f"Loaded Random Forest Classifier: {rf_classifier_path}")
    except FileNotFoundError:
        print(f"Error: Random Forest Classifier not found at {rf_classifier_path}")
    except Exception as e:
        print(f"Error loading Random Forest Classifier: {e}")

    return cnn_fe, rf_clf

# --- Load Class Labels ---
def load_class_labels(data_dir="Segmented Medicinal Leaf Images"):
    """
    Loads class names based on directory structure.
    It's highly recommended to save these labels during training (e.g., in data_prep.py)
    and load them for prediction to ensure consistency, especially if the order
    of os.listdir changes or non-class folders appear.
    """
    if os.path.exists(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        return classes
    else:
        print(f"Warning: Data directory '{data_dir}' not found. Cannot load class labels dynamically.")
        # Fallback - Replace with your actual class names if known
        return [f"Class_{i}" for i in range(10)] # Example placeholder

# --- Image Preprocessing ---
def preprocess_image(img_path, img_size=(224, 224)):
    """
    Loads and preprocesses a single image.
    Resizes to a standard size (224x224 for data_prep.py output) and normalizes.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not open or find the image at {img_path}")
        return None
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# --- Prediction Functions ---
def predict_with_decision_tree(model, image, class_labels):
    """
    Predicts the class of an image using the Decision Tree pipeline.
    """
    if model is None:
        return "Decision Tree model not loaded."

    start_time = time.time()
    features = extract_features_for_dt(image)
    features = features.reshape(1, -1) # Reshape for single sample prediction

    # Sanity check for feature count
    if hasattr(model.named_steps['scaler'], 'n_features_in_') and \
       features.shape[1] != model.named_steps['scaler'].n_features_in_:
        return (f"Decision Tree Prediction FAILED: Feature count mismatch. "
                f"Extracted {features.shape[1]} features, but model expects "
                f"{model.named_steps['scaler'].n_features_in_} features.")

    try:
        prediction_index = model.predict(features)[0]
        end_time = time.time()

        if 0 <= prediction_index < len(class_labels):
            predicted_class_name = class_labels[prediction_index]
            return f"Predicted: {predicted_class_name} (Confidence: N/A - Decision Tree) | Time: {end_time - start_time:.4f}s"
        else:
            return "Prediction out of bounds."
    except Exception as e:
        return f"Decision Tree Prediction FAILED: An error occurred during prediction: {e}"


def predict_with_cnn_rf_hybrid(cnn_fe_model, rf_classifier_model, image, class_labels):
    """
    Predicts the class of an image using the CNN feature extractor + Random Forest classifier.
    """
    if cnn_fe_model is None or rf_classifier_model is None:
        return "CNN-Random Forest Hybrid model not fully loaded."

    start_time = time.time()
    features = extract_hybrid_features_for_cnn_rf(image, cnn_fe_model)
    features = features.reshape(1, -1) # Reshape for single sample prediction

    # Sanity check for feature count for RF classifier
    # RandomForestClassifier in pipeline will have a 'scaler' which needs a feature count
    if hasattr(rf_classifier_model.named_steps['scaler'], 'n_features_in_') and \
       features.shape[1] != rf_classifier_model.named_steps['scaler'].n_features_in_:
        return (f"CNN-RF Prediction FAILED: Feature count mismatch. "
                f"Extracted {features.shape[1]} features, but RF classifier expects "
                f"{rf_classifier_model.named_steps['scaler'].n_features_in_} features.")

    try:
        prediction_index = rf_classifier_model.predict(features)[0]
        end_time = time.time()

        # You might also get probabilities from RandomForest for a confidence score
        # probabilities = rf_classifier_model.predict_proba(features)[0]
        # confidence = np.max(probabilities)

        if 0 <= prediction_index < len(class_labels):
            predicted_class_name = class_labels[prediction_index]
            return f"Predicted: {predicted_class_name} (Confidence: N/A - Hybrid) | Time: {end_time - start_time:.4f}s"
        else:
            return "Prediction out of bounds."
    except Exception as e:
        return f"CNN-RF Prediction FAILED: An error occurred during prediction: {e}"


def main():
    print("--- Medicinal Leaf Recognition Prediction ---")
    print("Loading models and preparing for prediction...")

    # --- Select Image File ---
    Tk().withdraw() # Hide the main tkinter window
    file_path = askopenfilename(
        title="Select a Medicinal Leaf Image for Prediction",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("No image selected. Exiting.")
        return

    print(f"\nSelected image: {os.path.basename(file_path)}")

    # --- Load Class Labels ---
    class_labels = load_class_labels()
    if not class_labels:
        print("Could not load class labels. Predictions might be uninterpretable. Exiting.")
        return
    print(f"Loaded {len(class_labels)} class labels from '{os.path.basename(os.path.normpath('Segmented Medicinal Leaf Images'))}'.")

    # --- Load Trained Models ---
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True) # Ensure models directory exists

    # 1. Load Decision Tree Model
    dt_model = None
    dt_models = [f for f in os.listdir(model_dir) if f.startswith('decision_tree_') and f.endswith('.pkl')]
    if dt_models:
        latest_dt_model_path = os.path.join(model_dir, sorted(dt_models)[-1])
        dt_model = load_decision_tree_model(latest_dt_model_path)
    else:
        print("No Decision Tree model found. Please train it first using decision_tree.py.")

    # 2. Load CNN-Random Forest Hybrid Model
    cnn_fe_model = None
    rf_classifier_model = None
    
    # CNN feature extractor saved as .keras
    cnn_fe_files = [f for f in os.listdir(model_dir) if f.startswith('cnn_feature_extractor_') and f.endswith('.keras')]
    # Random Forest classifier saved as .pkl (part of the pipeline in cnn_alg.py)
    rf_clf_files = [f for f in os.listdir(model_dir) if f.startswith('cnn_model_') and f.endswith('.pkl')] # This is the full pipeline

    if cnn_fe_files and rf_clf_files:
        latest_cnn_fe_path = os.path.join(model_dir, sorted(cnn_fe_files)[-1])
        latest_rf_clf_path = os.path.join(model_dir, sorted(rf_clf_files)[-1])
        cnn_fe_model, rf_classifier_model = load_cnn_rf_hybrid_model(latest_cnn_fe_path, latest_rf_clf_path)
    else:
        print("No complete CNN-Random Forest hybrid model found. Please train it first using cnn_alg.py.")


    # --- Preprocess Image ---
    # The base image loaded from disk should be 224x224 as per data_prep.py
    processed_image_224 = preprocess_image(file_path, img_size=(224, 224))
    if processed_image_224 is None:
        return

    print("\n--- Predictions ---")

    # Predict with Decision Tree
    dt_result = predict_with_decision_tree(dt_model, processed_image_224, class_labels)
    print(f"Decision Tree Model: {dt_result}")
    print("-" * 50) # Separator

    # Predict with CNN-Random Forest Hybrid
    cnn_rf_result = predict_with_cnn_rf_hybrid(cnn_fe_model, rf_classifier_model, processed_image_224, class_labels)
    print(f"CNN-Random Forest Hybrid Model: {cnn_rf_result}")
    print("-" * 50) # Separator

    print("\nPrediction complete!")

if __name__ == "__main__":
    main()
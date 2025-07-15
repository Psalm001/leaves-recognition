import cv2
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Enhanced feature extraction with CNN features 
def extract_hybrid_features(image, cnn_model):
    # Basic features
    image_uint8 = (image * 255).astype('uint8')
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    
    # 1. Traditional features
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 256]).flatten()
    
    try:
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
    except:
        contrast = 0.0
    
    # 2. CNN features (flatten last conv layer)
    cnn_features = cnn_model.predict(image[np.newaxis, ...], verbose=0).flatten()
    
    return np.concatenate([hist_h, [contrast], cnn_features])

# Simple CNN feature extractor
def create_cnn_feature_extractor(input_shape=(128, 128, 3)):
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D()
    ])
    return model

def save_metrics(y_true, y_pred, model_name, timestamp, model=None):
    """Save evaluation metrics to files - identical format to Decision Tree"""
    # Create directory if it doesn't exist
    os.makedirs(f"metrics/{model_name}", exist_ok=True)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Save metrics to JSON
    with open(f"metrics/{model_name}/metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"metrics/{model_name}/confusion_matrix_{timestamp}.png")
    plt.close()

    # Save feature importance plot if available (for RandomForest)
    if model is not None and hasattr(model.named_steps['clf'], 'feature_importances_'):
        importances = model.named_steps['clf'].feature_importances_
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances)
        plt.title("Feature Importances")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.savefig(f"metrics/{model_name}/feature_importance_{timestamp}.png")
        plt.close()

def main():
    # Load data
    data = np.load("cnn_data.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    # Create CNN feature extractor
    print("Creating CNN feature extractor...")
    cnn_model = create_cnn_feature_extractor(X_train[0].shape)
    
    # Extract hybrid features
    print("Extracting hybrid features...")
    X_train_fe = np.array([extract_hybrid_features(img, cnn_model) for img in tqdm(X_train)])
    X_test_fe = np.array([extract_hybrid_features(img, cnn_model) for img in tqdm(X_test)])
    
    # Train classifier
    print("Training classifier...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    model.fit(X_train_fe, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_fe)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save metrics with timestamp (same format as Decision Tree)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_metrics(y_test, y_pred, "cnn_model", timestamp, model)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/cnn_model_{timestamp}.pkl')
    cnn_model.save(f'models/cnn_feature_extractor_{timestamp}.keras')
    print("\nModels saved successfully")

if __name__ == "__main__":
    main()
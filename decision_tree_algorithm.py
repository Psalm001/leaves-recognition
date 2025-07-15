import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import joblib
import time
from tqdm import tqdm
import os
import json
import seaborn as sns

def extract_features(image):
    image_uint8 = (image * 255).astype('uint8')
    
    # 1. Color Features
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 256]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    
    # 2. Texture Features
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    texture_features = []
    for distance in [1, 3]:
        glcm = graycomatrix(gray, distances=[distance], angles=[0], levels=256, symmetric=True)
        props = ['contrast', 'homogeneity', 'ASM']
        texture_features.extend([graycoprops(glcm, prop)[0, 0] for prop in props])
    
    # 3. Shape Features
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
        shape_features = np.zeros(10)
    
    return np.concatenate([hist_h, hist_s, texture_features, shape_features])

def create_optimized_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selector', SelectFromModel(
            DecisionTreeClassifier(max_depth=5, random_state=42),
            threshold='median'
        )),
        ('classifier', BaggingClassifier(
            estimator=DecisionTreeClassifier(  
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ),
            n_estimators=15,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

def save_metrics(y_true, y_pred, model_name, timestamp):
    """Save evaluation metrics to files"""
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

    # Save feature importance plot if available
    if hasattr(pipeline.named_steps['classifier'].estimator_, 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].estimator_.feature_importances_
        selected = pipeline.named_steps['feature_selector'].get_support()
        important_features = importances[selected]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(important_features)), important_features)
        plt.title("Selected Feature Importances")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.savefig(f"metrics/{model_name}/feature_importance_{timestamp}.png")
        plt.close()

def main():
    # Load data
    data = np.load("cnn_data.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    # Feature extraction
    print("Extracting features...")
    X_train_fe = np.array([extract_features(img) for img in tqdm(X_train)])
    X_test_fe = np.array([extract_features(img) for img in tqdm(X_test)])
    
    # Create and train model
    global pipeline  # Make pipeline available for feature importance saving
    pipeline = create_optimized_pipeline()
    print("\nTraining model...")
    pipeline.fit(X_train_fe, y_train)
    
    # Cross-validated predictions
    print("\nComputing cross-validated predictions...")
    y_train_pred = cross_val_predict(pipeline, X_train_fe, y_train, cv=3, n_jobs=-1)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Final evaluation
    y_test_pred = pipeline.predict(X_test_fe)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nCross-validated Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Save metrics with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_metrics(y_test, y_test_pred, "decision_tree", timestamp)
    
    # Save model
    model_path = f'models/decision_tree_{timestamp}.pkl'
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")   

if __name__ == "__main__":
    main()
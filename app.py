import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import time
import tempfile
from io import StringIO
from predict_leaf import (
    predict_with_decision_tree,
    predict_with_cnn_rf_hybrid,
    preprocess_image,
    load_class_labels,
    load_decision_tree_model,
    load_cnn_rf_hybrid_model
)

# Streamlit page configuration
st.set_page_config(
    page_title="A Comparative Analysis Of Convolusional Neural Network (CNN) And Decision Tree for Recognition of Medicinal Plant Leaves",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header-style {
        font-size: 20px;
        font-weight: bold;
        color: #2e8b57;
    }
    .model-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-result {
        font-size: 18px;
        font-weight: bold;
        color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.dt_model = None
    st.session_state.cnn_model = None
    st.session_state.rf_model = None
    st.session_state.class_labels = []

# App Header
st.title("🍃 A Comparative Analysis Of Convolusional Neural Network (CNN) And Decision Tree for Recognition of Medicinal Plant Leaves")
st.markdown("Upload an image of a medicinal leaf to identify its species using our machine learning models.")

# Sidebar Info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses two machine learning approaches to identify medicinal leaves:

    1. **Decision Tree Model** – Based on extracted features (color, texture, shape)  
    2. **CNN-RF Hybrid** – Uses deep features from CNN + Random Forest classification

    Final Year Project by **ABDULWASIU OLAWALE MOHAMMED**,  
    **SALAUDEEN FARUQ OLAMILEKAN**,  
    **TIAMIYU ROFIAT OMOLARA.**

    **Supervised by** **DR.MRS. R.S BABATUNDE**  
    Department of Computer Science,  
    Kwara State University, Malete, Kwara State.
    """)

    st.header("Instructions")
    st.markdown("""
    1. Upload a clear image of a single medicinal leaf  
    2. The system processes the image  
    3. View predictions from both models  
    4. Compare and evaluate results
    """)

# Main Content Layout
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a leaf image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a single medicinal leaf"
    )

    with st.expander("🖼️ Example Leaf Images"):
        st.image([
            "https://example.com/leaf1.jpg",
            "https://example.com/leaf2.jpg",
            "https://example.com/leaf3.jpg"
        ], width=100, caption=["Example 1", "Example 2", "Example 3"])

with col2:
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error opening image: {e}")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_file = tmp.name

        if not st.session_state.models_loaded:
            with st.spinner("Loading machine learning models..."):
                model_dir = 'models'

                if not os.path.exists(model_dir):
                    st.error("Model directory not found.")
                    st.stop()

                try:
                    dt_models = [f for f in os.listdir(model_dir) if f.startswith('decision_tree_') and f.endswith('.pkl')]
                    if dt_models:
                        latest_dt_model_path = os.path.join(model_dir, sorted(dt_models)[-1])
                        st.session_state.dt_model = load_decision_tree_model(latest_dt_model_path)
                    else:
                        st.warning("No Decision Tree model found.")
                except Exception as e:
                    st.error(f"Error loading Decision Tree model: {e}")

                try:
                    cnn_fe_files = [f for f in os.listdir(model_dir) if f.startswith('cnn_feature_extractor_') and f.endswith('.keras')]
                    rf_clf_files = [f for f in os.listdir(model_dir) if f.startswith('cnn_model_') and f.endswith('.pkl')]
                    if cnn_fe_files and rf_clf_files:
                        latest_cnn_fe_path = os.path.join(model_dir, sorted(cnn_fe_files)[-1])
                        latest_rf_clf_path = os.path.join(model_dir, sorted(rf_clf_files)[-1])
                        st.session_state.cnn_model, st.session_state.rf_model = load_cnn_rf_hybrid_model(
                            latest_cnn_fe_path, latest_rf_clf_path
                        )
                    else:
                        st.warning("CNN or Random Forest model files not found.")
                except Exception as e:
                    st.error(f"Error loading CNN-RF Hybrid model: {e}")

                try:
                    st.session_state.class_labels = load_class_labels()
                except Exception as e:
                    st.error(f"Error loading class labels: {e}")

                st.session_state.models_loaded = True

        processed_image = preprocess_image(temp_file)

        if processed_image is None:
            st.error("Error processing the image. Please try another image.")
            st.stop()

        tab1, tab2 = st.tabs(["Predictions", "Feature Analysis"])

        with tab1:
            st.subheader("Model Predictions")
            col_pred1, col_pred2 = st.columns(2)

            result_str = ""

            with col_pred1:
                if st.session_state.dt_model and st.session_state.class_labels:
                    with st.spinner("Decision Tree analyzing..."):
                        try:
                            start_time = time.time()
                            dt_prediction = predict_with_decision_tree(
                                st.session_state.dt_model,
                                processed_image,
                                st.session_state.class_labels
                            )
                            dt_time = time.time() - start_time

                            st.markdown("<div class='model-card'>", unsafe_allow_html=True)
                            st.markdown("<p class='header-style'>Decision Tree Model</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='prediction-result'>{dt_prediction}</p>", unsafe_allow_html=True)
                            st.metric("Processing Time", f"{dt_time:.2f} seconds")
                            st.markdown("</div>", unsafe_allow_html=True)

                            result_str += f"Decision Tree Prediction: {dt_prediction}\nProcessing Time: {dt_time:.2f} sec\n\n"
                        except Exception as e:
                            st.error(f"Decision Tree prediction error: {e}")
                else:
                    st.warning("Decision Tree model or class labels not loaded.")

            with col_pred2:
                if st.session_state.cnn_model and st.session_state.rf_model:
                    with st.spinner("CNN-RF Hybrid analyzing..."):
                        try:
                            start_time = time.time()
                            hybrid_prediction = predict_with_cnn_rf_hybrid(
                                st.session_state.cnn_model,
                                st.session_state.rf_model,
                                processed_image,
                                st.session_state.class_labels
                            )
                            hybrid_time = time.time() - start_time

                            st.markdown("<div class='model-card'>", unsafe_allow_html=True)
                            st.markdown("<p class='header-style'>CNN-RF Hybrid Model</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='prediction-result'>{hybrid_prediction}</p>", unsafe_allow_html=True)
                            st.metric("Processing Time", f"{hybrid_time:.2f} seconds")
                            st.markdown("</div>", unsafe_allow_html=True)

                            result_str += f"CNN-RF Hybrid Prediction: {hybrid_prediction}\nProcessing Time: {hybrid_time:.2f} sec\n"
                        except Exception as e:
                            st.error(f"CNN-RF Hybrid prediction error: {e}")
                else:
                    st.warning("CNN-RF Hybrid model not properly loaded.")

            if result_str:
                st.download_button("📄 Download Prediction Result", result_str, file_name="leaf_prediction.txt")

        with tab2:
            st.subheader("Feature Analysis")
            img_array = np.array(image)
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

                st.markdown("**Color Channels**")
                cols = st.columns(4)
                with cols[0]: st.image(img_array, caption="Original", use_column_width=True, clamp=True)
                with cols[1]: st.image(gray, caption="Grayscale", use_column_width=True, clamp=True)
                with cols[2]: st.image(hsv[:, :, 0], caption="Hue", use_column_width=True, clamp=True)
                with cols[3]: st.image(hsv[:, :, 1], caption="Saturation", use_column_width=True, clamp=True)

                st.markdown("**Texture Analysis**")
                # Placeholder: You can add GLCM or other texture feature computations here.
            except Exception as e:
                st.error(f"Error during feature analysis: {e}")

        os.remove(temp_file)
    else:
        st.info("Please upload a leaf image to begin analysis.")
        if os.path.exists("assets/leaf_placeholder.jpg"):
            st.image("assets/leaf_placeholder.jpg", caption="Sample Leaf", use_column_width=True)
        else:
            st.warning("Placeholder image not found.")

# Footer
st.markdown("---")
st.markdown("""
<style>
    .footer {
        font-size: 12px;
        text-align: center;
        color: #6c757d;
    }
</style>
<p class="footer">Medicinal Leaf Recognition System • Final Year Project • © 2025</p>
""", unsafe_allow_html=True)

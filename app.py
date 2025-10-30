import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import AnimalDetector
from utils import load_image, preprocess_image, draw_boxes

# Page config
st.set_page_config(page_title="Advanced Animal Detection", page_icon="🐾", layout="wide")

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1rem;
    }
    .detection-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize detector
@st.cache_resource
def load_detector():
    return AnimalDetector()

detector = load_detector()

# Main header
st.markdown('<h1 class="main-header">🐾 Advanced Animal Detection</h1>', unsafe_allow_html=True)
st.markdown("Upload an image or take a photo to detect animals with cutting-edge AI technology!")

# Sidebar for input selection
st.sidebar.title("Input Options")
input_method = st.sidebar.radio("Choose input method:", ("Upload Image", "Take Photo"))

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_column_width=True)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

with col2:
    st.subheader("Detection Results")
    if 'image_cv' in locals():
        # Preprocess image
        processed_image = preprocess_image(image_cv)

        # Detect animals
        with st.spinner("Detecting animals..."):
            results = detector.detect_animals(processed_image)

        # Draw boxes
        result_image = draw_boxes(processed_image.copy(), results)

        # Display result
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

        # Show detected animals
        detected_animals = detector.get_detected_animals(results)
        if detected_animals:
            st.markdown('<div class="detection-result">', unsafe_allow_html=True)
            st.subheader("🐾 Detected Animals:")
            for animal in detected_animals:
                st.write(f"**{animal['name'].capitalize()}** - Confidence: {animal['confidence']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No animals detected in the image. Try uploading a different image!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and YOLOv5")

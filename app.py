import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Page title
st.set_page_config(page_title="YOLO Object Detection")
st.title("YOLO Object Detection App")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Upload image
uploaded_image = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image for YOLO
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # YOLO prediction
    results = model(img_bgr)

    # Draw bounding boxes
    output_img = results[0].plot()
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    st.subheader("Detection Result")
    st.image(output_img, use_container_width=True)

import streamlit as st
import numpy as np
import cv2
import tempfile
from shape_from_shading import (
    preprocessing,
    gradients,
    integration,
    shape_detection,
    texturing,
    visualization
)

st.set_page_config(page_title="3D Reconstruction from 2D Image", layout="wide")

st.title("🧠 3D Reconstruction from a Single 2D Image")
st.markdown("Upload a 2D image and view the reconstructed 3D surface using shape-from-shading.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    image_path = tfile.name

    # --- 1. Load and Preprocess ---
    img, rgb, gray = preprocessing.load_and_preprocess(image_path)

    # --- 2. Shape Detection (Optional Masking) ---
    mask, shape_type = shape_detection.detect_common_shape(gray)

    # --- 3. Gradient Computation ---
    gray_norm = preprocessing.normalize_image(gray)
    gx, gy = gradients.compute_gradients(gray_norm)

    # --- 4. Depth Estimation ---
    depth = integration.frankotchellappa(gx, gy)
    depth_masked = preprocessing.apply_blur(depth, kernel_size=(11, 11), sigma=1)
    depth_scaled = depth_masked * 15  # exaggerate depth for better visibility

    # --- 5. Texture Replication ---
    texture_image = texturing.prepare_texture(rgb, mask)

    # --- 6. Display Original Image ---
    st.success(f"Detected shape: {shape_type}")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    # --- 7. Display 3D Output ---
    st.subheader("🔁 Reconstructed 3D View (Interactive)")
    light_mode = st.toggle("Use Light Mode Background", value=False)

    fig = visualization.get_plotly_figure(depth_scaled, texture_image, light_mode=light_mode)
    st.plotly_chart(fig, use_container_width=True)
import argparse
import numpy as np

# Import the pipeline modules
from shape_from_shading import (
    preprocessing,
    gradients,
    integration,
    shape_detection,
    texturing,
    visualization
)

def main(image_path):
    # --- Step 1: Load & Preprocess ---
    img, rgb, gray = preprocessing.load_and_preprocess(image_path)
    
    # (Optional) Detect a shape to create a mask.
    # For replicating the entire image's color, you can simply use a full mask.
    mask, shape_type = shape_detection.detect_common_shape(gray)
    print("Detected shape type:", shape_type)
    # If you prefer to use the full image, uncomment the following line:
    mask = np.ones_like(gray, dtype=np.uint8)
    
    # --- Step 2: Compute Gradients ---
    gray_norm = preprocessing.normalize_image(gray)
    gx, gy = gradients.compute_gradients(gray_norm)
    
    # --- Step 3: Depth Integration ---
    depth = integration.frankotchellappa(gx, gy)
    
    # --- Step 4: (Optional) Mask the Depth ---
    # In this approach, we replicate the entire image,
    # so we use the full mask (or simply skip multiplication).
    depth_masked = depth  # or depth * mask
    
    # Apply an additional blur if desired
    depth_masked = preprocessing.apply_blur(depth_masked, kernel_size=(11, 11), sigma=1)
    
    # --- Step 5: Prepare the Texture (Per-vertex Colors) ---
    texture_image = texturing.prepare_texture(rgb, mask)
    
    # --- Step 6: 3D Visualization using Plotly ---
    # Choose a scale factor to exaggerate relief (adjust as needed)
    scale = 15
    depth_scaled = depth_masked * scale
    visualization.plot_3d_mesh(depth_scaled, texture_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Reconstruction from a Single Image (Plotly)")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args.image_path)
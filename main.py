import argparse
import cv2
from shape_from_shading import preprocessing, gradients, integration, shape_detection, texturing, visualization

def main(image_path):
    # --- Preprocessing: Load and Enhance Image ---
    img, rgb, gray = preprocessing.load_and_preprocess(image_path)
    
    # --- Gradient Computation ---
    gray_norm = preprocessing.normalize_image(gray)
    gx, gy = gradients.compute_gradients(gray_norm)
    
    # --- Gradient Integration ---
    depth = integration.frankotchellappa(gx, gy)
    
    # --- Shape Detection ---
    mask, shape_type = shape_detection.detect_common_shape(gray)
    print("Detected shape type:", shape_type)
    depth_masked = depth * mask
    depth_masked = preprocessing.apply_blur(depth_masked, kernel_size=(11, 11), sigma=1)
    
    # --- Texture Mapping ---
    texture_facecolors = texturing.prepare_texture(rgb, mask)
    
    # --- Visualization ---
    visualization.plot_3d_surface(depth_masked, texture_facecolors, scale=15, view_elev=50, view_azim=120)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Reconstruction from a single image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()
    main(args.image_path)
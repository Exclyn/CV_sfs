# texturing.py
import cv2
import numpy as np

def prepare_texture(rgb_image, mask):
    """
    Prepare the texture by applying the mask to the color image and adding an alpha channel.
    The resulting facecolor map will be used in the 3D visualization.
    """
    # Apply mask to the RGB image
    rgb_masked = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
    texture = rgb_masked.astype(np.float32) / 255.0
    # Ensure image has an alpha channel (fully opaque)
    if texture.shape[2] == 3:
        alpha = np.ones((texture.shape[0], texture.shape[1], 1), dtype=texture.dtype)
        texture = np.dstack((texture, alpha))
    # For plot_surface, use one row and column less than the full size
    return texture[:-1, :-1, :]
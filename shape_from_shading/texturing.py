# texturing.py
import cv2
import numpy as np

def prepare_texture(rgb_image, mask):
    """
    Apply a binary mask to an RGB image. For full image replication, the mask is all ones.
    """
    rgb_masked = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
    return rgb_masked

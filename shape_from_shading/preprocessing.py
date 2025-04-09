# preprocessing.py
import cv2
import numpy as np
from . import config

def load_and_preprocess(image_path):
    """
    Load image, convert to RGB and grayscale, and apply CLAHE enhancement.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    gray_enhanced = clahe.apply(gray)
    
    # Apply bilateral filter for smoothing while preserving edges
    gray_blur = cv2.bilateralFilter(gray_enhanced, config.BILATERAL_FILTER_DIAMETER,
                                    config.BILATERAL_FILTER_SIGMA_COLOR,
                                    config.BILATERAL_FILTER_SIGMA_SPACE)
    
    return img, rgb, gray_blur

def normalize_image(gray_image):
    """
    Normalize the grayscale image to the range [0, 1].
    """
    return gray_image.astype("float32") / 255.0

def apply_blur(image, kernel_size=(11, 11), sigma=1):
    """
    Apply Gaussian blur.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)
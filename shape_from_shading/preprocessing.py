# preprocessing.py
import cv2
import numpy as np
from . import config

def load_and_preprocess(image_path):
    """
    Load image, convert it to RGB (for texture) and grayscale (for processing),
    then apply CLAHE and bilateral filtering.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, 
                            tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    gray_enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to preserve edges
    gray_blur = cv2.bilateralFilter(gray_enhanced, 
                                    d=config.BILATERAL_FILTER_DIAMETER,
                                    sigmaColor=config.BILATERAL_FILTER_SIGMA_COLOR,
                                    sigmaSpace=config.BILATERAL_FILTER_SIGMA_SPACE)
    return img, rgb, gray_blur

def normalize_image(gray_image):
    """
    Normalize a grayscale image to the [0,1] range.
    """
    return gray_image.astype("float32") / 255.0

def apply_blur(image, kernel_size=(11,11), sigma=1):
    """
    Apply Gaussian blur.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)
# gradients.py
import cv2
import numpy as np

def compute_gradients(gray_norm):
    """
    Compute x and y gradients using the Sobel operator.
    """
    gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=5)
    return gx, gy
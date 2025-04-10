# gradients.py
import cv2
import numpy as np

def compute_gradients(gray_norm):
    """
    Compute image gradients using the Scharr operator.
    """
    gx = cv2.Scharr(gray_norm, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_norm, cv2.CV_32F, 0, 1)
    return gx, gy

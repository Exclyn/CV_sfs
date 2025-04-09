# shape_detection.py
import cv2
import numpy as np
from . import config

def detect_common_shape(gray_img):
    """
    Detect common shapes (circle, triangle, quadrilateral, ellipse) in the image.
    Returns a binary mask and a string indicating the detected shape.
    Defaults to the full image mask if no distinct shape is found.
    """
    # Try circle detection
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=config.HOUGH_DP, minDist=config.HOUGH_MIN_DIST,
                               param1=config.HOUGH_PARAM1, param2=config.HOUGH_PARAM2,
                               minRadius=config.HOUGH_MIN_RADIUS, maxRadius=config.HOUGH_MAX_RADIUS)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 1, -1)
        return mask, "circle"
    
    # Fallback: use edge detection and contour analysis
    edges = cv2.Canny(gray_img, config.CANNY_THRESHOLD1, config.CANNY_THRESHOLD2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Use the largest contour
        c = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        shape = "unknown"
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "quadrilateral"
        elif len(approx) > 4:
            shape = "ellipse"
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 1, -1)
        return mask, shape
    
    # Default: use full image mask
    mask = np.ones_like(gray_img, dtype=np.uint8)
    return mask, "full image"
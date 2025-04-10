# config.py
# Configuration parameters

# Preprocessing parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BILATERAL_FILTER_DIAMETER = 9
BILATERAL_FILTER_SIGMA_COLOR = 75
BILATERAL_FILTER_SIGMA_SPACE = 75

# Edge detection parameters
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Hough Circles parameters
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 50
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 50
HOUGH_MAX_RADIUS = 300

# Gaussian Blur parameters
GAUSSIAN_KERNEL_SIZE = (11, 11)
GAUSSIAN_SIGMA = 1
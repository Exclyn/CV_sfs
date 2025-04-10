# 3D Reconstruction from a Single Image using Shape-from-Shading and Plotly

This project demonstrates a full shape-from-shading (SfS) pipeline that reconstructs a 3D surface from a single 2D image. It uses classical computer vision techniques and visualizes the output using Plotly and Streamlit.

---

##  Features

-  **Preprocessing:** CLAHE (Contrast Limited Adaptive Histogram Equalization) and bilateral filtering to enhance contrast and reduce noise while preserving edges.
-  **Gradient Estimation:** Uses the Scharr operator for sharper, more robust gradient maps.
-  **Surface Integration:** Frankot-Chellappa algorithm converts gradients into an integrable depth map.
-  **Shape Detection:** Hough Transform and contour detection to extract circular or polygonal masks (optional).
-  **Texture Mapping:** Per-vertex color extraction from the original RGB image for realistic appearance.
-  **Interactive Visualization:** 3D mesh rendered with Plotly's Mesh3d (hardware-accelerated and embeddable in Streamlit).
-  **Dark/Light Mode Toggle:** Select background style for better contrast and presentation.

---

##  Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Exclyn/CV_sfs.git
cd CV_sfs
pip install -r requirements.txt
```

## Usage (CLI)
To run the reconstruction pipeline via Python:
python main.py --image_path path/to/your/image.jpg

## Streamlit Web App
You can also run the app interactively using Streamlit:
streamlit run streamlit_app.py
Then upload an image in your browser and explore the 3D result!

## Project Structure
```
CV_sfs/
│
├── main.py                     # Command-line pipeline runner
├── streamlit_app.py            # Streamlit-based web UI
├── requirements.txt            # All required Python packages
├── README.md                   # You're reading it!
│
└── shape_from_shading/
    ├── preprocessing.py        # CLAHE, filtering, normalization
    ├── gradients.py            # Scharr gradient computation
    ├── integration.py          # Frankot-Chellappa depth recovery
    ├── shape_detection.py      # Circle/polygon mask detection
    ├── texturing.py            # RGB texture preparation
    └── visualization.py        # Plotly-based 3D rendering
``` 

## ⚠ Limitations
Assumes evenly lit grayscale-compatible images.

Depth estimation is relative, not metric.

Texture is per-vertex only — no reflectance modeling or shadow handling.


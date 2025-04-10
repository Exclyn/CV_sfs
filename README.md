# 3D Reconstruction from a Single Image (Using Plotly)

This project demonstrates a shape-from-shading pipeline that reconstructs a 3D surface from a single 2D image. The workflow includes:

1. **Preprocessing:** CLAHE and bilateral filtering to enhance and smooth the image.
2. **Gradient Computation:** Using the Scharr operator for robust gradients.
3. **Integration:** Frankot-Chellappa method to compute a depth map from gradients.
4. **Shape Detection:** Identifies a dominant shape (circle, triangle, quadrilateral, or ellipse) to isolate the object.
5. **Masking & Texturing:** Applies the shape mask and extracts per-pixel colors for 3D mesh vertices.
6. **Interactive 3D Visualization:** Plotly’s `Mesh3d` is used to provide an interactive, hardware-accelerated 3D mesh.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://https://github.com/Exclyn/CV_sfs
cd 3d_reconstruction
pip install -r requirements.txt
```
## Usage 
python main.py --image_path path/to/your/image.jpg
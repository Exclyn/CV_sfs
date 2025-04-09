# 3D Reconstruction from Single Image

This project implements a pipeline that reconstructs 3D surfaces from single 2D images using shape-from-shading techniques. The process includes image preprocessing, gradient computation, gradient integration (Frankot-Chellappa), shape detection, texture mapping, and 3D visualization.

## Features

- **Preprocessing:** Enhances contrast using CLAHE and performs edge-aware smoothing.
- **Gradient Computation:** Computes gradients using the Sobel operator.
- **Integration:** Implements the Frankot-Chellappa algorithm for robust gradient integration.
- **Shape Detection:** Detects common shapes in the image to isolate the object.
- **Texture Mapping:** Maps the original image colors onto the reconstructed 3D surface.
- **Visualization:** 3D visualization using Matplotlib.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/3d_reconstruction.git
cd 3d_reconstruction
pip install -r requirements.txt
```
## Usage 
python main.py --image_path path/to/your/image.jpg
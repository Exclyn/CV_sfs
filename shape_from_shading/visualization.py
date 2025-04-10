# visualization.py
import numpy as np
import plotly.graph_objects as go

def plot_3d_mesh(depth, rgb_image, scale_factor=1):
    """
    Build a 3D mesh from the depth map and assign per-vertex colors from the RGB image,
    then display it using Plotly.
    
    Parameters:
      - depth: 2D NumPy array representing the scaled depth map.
      - rgb_image: Original RGB image (H x W x 3) used for vertex colors.
      - scale_factor: (Optional) further scaling factor for the depth values.
    """
    h, w = depth.shape
    # Create a grid of pixel coordinates
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth * scale_factor  # depth is already scaled; this is optional

    # Flatten grids to create vertex list
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()
    vertices = np.vstack([Xf, Yf, Zf]).T

    # Generate triangle connectivity for a grid mesh
    triangles = []
    for i in range(h - 1):
        for j in range(w - 1):
            v1 = i * w + j
            v2 = i * w + (j + 1)
            v3 = (i + 1) * w + j
            v4 = (i + 1) * w + (j + 1)
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])
    triangles = np.array(triangles)
    i_idx = triangles[:, 0]
    j_idx = triangles[:, 1]
    k_idx = triangles[:, 2]

    # Flatten the rgb image and convert each color to hex string
    rgb_flat = rgb_image.reshape(-1, 3)
    def rgb_to_hex(color):
        return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    vertex_colors = [rgb_to_hex(color) for color in rgb_flat]

    # Create Plotly Mesh3d figure
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i_idx,
        j=j_idx,
        k=k_idx,
        vertexcolor=vertex_colors,
        flatshading=True,
        opacity=1.0,
        showscale=False
    )])
    
    fig.update_layout(
        title="3D Model Replicating the 2D Image",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Depth")
        ),
        width=800,
        height=800
    )
    fig.show()
import numpy as np
import plotly.graph_objects as go

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))

def apply_gamma_correction(image, gamma=1.3):
    """Enhance image colors using gamma correction."""
    corrected = np.power(image / 255.0, gamma)
    corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    return corrected

def plot_3d_mesh(depth, rgb_image, scale_factor=1):
    """
    Build a 3D mesh from the depth map and assign per-vertex colors from the RGB image,
    then display it using Plotly.
    """
    h, w = depth.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth * scale_factor

    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()
    vertices = np.vstack([Xf, Yf, Zf]).T

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

    rgb_corrected = apply_gamma_correction(rgb_image)
    rgb_flat = rgb_corrected.reshape(-1, 3)
    vertex_colors = [rgb_to_hex(c) for c in rgb_flat]

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
            xaxis=dict(title="X", backgroundcolor='rgb(255,255,255)'),
            yaxis=dict(title="Y", backgroundcolor='rgb(255,255,255)'),
            zaxis=dict(title="Depth", backgroundcolor='rgb(255,255,255)'),
            bgcolor='rgb(255,255,255)'
        ),
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(255,255,255)',
        width=800,
        height=800
    )
    fig.show()

def get_plotly_figure(depth, rgb_image, scale_factor=1, light_mode=False):
    import plotly.graph_objects as go
    import numpy as np

    h, w = depth.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth * scale_factor

    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()
    vertices = np.vstack([Xf, Yf, Zf]).T

    # Build triangle mesh
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

    rgb_flat = rgb_image.reshape(-1, 3)
    def rgb_to_hex(color):
        return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    vertex_colors = [rgb_to_hex(c) for c in rgb_flat]

    bg_color = 'rgb(255,255,255)' if light_mode else 'rgb(0,0,0)'

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
        title="3D Reconstruction",
        scene=dict(
            xaxis=dict(title="X", backgroundcolor=bg_color),
            yaxis=dict(title="Y", backgroundcolor=bg_color),
            zaxis=dict(title="Depth", backgroundcolor=bg_color),
            bgcolor=bg_color
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        width=800,
        height=800
    )
    return fig
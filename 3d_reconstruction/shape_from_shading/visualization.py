# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def plot_3d_surface(depth, facecolors, scale=15, view_elev=50, view_azim=120):
    """
    Plot the 3D surface with the provided depth and texture (facecolors).
    """
    h, w = depth.shape
    X, Y = np.meshgrid(range(w), range(h))
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, depth * scale,
                    rstride=1, cstride=1,
                    facecolors=facecolors,
                    shade=False, antialiased=True)
    ax.set_title("Enhanced 3D Reconstruction with Color Texture")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")
    ax.view_init(elev=view_elev, azim=view_azim)
    plt.tight_layout()
    plt.show()
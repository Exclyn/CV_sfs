# integration.py
import numpy as np
from scipy.fftpack import fft2, ifft2

def frankotchellappa(p, q):
    """
    Integrate gradients using the Frankot-Chellappa algorithm.
    """
    h, w = p.shape
    p_fft = fft2(p)
    q_fft = fft2(q)
    u = np.fft.fftfreq(w).reshape(1, w)
    v = np.fft.fftfreq(h).reshape(h, 1)
    denom = u**2 + v**2
    denom[0, 0] = 1  # prevent division by zero
    Z = (-1j * u * p_fft - 1j * v * q_fft) / (2 * np.pi * denom)
    z = ifft2(Z).real
    z -= z.min()
    if z.max() != 0:
        z /= z.max()
    return z

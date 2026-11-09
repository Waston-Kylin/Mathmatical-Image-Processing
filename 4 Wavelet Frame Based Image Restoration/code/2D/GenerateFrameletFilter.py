# 文件: GenerateFrameletFilter.py
import numpy as np

def GenerateFrameletFilter(frame):
    if frame == 0:  # Haar Wavelet
        D = [np.array([0, 1, 1], dtype=float)/2, np.array([0, 1, -1], dtype=float)/2, 'cc']
        R = [np.array([1, 1, 0], dtype=float)/2, np.array([-1, 1, 0], dtype=float)/2, 'cc']
    elif frame == 1:  # Piecewise Linear Framelet
        D = [
            np.array([1, 2, 1], dtype=float)/4,
            np.array([1, 0, -1], dtype=float)/4 * np.sqrt(2),
            np.array([-1, 2, -1], dtype=float)/4,
            'ccc'
        ]
        R = [
            np.array([1, 2, 1], dtype=float)/4,
            np.array([-1, 0, 1], dtype=float)/4 * np.sqrt(2),
            np.array([-1, 2, -1], dtype=float)/4,
            'ccc'
        ]
    elif frame == 3:  # Piecewise Cubic Framelet
        D = [
            np.array([1, 4, 6, 4, 1], dtype=float)/16,
            np.array([1, 2, 0, -2, -1], dtype=float)/8,
            np.array([-1, 0, 2, 0, -1], dtype=float)/16 * np.sqrt(6),
            np.array([-1, 2, 0, -2, 1], dtype=float)/8,
            np.array([1, -4, 6, -4, 1], dtype=float)/16,
            'ccccc'
        ]
        R = [
            np.array([1, 4, 6, 4, 1], dtype=float)/16,
            np.array([-1, -2, 0, 2, 1], dtype=float)/8,
            np.array([-1, 0, 2, 0, -1], dtype=float)/16 * np.sqrt(6),
            np.array([1, -2, 0, 2, -1], dtype=float)/8,
            np.array([1, -4, 6, -4, 1], dtype=float)/16,
            'ccccc'
        ]
    else:
        raise ValueError("Unsupported frame type")
    return D, R
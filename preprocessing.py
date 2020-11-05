"""Preprocessing functions for BlazeFace"""

import numpy as np


def normalize(image):
    """Normalize the input image.

    Args:
        inputs: a RGB image as numpy array.

    Returns:
        a normalized image.
    """
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalization
    return ((image / 255.0) - img_mean)/img_std

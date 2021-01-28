#!/usr/bin/env python3
"""
performs a valid convolution on grayscale images
"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = int(ceil(float(h - kh + 1)))
    output_w = int(ceil(float(w - kw + 1)))
    output = np.zeros((m, output_h, output_w))
    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * images[:, y: y + kh,
                                               x: x + kw]).sum(axis=(1, 2))
    return output

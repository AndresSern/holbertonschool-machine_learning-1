#!/usr/bin/env python3
"""
performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        *m is the number of images
        *h is the height in pixels of the images
        *w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        *kh is the height of the kernel
        *kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * images[:, y: y + kh,
                                               x: x + kw]).sum(axis=(1, 2))
    return output

#!/usr/bin/env python3
"""
performs a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    output_h = int(ceil(float(h - kh + (2 * p_h) + 1)))
    output_w = int(ceil(float(w - kw + (2 * p_w) + 1)))
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    p_h, p_w = padding

    output_h = h - kh + (2 * p_h) + 1
    output_w = w - kw + (2 * p_w) + 1
    output = np.zeros((m, output_h, output_w))
    image_padded = np.zeros((m, h + output_h, w + output_w))
    image_padded = np.pad(images, ((0, 0), (p_h, p_h), (p_w, p_w)), 'constant')
    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * image_padded[
                :, y: y + kh, x: x + kw]).sum(axis=(1, 2))
    return output

#!/usr/bin/env python3
"""
performs a same convolution on grayscale images
from math import ceil, floor
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    if kh % 2 == 1:
        p_h = (kh - 1) // 2
    else:
        p_h = kh // 2
    if kw % 2 == 1:
        p_w = (kw - 1) // 2
    else:
        p_w = kw // 2
    '''
    output_h = int(ceil(float(h - kh + (2 * p_h) + 1)))
    output_w = int(ceil(float(w - kw + (2 * p_w) + 1)))
    '''
    output_h = h
    output_w = w
    output = np.zeros((m, output_h, output_w))
    image_padded = np.zeros((m, h + 2 * p_h, w + 2 * p_w))
    image_padded = np.pad(images, ((0, 0), (p_h, p_h), (p_w, p_w)), 'constant')
    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * image_padded[
                :, y: y + kh, x: x + kw]).sum(axis=(1, 2))
    return output

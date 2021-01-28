#!/usr/bin/env python3
"""
performs a convolution on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    *images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    *kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    *padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    the image should be padded with 0’s
    *stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    sh, sw = stride
    if padding == 'valid':
        output_h = int(((h - kh + 1)/sh))
        output_w = int(((w - kw + 1)/sw))
        image_padded = np.copy(images)
    else:
        if padding == 'same':
            p_h = int(((h - 1) * sh + kh - h) / 2)
            p_w = int(((w - 1) * sw + kw - w) / 2)
        else:
            p_h, p_w = padding
        """ output_h = h and output_w = w"""
        output_h = np.int((h - kh + (2 * p_h) + 1)/sh)
        output_w = np.int((w - kw + (2 * p_w) + 1)/sw)

        image_padded = np.zeros((m, output_h, output_w))
        image_padded = np.pad(images, ((0, 0), (p_h, p_h),
                              (p_w, p_w)), 'constant')

    output = np.zeros((m, output_h, output_w))
    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] = (kernel * image_padded[:, (sh * y): (sh * y) +
                               kh, (sw * x):  (sw * x) + kw]).sum(axis=(1, 2))
    return output

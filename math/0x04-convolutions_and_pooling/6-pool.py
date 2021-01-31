#!/usr/bin/env python3
"""
performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    *images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
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

    kh, kw = kernel_shape
    m, h, w, c = images.shape

    sh, sw = stride

    output_h = int(((h - kh) / sh) + 1)
    output_w = int(((w - kw) / sw) + 1)

    output = np.zeros((m, output_h, output_w, c))
    for x in range(output_w):
        for y in range(output_h):
            img = images[:, (sh * y): (sh * y) +
                         kh, (sw * x):  (sw * x) + kw]
            if mode == 'max':
                output[:, y, x] = np.max(img, axis=(1, 2))
            else:
                output[:, y, x] = np.average(img, axis=(1, 2))
    return output

#!/usr/bin/env python3
"""
performs pooling on images
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    *A_prev is a numpy.ndarray of shape
    containing the output of the previous layer
        m is the number of images
        h_prev is the height in pixels of the images
        w_prev is the width in pixels of the images
        c_prev is the number of channels in the image
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
    m, h_prev, w_prev, c_prev = A_prev.shape

    sh, sw = stride

    output_h = int(((h_prev - kh) / sh) + 1)
    output_w = int(((w_prev - kw) / sw) + 1)

    output = np.zeros((m, output_h, output_w, c_prev))
    for x in range(output_w):
        for y in range(output_h):
            img = A_prev[:, (sh * y): (sh * y) +
                         kh, (sw * x):  (sw * x) + kw]
            if mode == 'max':
                output[:, y, x] = np.max(img, axis=(1, 2))
            else:
                output[:, y, x] = np.average(img, axis=(1, 2))
    return output

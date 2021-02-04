#!/usr/bin/env python3
"""
performs back propagation over a pooling layer of a neural network
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
    output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
    for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively

    Returns: the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    sh, sw = stride
    (m, h_new, w_new, c_new) = dA.shape
    (m, h_prev, w_prev, c) = A_prev.shape
    (kh, kw) = kernel_shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):

        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):

                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == "max":

                        a_prev_slice = a_prev[vert_start:vert_end,
                                              horiz_start:horiz_end, c]

                        mask = (a_prev_slice == np.max(a_prev_slice))

                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "avg":

                        da = dA[i, h, w, c]

                        shape = (kh, kw)
                        average = da / (kh * kw)

                        a = np.ones(shape) * average
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                c] += a
    return dA_prev

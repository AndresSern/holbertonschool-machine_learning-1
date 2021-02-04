#!/usr/bin/env python3
"""
performs back propagation over a convolutional layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer
    (Z),numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(),
    output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of
    the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer
    (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    (m, h_new, w_new, c_new) = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    (kh, kw, c_prev, c_new) = W.shape

    sh, sw = stride

    if padding == 'valid':

        ph = 0
        pw = 0

    elif padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)

    """Initialize dA_prev, dW, db with the correct shapes"""

    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1,  c_new))

    A_prev_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    dA_prev_pad = np.pad(dA_prev,
                         pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        """loop over vertical axis of the output volume"""
        for h in range(h_new):
            """loop over horizontal axis of the output volume"""
            for w in range(w_new):
                """loop over the channels of the output volume"""
                for c in range(c_new):

                    """Find the corners of the current slice"""
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    """
                    Use the corners to define the slice from a_prev_pad
                    * silce is the old prediction
                    """
                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    """
                    Update gradients for the window and the filter's
                    parameters using
                    the code formulas given above

                    dw = np.matmul(dz, self.__cache["A"+str(i-1)].T)
                    db = np.sum(dz, axis=1, keepdims=True)
                    """
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end, :] += W[
                                    :, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
            """ original da without padding"""
            dA_prev[i, :, :, :] = da_prev_pad[ph:h_prev + ph,
                                              pw: w_prev + pw, :]
    return dA_prev, dW, db

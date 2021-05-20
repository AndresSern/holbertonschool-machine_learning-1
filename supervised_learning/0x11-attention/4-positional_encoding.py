#!/usr/bin/env python3
"""
calculates the positional encoding for a transformer:
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    ARGS:
        -max_seq_len {integer} : representing the maximum sequence length
        -dm {integer} :the model depth
    Returns:
        numpy.ndarray of shape (max_seq_len, dm)
            containing the positional encoding vectors
    """

    def cal_angle(pos_i, pos_j):
        """ get i value in the vector"""
        return pos_i / np.power(10000, 2 * (pos_j // 2) / dm)

    def get_pos_i_encod_vec(pos_i):
        """get positional vector"""
        return [cal_angle(pos_i, j) for j in range(dm)]

    pos_encod_vec = np.array([get_pos_i_encod_vec(pos_i)
                              for pos_i in range(max_seq_len)])

    pos_encod_vec[:, 0::2] = np.sin(pos_encod_vec[:, 0::2])  # dim 2i
    pos_encod_vec[:, 1::2] = np.cos(pos_encod_vec[:, 1::2])  # dim 2i+1

    return pos_encod_vec

#!/usr/bin/env python3
import pandas as pd
"""create a pd.DataFrame from a np.ndarray"""


def from_numpy(array):
    """
    ARGS:
        -array : {np.ndarray}
    The columns of the pd.DataFrame should be labeled
        in alphabetical order and capitalized
    Returns:
        -the newly created pd.DataFrame
    """
    alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    df = pd.DataFrame(data=array, columns=alpha[:array.shape[1]])
    return df

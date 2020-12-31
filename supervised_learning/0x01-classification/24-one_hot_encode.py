#!/usr/bin/env python3
""" one hot encode """


def one_hot_encode(Y, classes):
    onehot_encoded = []
    for value in Y:
        letter = [0. for _ in range(classes)]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

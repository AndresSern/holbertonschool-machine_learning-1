#!/usr/bin/env python3
"""
updates a variable using the gradient
descent with momentum optimization algorithm:
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ W, dW_prev = update_variables_momentum(0.01, 0.9, W, dW, dW_prev)"""
    vdw = (beta1 * v) + ((1 - beta1)*grad)
    new = var - (alpha * vdw)
    return new, vdw

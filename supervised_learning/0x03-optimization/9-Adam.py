#!/usr/bin/env python3
"""
update_variables_Adam(0.001, 0.9, 0.99, 1e-8, W, dW, dW_prev1, dW_prev2, i + 1)
updates a variable in place using the Adam optimization algorithm:
alpha is the learning rate
beta1 is the weight used for the first moment
beta2 is the weight used for the second moment
epsilon is a small number to avoid division by zero
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
v is the previous first moment of var
s is the previous second moment of var
t is the time step used for bias correction
Returns: the updated variable, the new first moment,
and the new second moment, respectively
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ adam prop"""
    mom_dw = (beta1 * v) + ((1 - beta1)*grad)
    rms_dw = (beta2 * s) + ((1 - beta2)*grad**2)
    mom_dw_corr = mom_dw/(1 - beta1**t)
    rms_dw_corr = rms_dw/(1-beta2**t)
    w = var - alpha * (mom_dw_corr/((rms_dw_corr**0.5)+epsilon))
    return w, mom_dw, rms_dw

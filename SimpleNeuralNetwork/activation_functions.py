import numpy as np


# activation function ==> S(x) = 1/1+e^(-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # return np.exp(-x) / (1 + np.exp(-x) ** 2)
    return x * (1 - x)


def relu(z):
    return max(0., z)


def relu_deriv(z):
    return 1. if z > 0 else 0.

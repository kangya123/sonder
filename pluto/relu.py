import numpy as np

def sigmoid(x, derive=False):
    if not derive:
        return 1/(1+np.exp(-x))
    return x*(1-x)

def relu(x, derive=False):
    if not derive:
        return np.maximum(0, x)
    return (x > 0).astype(float)

def stepFunction(x):
    y = x > 0
    return y.astype(np.int)

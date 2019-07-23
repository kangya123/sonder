import numpy as np

def softmax(a):
    expA = np.exp(a)
    sumExpA = np.sum(expA)
    y = expA/sumExpA

    return y


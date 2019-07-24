# -*- coding: cp936 -*-
import numpy as np

from relu import *


rate = 0.5
noneline = sigmoid
class layer:
    def __init__(self, vec, size):
        self.w = 2*np.random.random((vec, size)) - 1
        self.x = None
        
        self.a = None
        self.z = None

    def forward(self, x):
        self.a = np.dot(x, self.w)
        self.z = noneline(self.a)
        self.x = x
        return self.z

    def backward(self, cost):
        deltaA = cost*noneline(self.z, derive=True)
        deltaW = np.dot(self.x.T, deltaA)
        self.w -= rate*deltaW
        return np.dot(deltaA, self.w.T)

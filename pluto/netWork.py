from conv import conv
from pool import pool
from relu import *
from affine import affine
from softmax import softmax

from func import *
import numpy as np
from os import path
from random import randint
import matplotlib.pyplot as plt

noneline = sigmoid

rate = 0.5
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
    

def main():
    np.random.seed(42*randint(0, 100))
    p = pool(2, 2)
    
    x = reads(path.abspath('sample'), 'jpg', 64, 64)
    x = relu(x)
    y = np.array([[1], [0], [1]])
    
    #w1 = np.random.randint(0, 2, (5, 3, 3, 3))
    w1 = randArray(0, 1, 5, 3, 3, 3)
    c1 = conv(w1, 1, 0)
    c1Out = c1.forward(x)
    pOut = p.forward(c1Out)
    r1Out = relu(pOut)

    w2 = randArray(0, 1, 5, 5, 3, 3)
    c2 = conv(w2, 1, 0)
    c2Out = c2.forward(r1Out)
    pOut = p.forward(c1Out)
    r2Out = relu(pOut)

    print r2Out.shape
    r2Out = r2Out.reshape(r2Out.shape[0], -1)
    print r2Out.shape
    
    trainTime = 1024
    lay0 = layer(18605, 10)
    layOut = layer(10, 1)

    ax = []
    ay = []
    #plt.ion()
    for i in range(trainTime):
        l0Out = lay0.forward(r2Out) # 4x4
        yOut = layOut.forward(l0Out)

        _y = yOut

        cost = _y - y
        ax.append(i)
        ay.append(np.mean(np.abs(cost)))

        lay0.backward(layOut.backward(cost))

    print 'final cost => ', np.mean(np.abs(cost))
    print 'out => \n', _y
    plt.plot(ax,ay)
    plt.show()

if __name__ == '__main__':
    main()

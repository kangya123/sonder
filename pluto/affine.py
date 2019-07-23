# -*- coding: cp936 -*-
import numpy as np

class affine:
    def __init__(self, W, b):
        self.W = W           # 节点参数
        self.b = b           # 偏置
        self.x = None        # 输入
        self.dW = None
        self.db = None

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.x = x
        print x.shape
        #print x.shape, self.W.shape
        out = np.dot(x, self.W) + self.b

        return out.reshape(out.shape[0])

'''
    def backward(self, dout):
        print self.W.T.shape, dout.shape
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


w = np.random.randint(0, 2, (10, 20, 1))

x = np.random.randint(0, 5, (1, 1, 4, 5))
'''

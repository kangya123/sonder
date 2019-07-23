# -*- coding: cp936 -*-
from func import *
import numpy as np

class pool:
    def __init__(self, poolH, poolW, stride=1, pad=0):
        self.poolH = poolH
        self.poolW = poolW
        self.stride = stride
        self.pad = pad

        self.x = None
        self.argMax = None

    def forward(self, x):
        N, C, H, W = x.shape
        # 计算输出数据大小
        outH = int(1 + (H - self.poolH)/self.stride)
        outW = int(1 + (W - self.poolW)/self.stride)

        # 展开数据
        col = im2col(x, self.poolH, self.poolW, self.stride, self.pad)
        col = col.reshape(-1, self.poolH*self.poolW)

        #print 'tmp => \n', col

        # 池化
        argMax = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # 转换
        out = out.reshape(N, outH, outW, C).transpose(0, 3, 1, 2)

        self.x = x
        self.argMax = argMax

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pSize = self.poolH*self.poolW
        dMax = np.zeros((dout.size, pSize))
        dMax[np.arange(self.argMax.size), self.argMax.flatten()] = dout.flatten()
        dMax = dMax.reshape(dout.shape + (pSize, ))

        dcol = dMax.reshape(dMax.shape[0]*dMax.shape[1]*dMax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.poolH, self.poolW, self.stride, self.pad)

        return dx

def test():
    np.random.seed(0)
    x = np.random.randint(0, 5, (1, 1, 6, 5))
    print 'x => \n', x
    p = pool(2, 2, 2)
    out = p.forward(x)
    print 'out => \n', out
    dx = p.backward(out)
    print 'dx => \n', dx

if __name__ == '__main__':
    test()

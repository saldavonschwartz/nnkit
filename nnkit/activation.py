# The MIT License (MIT)
#
# Copyright (c) 2018 Federico Saldarini
# https://www.linkedin.com/in/federicosaldarini
# https://github.com/saldavonschwartz
# http://0xfede.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from . import NetOp


class ReLU(NetOp):
    """Rectified linear unit activation.

    y = max(0, x)
    """
    def __init__(self, x):
        """
        :param x: NetVar: input.
        """
        super().__init__(
            np.maximum(0, x.data),
            x
        )

    def _back(self, x):
        x.g += np.where(x.data > 0, self.g, 0)
        super()._back()


class LReLU(NetOp):
    """Leaky rectified linear activation.

    y = max(sx, x)
    """
    def __init__(self, x, s=0.01):
        """
        :param x: NetVar: input.
        :param s: negative slope.
        """
        super().__init__(
            np.maximum(s * x.data, x.data),
            x
        )

        self.cache = s

    def _back(self, x):
        s = self.cache
        x.g += np.where(x.data > 0, self.g, s)
        super()._back()


class Sigmoid(NetOp):
    """Sigmoid activation.

    y = 1/(1+e^-x)
    """
    def __init__(self, x):
        """
        :param x: NetVar: input.
        """
        super().__init__(
            1. / (1. + np.exp(-x.data)),
            x
        )

    def _back(self, x):
        x.g += self.g * self.data * (1. - self.data)
        super()._back()


class Tanh(NetOp):
    """Hyperbolic Tangent activation.

    y = (e^x - e^-x)/(e^x + e^-x)
    """
    def __init__(self, x):
        """
        :param x: NetVar: input.
        """
        ex, _ex = np.exp(x.data), np.exp(-x.data)
        super().__init__(
            (ex - _ex) / (ex + _ex),
            x
        )

    def _back(self, x):
        x.g += self.g * (1. - self.data ** 2)
        super()._back()


class SoftMax(NetOp):
    """Softmax activation.

    y = e^x_i/∑{X}:x_j, ∀x_i in X:
    """
    def __init__(self, x):
        """
        :param x: NetVar: input.
        """
        ex = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))

        super().__init__(
            ex / np.sum(ex, axis=1, keepdims=True),
            x
        )

    def _back(self, x):
        x.g += self.g * self.data * (1. - self.data)
        super()._back()



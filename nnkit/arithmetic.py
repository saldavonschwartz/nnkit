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


class Multiply(NetOp):
    """Multiplication.

    y = xw
    """
    def __init__(self, x, w):
        """
        :param x: NetVar: input 1.

        :param w: NetVar: input 2.
        """
        super().__init__(
            x.data @ w.data,
            x, w
        )

    def _back(self, x, w):
        x.g += self.g @ w.data.T
        w.g += x.data.T @ self.g
        super()._back()


class Add(NetOp):
    """Addition.

    y = x+b
    """
    def __init__(self, x, b):
        """
        :param x: NetVar: input 1.
        :param b: NetVar: input 2.
        """
        super().__init__(
            x.data + b.data,
            x, b
        )

    def _back(self, x, b):
        x.g += self.g
        b.g += np.sum(self.g, axis=0)
        super()._back()

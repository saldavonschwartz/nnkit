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


class L2Reg(NetOp):
    """L2 Norm (squared) Regularization:

    y = loss + rate/(2*B) * ∑{Params}(p_i^2)
    """
    def __init__(self, l, ps, r, t):
        """
        :param l: loss from forward pass.
        :param ps: parameters for which to compute norm.
        :param r: by how much to penalize overfitting.
        :param target: truth as passed to loss, (to derive batch size).
        """
        b = len(t.data)
        frobeniusSqr = np.sum([np.linalg.norm(p.data)**2 for p in ps])

        super().__init__(
            l.data + r / (2 * b) * frobeniusSqr,
            l, *ps
        )

        self.cache = r, b

    def _back(self, l, *ps):
        r, b = self.cache
        l.g += self.g

        for p in ps:
            p.g += r / b * p.data

        super()._back()


class Dropout(NetOp):
    """Dropout Regularization

    y = xb/k

    Where:
    b = random boolean tensor.
    k = probability used to generate b.
    """
    def __init__(self, x, k=0.8):
        """
        :param x: input.
        :param k: probability that a unit will NOT be masked.
        """
        chance = np.random.rand(*x.data.shape) < k
        super().__init__(
            x.data * chance / k,
            x
        )

        self.cache = chance

    def _back(self, x):
        chance = self.cache
        x.g += self.g * chance
        super()._back()



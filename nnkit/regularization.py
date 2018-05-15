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
    """L2 (squared) Norm Regularization.

    y = loss + λ/2B * ∑{P}(||p_i||^2)

    Where:
    . λ: regularization hyperparameter. i.e.: by how much to penalize large parameter values.
    . B: batch size.
    . P: parameters to penalize.
    . ||p_i||^2: Frobenius squared norm of a parameter.

    This node should be appended after a loss node during training.
    """
    def __init__(self, l, params, r, t):
        """
        :param l: NetVar: loss from forward pass.
        :param params: list of NetVar: parameters for which to compute norm.
        :param r: by how much to penalize overfitting.
        :param t: NetVar: target as passed to loss (to derive batch size).
        """
        b = len(t.data)
        frobeniusSquared = np.sum([np.linalg.norm(p.data)**2 for p in params])

        super().__init__(
            l.data + r / (2 * b) * frobeniusSquared,
            l, *params
        )

        self.cache = r, b

    def _back(self, l, *params):
        r, b = self.cache
        l.g += self.g

        for p in params:
            p.g += r / b * p.data

        super()._back()


class Dropout(NetOp):
    """Dropout Regularization.

    y = xb/k

    Where:
    . b: random boolean tensor which masks out some components in x.
    . k: [0-1] probability used to generate b.

    This node should be removed (or k set to 1.) when not training.
    """
    def __init__(self, x, k=0.8):
        """
        :param x: NetVar: input.
        :param k: [0-1] probability that a unit will NOT be masked.
        """
        dropout = np.random.rand(*x.data.shape) < k
        super().__init__(
            x.data * dropout / k,
            x
        )

        self.cache = dropout

    def _back(self, x):
        dropout = self.cache
        x.g += self.g * dropout
        super()._back()



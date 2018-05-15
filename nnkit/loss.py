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


class L1Loss(NetOp):
    """L1 Norm Loss.

    y = 1/B ∑{B}(|p_i-t_i|)

    Where:
    . p: prediction.
    . t: target.
    . B: batch size.
    """
    def __init__(self, p, t):
        """
        :param p: NetVar: prediction.
        :param t: NetVar: target.
        """
        super().__init__(
            np.mean(np.abs(p.data - t.data)),
            p, t
        )

        self.cache = len(t.data)

    def _back(self, p, t):
        b = self.cache
        dx = self.g * np.sign(p.data - t.data) / b
        p.g += dx
        t.g -= dx
        super()._back()


class L2Loss(NetOp):
    """L2 (squared) Norm Loss.

    y = 1/2B * ∑{B}(p_i-t_i)^2

    Where:
    . p: prediction.
    . t: target.
    . B: batch size.
    """
    def __init__(self, p, t):
        """
        :param p: NetVar: prediction.
        :param t: NetVar: target.
        """
        super().__init__(
            .5 * np.mean(np.square(p.data - t.data)),
            p, t
        )

        self.cache = len(t.data)

    def _back(self, p, t):
        b = self.cache
        dx = self.g * (p.data - t.data) / b
        p.g += dx
        t.g -= dx
        super()._back()


class CELoss(NetOp):
    """Cross Entropy Loss.

    y = -1/B * ∑{B}t_i*ln(p_i)

    Where:
    . p: prediction.
    . t: target.
    . B: batch size.
    """
    def __init__(self, p, t):
        """
        :param p: NetVar: prediction.
        :param t: NetVar: target.
        """
        b = len(t.data)

        super().__init__(
            -np.mean(np.sum(t.data * np.log(p.data), axis=1)),
            p, t
        )

        self.cache = b

    def _back(self, p, t):
        b = self.cache
        p.g += self.g * (p.data - t.data) / b
        t.g += self.g * -np.log(p.data) / b
        super()._back()


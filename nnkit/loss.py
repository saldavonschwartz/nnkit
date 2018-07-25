# The MIT License (MIT)
#
# Copyright (c) 2018 Federico Saldarini
# https://www.linkedin.com/in/federicosaldarini
# https://github.com/saldavonschwartz
# https://0xfede.io
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

"""These functions behave as loss or 'cost'/'objective', depending on whether they are passed as single or many 
sample points respectively, since in the case of many sample points, they compute the average of all loses.
"""


class L1Loss(NetOp):
    """L1 Norm Loss.

    y = |p-t|

    Where:
    . p: prediction.
    . t: target.
    """
    def __init__(self, p, t):
        """
        :param p: NetVar: prediction.
        :param t: NetVar: target.
        """
        super().__init__(
            np.mean(np.sum(np.abs(p.data - t.data), axis=1)),
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
    This is really MSE (mean squared error) with
    a 1/2 to cancel out the 2 in the derivative...

    y = 1/2 * (p-t)^2

    Where:
    . p: prediction.
    . t: target.
    """
    def __init__(self, p, t):
        """
        :param p: NetVar: prediction.
        :param t: NetVar: target.
        """
        super().__init__(
            .5 * np.mean(np.sum((p.data - t.data)**2, axis=1)),
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

    y = -t*ln(p)

    Where:
    . p: prediction.
    . t: target.
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
        t.g += self.g * -np.nan_to_num(np.log(p.data)) / b
        super()._back()


class HuberLoss(NetOp):
    """Huber Loss.
    Combination of MSE and L1.

    y = {
        1/2 * (p-t)^2       :if |p-t| <= d
        d * |p-t| - d^2/2)  :if |p-t| > d
    }

    Where:
    . p: prediction.
    . t: target.
    """
    def __init__(self, p, t, d=1):
        """
        :param p: NetVar: prediction.
        :param t: NetVar: target.
        :param d: float: delta, the threshold to select between MSE and L1.
        """
        diff = p.data - t.data
        abs = np.abs(diff)
        loss = np.where(abs <= d, 0.5 * np.square(diff), d*abs - 0.5*(d**2))

        super().__init__(
            np.mean(np.sum(loss, axis=1)),
            p, t
        )

        self.cache = diff, abs, d

    def _back(self, p, t):
        b = len(t.data)
        diff, abs, d = self.cache
        dx = np.where(abs <= d, diff, d * np.sign(diff))/b
        p.g += dx
        t.g -= dx
        super()._back()


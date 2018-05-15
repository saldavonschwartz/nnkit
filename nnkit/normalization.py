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


class BatchNorm(NetOp):
    """Batch Normalization.

    y = (x-μ_B)/σ_B * γ + β

    Where:
    . B: batch size.
    . μ_B: 1/B * ∑{B}:x = mean of a feature over a batch.
    . σ_B: √(σ_B^2 + 1e-8) = √((1/B * ∑{B}:(x-μ_B)^2) + 1e-8) = std deviation of a feature over a batch.
    . γ: learnable variance.
    . β: learnable mean.
    """
    def __init__(self, x, gamma, beta, avgVar, avgMean, useAvg):
        """
        :param x: NetVar: input.
        :param gamma: NetVar: size (|x|, 1): learnable variance (should be initialized to 1).
        :param beta: NetVar: size (|x|, 1): learnable mean (should be initialized to 0).
        :param avgVar: NetVar: size (|x|, 1): average learned variance, computed in training and used in prediction.
        :param avgMean: NetVar: size (|x|, 1): average learned mean, computed in training and used in prediction.
        :param useAvg: whether to compute batch mean and variance or use averaged values.
        """
        if useAvg:
            mean = avgMean.data
        else:
            mean = np.mean(x.data, axis=0)
            avgMean.data = 0.9 * avgMean.data + 0.1 * mean

        xCenter = x.data - mean

        if useAvg:
            var = avgVar.data
        else:
            var = np.mean(xCenter ** 2, axis=0)
            avgVar.data = 0.9 * avgVar.data + 0.1 * var

        xNormalized = xCenter / np.sqrt(var + 1e-8)

        super().__init__(
            xNormalized * gamma.data + beta.data,
            x, gamma, beta
        )

        self.cache = var, xCenter, xNormalized

    def _back(self, x, gamma, beta):
        var, xCenter, xNormalized = self.cache

        gamma.g += np.sum(self.g * xNormalized, axis=0)
        beta.g += np.sum(self.g, axis=0)

        bSize = len(x.data)
        t1 = 1/bSize * gamma.data * (var + 1e-8) ** (-1/2)
        t2 = bSize * self.g
        t3 = np.sum(self.g, axis=0)
        t4 = xCenter * (var + 1e-8) ** -1 * np.sum(self.g * xCenter, axis=0)
        x.g += t1 * (t2 - t3 - t4)

        super()._back()

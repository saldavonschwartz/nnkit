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


class Optimizer:
    """Base class for all optimizers.

    Optimizers implement the minimization step when training a neural net,
    which consists of adjusting parameter values in the direction opposite of the gradient of
    the network output w.r.t. each parameter.

    Attributes:
    . learnRate: [0,1]
    how big of an adjustment each parameter undergoes during an optimization step.

    . vars:
    parameters to adjust during an optimization step.
    """
    def __init__(self, params):
        """Create a new optimizer for a list of parameters.

        :param params: a list of NetVars to update on each optimization step.
        """
        self.learnRate = 0.1
        self.params = params

    def step(self):
        """Do one step of minimization to all parameters.

        Subclasses override this method to implement different minimization techniques.
        """
        pass


class GD(Optimizer):
    """Gradient descent with optional momentum.

    Implements the following update for each parameter p:

    m = β1m + (1-β1)df/dp
    p = p - αm

    Attributes:
    . learnRate: α ∈ [0,1]
    how big of an adjustment each parameter undergoes during an optimization step.

    . momentum: β1 ∈ [0,1]
    over how many samples the exponential moving average m takes place.
    If set to 0 momentum is disabled and the algorithm becomes simply gradient descent.
    """
    def __init__(self, params):
        super().__init__(params)
        self.momentum = 0.9
        self.m = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * p.g
            p.data -= self.learnRate * self.m[i]
            p.reset()


class Adam(GD):
    """Adaptive moment estimation.

    Implements the following update for each parameter p:

    m = β1m + (1-β1)df/dp
    r = β2r + (1-β2)(df/dp)^2
    p = p - α m/√(r + 1e-8)

    Attributes:
    . learnRate: α ∈ [0,1]
    how big of an adjustment each parameter undergoes during an optimization step.

    . momentum: β1 ∈ [0,1]
    over how many samples the exponential moving average m takes place.
    If set to 0 momentum is disabled and the algorithm becomes RMSProp

    . momentum: β2 ∈ [0,1]
    over how many samples the exponential squared moving average r takes place.
    """
    def __init__(self, params):
        super().__init__(params)
        self.rms = 0.999
        self.r = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * p.g
            self.r[i] = self.rms * self.r[i] + (1 - self.rms) * p.g**2
            p.data -= self.learnRate * (self.m[i] / np.sqrt(self.r[i] + 1e-8))
            p.reset()


def decay(step, totalSteps, minmax):
    """Exponentially decay a value over a fixed number of steps.

    Useful for implementing hyperparameter decay (i.e. learn rate decay).
    """
    slope = (minmax[0] - minmax[1]) / totalSteps
    val = max(slope * step + minmax[1], minmax[0])
    return val











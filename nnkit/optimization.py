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

    At a minimum, an optimizer has a learn rate which is usually between 0 and 1 and determines
    how big of an adjustment each parameter undergoes during an optimization step.
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

    Where:
    . m is an exponential moving average of the partial derivative of the network output
    w.r.t each parameter.

    . β1:[0,1] is the momentum term, which controls over how many samples the average takes place.
    If set to 0 momentum is disabled.

    . α:[0,1] is the learning rate.
    """
    def __init__(self, params):
        super().__init__(params)
        self.momentum = 0.9
        self.m = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            self.m[i] = self.momentum * self.m[i] + (1 - self.momentum) * p.g
            p.data -= self.learnRate * self.m[i]


class Adam(GD):
    """Adaptive moment estimation.

    Implements the following update for each parameter p:

    m = β1m + (1-β1)df/dp
    r = β2r + (1-β2)(df/dp)^2
    p = p - α m/√(r + 1e-8)


    Where:
    . m, β1 and α are the exponential moving average, momentum and learn rate as
    discussed in the gradient descent optimizer (GD).

    . r and β2 work similar to m and β1 but for a moving average of squares of the gradients.

    . m/√(r + 1e-8) is the root mean square term with momentum.

    Setting momentum to 0 turns this algorithm into RMSProp.
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


def miniBatch(data, **kwargs):
    """Helper to generate mini batches out of a whole dataset.

    :param data: A list with 2 elements each in turn being lists or numpy arrays.
    The first element should be the inputs to the network and the second the expected (truth / target)
    network output for each input.

    :param kwargs:
    . 'size': In this case as many batches as possible will be generated, each of size 'size' but
    for the last one, which will have 'size' or less elements.

    . 'count': In this case 'count' batches will be generated, all having some fixed size that allows
    for the requested count, but for the last batch, which will have 'size' or less elements.

    :return: On each iteration, a batch in the form of a tuple (data[0], data[1]).
    """
    if kwargs.get('size', None):
        bSize = kwargs['size']
    elif kwargs.get('count', None):
        bSize = int(np.ceil(len(data[0]) / kwargs['count']))

    for i in range(0, len(data[0]), bSize):
        yield data[0][i:i + bSize], data[1][i:i + bSize], i


def decay(step, totalSteps, minmax):
    """Helper to exponentially decay a value over a fixed number of time steps.

    Useful for implementing hyperparameter decay (i.e. learning rate decay).
    """
    slope = (minmax[0] - minmax[1]) / totalSteps
    val = max(slope * step + minmax[1], minmax[0])
    return val











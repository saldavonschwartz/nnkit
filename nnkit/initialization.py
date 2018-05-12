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
from . import NetVar


def xavier(*dim, n=1.):
    """Parameter initialization according to Xavier equation."""
    v = np.sqrt(n/dim[0] + 1e-8)
    return NetVar(np.random.randn(*dim) * v)


def rand1(*dim, **kwargs):
    """Parameter random initialization with optional scaling."""
    kwargs = kwargs or {'scale': 0.01}
    return NetVar(np.random.randn(*dim) * kwargs['scale'])


def rand2(*dim):
    """Parameter random initialization with clamping."""
    limit = np.sqrt(3. / (np.mean(dim) + 1e-8))
    return NetVar(np.random.uniform(low=-limit, high=limit, size=dim))


def zero(*dim):
    """Parameter zero initialization."""
    return NetVar(np.zeros(dim))



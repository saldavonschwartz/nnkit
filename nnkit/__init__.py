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
"""The element type of numpy arrays in the .data property of all nodes throughout the framework."""
dtype = np.float32


class NetVar:
    """Base class for all nodes in a network holding a value, which can be optionally learned.

    Variables are the building blocks of a network and hold data (their value )which gets
    propagated and transformed throughout a network. They also have a gradient property
    which holds the gradient of a network's output w.r.t. the variable.

    Whether a variable has its gradient computed depends on the implementation of an
    operator node taking the variable as an input.

    Whether a variable is optimized as a learnable parameter of the model depends on it
    having its gradient computed and it being passed to an optimizer's list of parameters.
    """
    def __init__(self, data=None):
        """Create a variable, optionally initializing it to a value.

        :param data: an optional list or numpy array to initialize the variable with.
        This can be modified or completely replaced after initialization too.

        A variable's data also determines the shape of its gradient (g, a numpy array with
        the gradient of the network's output w.r.t the variable).
        """
        self.g, self._data = None, None

        if data is not None:
            self.data = data

    @property
    def data(self):
        """Return this variable's value.

        :return: a numpy array.
        """
        return self._data

    @data.setter
    def data(self, data):
        """Set this variable's value

        Setting a variable's value implies resetting its gradient.

        :param data: a list or numpy array.
        """
        if data is None:
            self._data = None
        else:
            # Make data into a ndarray if it is not already:
            self._data = self._data = np.array(
                data, dtype=dtype, copy=False, ndmin=2
            )

        self.reset()

    def reset(self):
        """Reset this variable's gradient.

        Resetting a the gradient sets it to a numpy array of zeroes, with the
        shape of the variable's data.
        """
        if self.data is None:
            self.g = None
        else:
            self.g = np.zeros_like(
                self._data, dtype=dtype
            )


class NetOp(NetVar):
    """Base class for network nodes implementing an operation.

    This class should be subclassed to add specific operators to the framework.

    Operators take other variables as inputs (which become their parents in
    the network) and their values are the result of applying transformations on
    those inputs. They also provide the entry point to backpropagation of gradients.
    """
    def __init__(self, data, *parents):
        """Create an operator and compute its forward pass.

        Subclasses should implement their own initializer with a signature which makes
        any required arguments (parent nodes or otherwise) explicit.

        The subclass initializer should then:
        1. Compute its forward pass.
        2. Pass the forward pass result and any parent nodes to this initializer.
        3. Optionally save any required data for the backprop pass (i.e. in a cache property).

        :param data: the result of a subclass forward pass.

        :param parents: the parent nodes of a subclass.
        """
        super().__init__(data)
        self.parents = parents

    def back(self):
        """Backpropagate gradients through the network."""
        # Base case: gradient of operator with respect to itself is 1:
        self.g = np.ones_like(self.data, dtype)
        self._back(*self.parents)

    def _back(self, *parents):
        """
        Subclasses should override this method or provide one with explicit parents.
        The subclass method should:
        1. Update the parent's gradients.
        2. Call this implementation to backprop to the parents.

        :param parents: dummy parameter to make this method's signature compatible with
        subclasses declaring explicit parents.

        Subclasses should not pass anything to this method when calling super()._back().
        However, subclasses can override this signature in which case they will get the
        same data they already have in self.parents.
        """
        for p in [p for p in self.parents if isinstance(p, NetOp)]:
            p._back(*p.parents)


class FFN:
    """Convenience class to implement a feed forward neural network (FFN)."""
    def __init__(self, *topology):
        """Creates a feed forward network with an initial topology.

        :param topology: a list of tuples describing each layer in the network. Each layer will have as its
        first input the layer before it in the list, with the exception of the first layer, which will take
        as first input the NetVar passed to the network's call operator.

        The first and only required element in a tuple is the NetOp class for the layer (i.e.: Multiply).
        If the layer takes additional args or variables (except the out of the previous layer), these
        should follow the NetOp class.

        The topology of the network can be modified or completely replaced after creation, since this class
        recreates the underlying computation graph on each invocation of its call operator (i.e. the forward pass).

        Each time the net's forward pass executes, the net also recreates its layers property, which is a list
        of instantiated operators for that pass.
        """
        self.topology = list(topology)
        self.layers = []

    @property
    def vars(self):
        """Get the network's fixed variables.

        In general, these are the learnable parameters in the model.
        However, whether any of these variables are learned actually depends on whether
        they are passed to an optimizer's parameter list.

        :return: a list of NetVars.
        """
        return [
            p for n in self.topology for p in n[1:]
            if type(p) is NetVar
        ]

    def __call__(self, x):
        """Evaluate an input by executing the network's forward pass.

        :param x: input to the network.

        :return: the value of the last node in the network. This is isually
        the networks prediction but could also be the network loss if the last
        node in the network is a loss node (i.e.: during training).
        """
        self.layers.clear()

        for n in self.topology:
            x = n[0](x, *n[1:])
            self.layers.append(x)

        return x.data

    def back(self, reset=True):
        """Compute gradients by executing the network's backprop pass.

        :param reset: Whether node gradients be reset before doing the back pass.
        If false, gradients from this back pass will accumulate with those from previous passes.
        """
        if reset:
            self.reset()

        # Backprop starts at the end (output) of the net:
        self.layers[-1].back()

    def reset(self):
        """Reset gradients for all nodes in the network."""
        for p in self.params:
            p.reset()


# Import all other modules so one can access them by importing just nnkit
# (imported here because of circular dependencies):

from .initialization import *
from .regularization import *
from .normalization import *
from .serialization import *
from .optimization import *
from .activation import *
from .arithmetic import *
from .loss import *


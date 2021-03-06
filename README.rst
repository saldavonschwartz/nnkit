NNKit: A Python framework for creating dynamic neural networks
==============================================================

NNKit is a framework for creating and training neural network models, based on dynamic computation graphs.
See `this post <https://0xfede.io/2018/05/18/nnkit.html>`_ for more info on how the framework works.

Dependencies:
=============
- `numpy <http://www.numpy.org>`_.

Installation:
=============
You can `pip install nnkit`, in which case Numpy will also be installed.
Otherwise you can download the source and manually install numpy if necessary.


Modules:
========

The following is a list of modules, nodes and optimizers, along with the framework
version in which they were added.


activation:
-----------
* ReLU (1.0)
* LReLU (1.0)
* Sigmoid (1.0)
* Tanh (1.0)
* Softmax (1.0)

arithmetic:
-----------
* Multiply (1.0)
* Add (1.0)

loss:
-----
* L1 (1.0)
* L2 (1.0)
* Cross Entropy (1.0)
* Huber (1.4.0)

normalization:
--------------
* Batch Normalization (1.0)

regularization:
---------------
* L2 (1.0)
* Dropout (1.0)

optimization:
-------------
* Gradient descent / momentum (1.0)
* Adam / RMSProp (1.0)

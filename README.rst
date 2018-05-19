NNKit: A Python framework for creating dynamic neural networks
==============================================================

NNKit is a framework for creating and training neural network models.

It is based on the concept of dynamic computation graphs, meaning that graph topology can change at runtime. This approach allows for faster iteration and experimentation and is especially well suited to tasks with inputs of varying length,
like those in sequential models (i.e. RNNs). NNKit graphs particularly are implicitly recreated on each forward pass, as a consequence of
feeding nodes into other nodes to describe a computation.

See `this article <https://0xfede.io/2018/05/18/nnkit.html>`_ for more info on how the framework works.

Modules
=======

The following is a list of net nodes and optimizers in the framework, along with the framework
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

normalization:
--------------
* Batch Normalization (1.0)

regularization:
---------------
* L2 (1.0)
* Dropout (1.0)

training:
---------
* Gradient descent w/ momentum (1.0)
* Adam / RMSProp (1.0)


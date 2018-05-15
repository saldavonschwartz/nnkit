# NNKit: A dynamic neural network framework.

NNKit is a framework for training neural network models.

It is based on the concept of dynamic computation graphs, where graphs are implicitly generated
on each forward pass as nodes are passed into other nodes as inputs

This dynamic approach makes it practical to implement recurrent topologies for varying length
inputs and allows generating graphs with a minimal amount of code.

#### Modules
The following is a list of net nodes and optimizers in the framework, along with the framework
version in which they were added.

## activation:
* ReLU (1.0)
* LReLU (1.0)
* Sigmoid (1.0)
* Tanh (1.0)
* Softmax (1.0)

## arithmetic:
* Multiply (1.0)
* Add (1.0)

## loss:
* L1 (1.0)
* L2 (1.0)
* Cross Entropy (1.0)

## normalization:
* Batch Normalization (1.0)

## regularization:
* L2 (1.0)
* Dropout (1.0)

## training:
* Gradient descent w/ momentum (1.0)
* Adam / RMSProp (1.0)



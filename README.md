**Automatic Differentiation Library and Transformer**

In this project I implemented an automatic differentiation system from scratch to build components of a Transformer model. I implemented core differentiation operators such as MatMul, SoftMax, LayerNorm, and others for both the computation and gradient of each function. I then built an evaluator class to support forward and backwards passes of a computational graph, implementing depth first search (DFS) sorting to compute the node order. Using my custom auto-diff library, I implemented the linear layer, single-head attention layer, transformer encoder layer, and training pipeline for the transformer. Finally, I implemented a few fused operators, such as MatMul + LayerNorm and MatMul + SoftMax. The dataset used for testing in this project is MNIST, a widely used dataset for image classification tasks.

The core concepts implemented in this project are:
- Automatic Differentiation: Implemented from scratch using computational graphs.
- Gradient Computation: Defined custom gradient functions for key operators.
- Evaluator Class: Supports forward and backward passes with DFS-based sorting.
- Transformer Components: Implemented Linear Layer, Attention, and Encoder Layer.
- Training Pipeline: Developed Softmax Loss and SGD-based training.
- Fused Operators: Implemented MatMul + LayerNorm and MatMul + SoftMax for efficiency.

This is a simple implementation of a transformer, so the accuracy of the predictions is about 50 percent. To run tests and evaluate performance, run the Jupyter notebook file.

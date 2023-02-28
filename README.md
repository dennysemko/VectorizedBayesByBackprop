
 # Vectorized Implementation of Bayes By Backprop (incl. Local Reparameterization Trick) 


I have not found a vectorized implementation of the BBB algorithm on Github, and given the significant speed-up factor achieved and the notorious long training times BNNs require, I decided to make this code publicly available. This repository makes use of [PyTorch](https://pytorch.org/docs/stable/index.html) and [functorch](https://pytorch.org/functorch/stable/).

## Summary

The code and reports were written as part of the "Machine Learning Project" module at the Technical University Berlin, ultimately graded with the highest mark possible 30/30.

The project consisted in implementing the Bayes By Backprop (**BBB**) algorithm proposed by [Blundell et al.](https://arxiv.org/abs/1505.05424) in milestone 1, recreating the experiments from the paper in milestone 2, and going "beyond" the paper in milestone 3, which - among other things - meant:
 - Parallelizing/Vectorizing the BBB algorithm using the vmap higher-order function 
 - Implementing the Local Reparameterization Trick (**LRT**) as described by [Kingma, Salimans & Welling](https://arxiv.org/abs/1506.02557)

The vectorized implementation of BBB with the LRT provided a speed-up by a factor of 11 when classifying on the MNIST dataset, and a factor of 10 when tackling the simple regression task introduced in the paper by Blundell et al. 
Obviously the speedup through parallelization will heavily depend on the hardware available; we were provided an NVIDIA A100 80GB PCIe 4.0 GPU, 64 CPU cores, and 755GB of RAM for our experiments. While training a Bayesian Neural Network (**BNN**) with a multilayer perceptron structure and 3 weight-samples per epoch on the MNIST dataset initially took 10 hours to train, after vectorization and the LRT, this time was slashed to less than 1 hour.
It was observed that the increase in time as a function of number of weight-samples per epoch was linear for the classic sequential and the vectorized implementation -- but again, depending on the hardware available, your mileage may vary.

![Speedup for MNIST classification](https://github.com/dennysemko/VectorizedBayesByBackprop/blob/main/classification_speedup.png?raw=true)
![Speedup for MNIST classification](https://github.com/dennysemko/VectorizedBayesByBackprop/blob/main/classification_learning_curves.png?raw=true)

## Structure
The 'Reports' directory contains the reports written for the 3 milestones of my project.

The 'BBB' directory contains all the essential building blocks required for the BBB algorithm; priors, posteriors, and the implementation of Bayesian linear layers. 

The 'BNN' directory contains the implementation of some Bayesian Neural Nets following a simple multilayer perceptron structure. This is where the vectorization happens.

'bnn_lrt_mnist.py' is a very simple example of how to train a BNN using the MNIST dataset, it can be run via:

```bash
python3 bnn_lrt_mnist.py
```

'bnn_lrt_reg.py' is a very simple example of how to train a BNN on a regression task, it can be run via:

```bash
python3 bnn_lrt_reg.py
```

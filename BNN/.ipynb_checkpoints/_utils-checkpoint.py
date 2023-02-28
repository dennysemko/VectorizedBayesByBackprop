import numpy as np
import torch
import torch.nn as nn

def minibatch_complexity_cost(batch_idx, n_batches):
    # calculate the minibatch complexity cost for the current batch
    # this is used to scale the weight of the regularization term in the loss function
    return 2 ** (n_batches - batch_idx) / (2 ** n_batches - 1)

def nll_regression(y_hat, y, likelihood_noise=.2):
    # Calculate negative log-likelihood loss for regression problem
    # based on the Gaussian (Normal) distribution

    # Create a Normal distribution object with mean y_hat and standard deviation likelihood_noise
    dist = torch.distributions.Normal(y_hat, likelihood_noise, validate_args=False)
    # Calculate the negative log-likelihood loss by computing the log probability of each element in y
    # under the normal distribution and summing them up
    nll_loss = -dist.log_prob(y).sum()

    return nll_loss
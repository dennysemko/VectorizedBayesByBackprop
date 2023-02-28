import numpy as np
import torch
import torch.nn as nn


class GaussianPrior(nn.Module):
    def __init__(self, sigma, mu=0):
        super().__init__()
        
        # initialize the parameters of the prior distribution
        self.sigma = sigma
        self.mu = mu
        
        self.N = torch.distributions.Normal(mu, sigma, validate_args=False)

    def log_prob(self, w):
        # calculate the log probability density of the weights under the prior distribution
        # this is used to compute the regularization term in the loss function
        return self.N.log_prob(w).sum()
import numpy as np
import torch
import torch.nn as nn


class ScaleMixturePrior(nn.Module):
    def __init__(self, sigma1, sigma2, pi):
        super().__init__()
        
        # initialize the parameters of the scale mixture prior distribution
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        # create normal distributions with means 0 and variances sigma1 and sigma2, respectively
        self.N1 = torch.distributions.Normal(0, sigma1)
        self.N2 = torch.distributions.Normal(0, sigma2)
        
        # pi is the mixing coefficient that determines the weight given to each component of the mixture
        self.pi = pi

    def log_prob(self, w):
        # calculate the log probability density of the weights under the scale mixture prior distribution
        # this is used to compute the regularization term in the loss function
        p1 = torch.exp(self.N1.log_prob(w))
        p2 = torch.exp(self.N2.log_prob(w))
        return (torch.log(self.pi * p1 + (1 - self.pi) * p2)).sum()
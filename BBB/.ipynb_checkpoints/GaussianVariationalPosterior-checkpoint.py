import numpy as np
import torch
import torch.nn as nn


class GaussianVariationalPosterior(nn.Module):
    def __init__(self, mu, rho):
        super().__init__()
        
        # initialize the mean and rho (log variance) of the posterior distribution as parameters
        # these parameters will be learned during training
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        
        # create a normal distribution with mean 0 and variance 1
        self.N = torch.distributions.Normal(0, 1, validate_args=False)
        
    def sigma(self):
        # calculate the standard deviation of the posterior distribution using the log variance
        # sigma is constrained to be positive by taking the exponential of rho and adding 1
        # this ensures that sigma is always positive
        return torch.log(1 + torch.exp(self.rho))
        
    def sample(self):
        # generate a sample from the posterior distribution using the reparameterization trick
        # the reparameterization trick is used to obtain a sample that is differentiable with respect to mu and rho
        # this allows gradients to be backpropagated through the sampling operation
        eps = self.N.sample(self.mu.size())
        self.w = self.mu + self.sigma() * eps
        return self.w
    
    def log_prob(self):
        # calculate the log probability density of the weights under the posterior distribution
        # this is used to compute the log likelihood term in the loss function
        log_probs = torch.distributions.Normal(loc=self.mu, scale=self.sigma(), validate_args=False).log_prob(self.w)
        return log_probs.sum()
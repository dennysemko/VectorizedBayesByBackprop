import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from BBB.GaussianPrior import GaussianPrior
from BBB.ScaleMixturePrior import ScaleMixturePrior
from BBB.GaussianVariationalPosterior import GaussianVariationalPosterior

# Define a Bayesian linear layer
class BayesianLinearLayer(nn.Module):
    def __init__(self, n_in, n_out, pi=.5, sigma1=np.exp(-0), sigma2=np.exp(-8)):
        super().__init__()

        # Define priors on the weight and bias parameters
        self.w_prior = GaussianPrior(sigma1)
        self.bias_prior = GaussianPrior(sigma1)

        # Initialize the weight and bias parameters with Gaussian variational posteriors
        self.w_mu = torch.zeros(n_out, n_in).uniform_(-.2, .2)
        self.w_rho = torch.zeros(n_out, n_in).uniform_(-5., -4.)
        self.w_var_post = GaussianVariationalPosterior(self.w_mu, self.w_rho)

        self.bias_mu = torch.zeros(n_out).uniform_(-.2, .2)
        self.bias_rho = torch.zeros(n_out).uniform_(-5., -4.)
        self.bias_var_post = GaussianVariationalPosterior(self.bias_mu, self.bias_rho)

        self.kld = 0.

    def forward(self, x):
        # Sample weight and bias parameters from the Gaussian variational posteriors
        w = self.w_var_post.sample()
        b = self.bias_var_post.sample()

        # Compute the KL divergence between the variational posterior and prior distributions
        self.kld = (self.w_var_post.log_prob() + self.bias_var_post.log_prob()) - (self.w_prior.log_prob(w) + self.bias_prior.log_prob(b))

        # Compute the output of the linear layer with the sampled weight and bias parameters
        return F.linear(x, w, b)
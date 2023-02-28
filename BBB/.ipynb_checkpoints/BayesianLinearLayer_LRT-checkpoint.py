import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from BBB.GaussianPrior import GaussianPrior
from BBB.ScaleMixturePrior import ScaleMixturePrior
from BBB.GaussianVariationalPosterior import GaussianVariationalPosterior

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a Bayesian linear layer
class BayesianLinearLayer_LRT(nn.Module):
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
        # Local Reparameterization Trick - basically, sample from a distribution over the activations
        gamma = F.linear(x, self.w_var_post.mu)
        delta = F.linear(x**2, self.w_var_post.sigma()**2) 

        zeta_w = torch.distributions.Normal(0,1).sample(gamma.size()).to(DEVICE)
        zeta_bias = torch.distributions.Normal(0,1).sample(self.bias_var_post.mu.size()).to(DEVICE)
        # Adding 1e-32 for numerical stability
        activations = gamma + torch.sqrt(delta + 1e-32) * zeta_w + self.bias_var_post.mu + self.bias_var_post.sigma() * zeta_bias

        self.kld = self._kld_gaussians_closed_form()

        return activations
    
    def _kld_gaussians_closed_form(self):
        # Closed form solution of KLD between two univariate gaussians
        kld_w = (
            torch.log(self.w_prior.sigma / self.w_var_post.sigma())
            + ((self.w_var_post.sigma() ** 2 + (self.w_var_post.mu - self.w_prior.mu) ** 2)
               / (2 * (self.w_prior.sigma ** 2)))
            - 0.5
        ).sum()
        
        kld_bias = (
            torch.log(self.bias_prior.sigma / self.bias_var_post.sigma())
            + ((self.bias_var_post.sigma() ** 2 + (self.bias_var_post.mu - self.bias_prior.mu) ** 2)
               / (2 * (self.bias_prior.sigma ** 2)))
            - 0.5
        ).sum()
        return kld_w + kld_bias
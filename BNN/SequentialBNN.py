import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from BBB.BayesianLinearLayer import BayesianLinearLayer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequentialBNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=800, n_hidden_layers=2):
        super().__init__()
        self.n_output = n_output
        self.layers = nn.ModuleList()
        
        # Create input layer
        self.layers.append(BayesianLinearLayer(n_input, n_hidden))
        
        # Create hidden layers
        for _ in range(n_hidden_layers):
            self.layers.append(BayesianLinearLayer(n_hidden, n_hidden))
        
        # Create output layer
        self.layers.append(BayesianLinearLayer(n_hidden, n_output))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
    def kld(self):
        kld = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinearLayer):
                kld += layer.kld
        return kld
    
    def elbo(self, x, y, nll_fn, pi_i, n_samples, method):
        klds = torch.zeros(n_samples).to(DEVICE)
        nlls = torch.zeros(n_samples).to(DEVICE)
        yhats = torch.zeros(n_samples).to(DEVICE)
        
        for i in range(n_samples):
            y_hat = self.forward(x)
            klds[i] = self.kld()
            nlls[i] = nll_fn(y_hat, y)
            yhats[i] = F.softmax(y_hat, dim=1)

        return pi_i * klds.mean() + nlls.mean(), yhats.mean(axis=0)
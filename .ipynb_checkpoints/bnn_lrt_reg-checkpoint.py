import numpy as np
import time
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from BNN.VectorizedBNN_LRT import VectorizedBNN_LRT
from BNN._utils import minibatch_complexity_cost, nll_regression


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_samples, n_epochs = 10, 1500


def curve_data(low=0.0, high=0.5, num=1024, batch_size=128, train=True):
    if train == True:
        test_offset = 0
    else:
        test_offset = 2**31
    for i in range(num//batch_size):
        np.random.seed(i + test_offset)
        x = np.random.uniform(low=low, high=high, size=batch_size).astype(np.float32)
        eps = np.random.normal(loc=.0, scale=.02, size=len(x))
        y = (x + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps)) + eps).astype(np.float32)
        yield torch.tensor(x).view(batch_size, 1), torch.tensor(y).view(batch_size, 1)

        
model = VectorizedBNN_LRT(1, 1, n_hidden=400).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
nll_fn = nll_regression

print('Training a BNN for Curve Regression ({} samples per epoch), {} epochs'.format(n_samples, n_epochs))
print('Using the Local Reparameterization Trick and auto-vectorization (vmap)')
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))

epoch_times = []
MEAN_TIMES = []
for epoch in range(n_epochs):
    model.train()

    train_dist = torch.zeros(1024 // 128).to(DEVICE)
    train_loss = 0.0

    trainset = curve_data(train=True)
    testnset = curve_data(train=False)

    epoch_start_time = time.time()

    for batch_idx, (x, y) in enumerate(trainset):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pi_i = minibatch_complexity_cost(batch_idx=batch_idx, n_batches=128)
        batch_elbo, y_hat = model.elbo(x, y, nll_fn, pi_i, n_samples=n_samples)
        batch_elbo.backward()
        optimizer.step()

    epoch_times.append(time.time() - epoch_start_time)
    MEAN_TIMES.append(np.mean(epoch_times))

    if (epoch + 1) % 50 == 0:
        print('Epoch: {}'.format(epoch+1).ljust(12), 
              's/Epoch(mean): {:.4f}'.format(MEAN_TIMES[-1]).ljust(23),
              'Train ELBO: {:.2f}'.format(batch_elbo).ljust(21))
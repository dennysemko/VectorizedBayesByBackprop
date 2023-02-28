import numpy as np
import time
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from BNN.VectorizedBNN_LRT import VectorizedBNN_LRT
from BNN._utils import minibatch_complexity_cost


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "~/home/space/datasets/"
n_samples, n_epochs = 3, 600

trainset = datasets.MNIST(DATA_PATH,
                          train=True,
                          download=True,
                          transform=transforms.ToTensor())

testset = datasets.MNIST(DATA_PATH,
                         train=False,
                         download=True,
                         transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True,
                                          num_workers=0)

test_loader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=True,
                                         drop_last=True,
                                         pin_memory=True,
                                         num_workers=0)

model = VectorizedBNN_LRT(28*28, 10, n_hidden = 800).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
nll_fn = nn.CrossEntropyLoss(reduction='sum')

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

print('Training a BNN for MNIST ({} samples per epoch), {} epochs'.format(n_samples,n_epochs))
print('Using the Local Reparameterization Trick and auto-vectorization (vmap)')
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))

epoch_times = []
MEAN_TIMES = []
for epoch in range(n_epochs):
    model.train()

    train_losses = []
    valid_losses = []
    train_loss = 0.0

    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pi_i = minibatch_complexity_cost(batch_idx = batch_idx, n_batches=128)
        batch_elbo, y_hat = model.elbo(x, y, nll_fn, pi_i, n_samples=n_samples)
        y_hat = torch.argmax(y_hat, dim=1)
        train_losses.append(batch_elbo.clone().item())

        batch_elbo.backward()
        optimizer.step()

    train_loss = sum(train_losses) / len(train_losses)
    scheduler.step()

    epoch_times.append(time.time() - epoch_start_time)
    MEAN_TIMES.append(np.mean(epoch_times))

    if (epoch+1)%1==0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                pi_i = minibatch_complexity_cost(batch_idx = batch_idx, n_batches=128)
                valid_batch_elbo, preds = model.elbo(x, y, nll_fn, pi_i, n_samples=10)
                preds = torch.argmax(preds, dim=1)
                valid_losses.append(valid_batch_elbo.clone().item())

                total += len(y)
                correct += (y == preds).sum()

        eval_score = correct / total
        valid_loss = sum(valid_losses) / len(valid_losses)

        current_error = (1 - eval_score) * 100

        print('Epoch: {}'.format(epoch+1).ljust(12), 
              's/Epoch(mean): {:.4f}'.format(MEAN_TIMES[-1]).ljust(23), 
              'Train ELBO: {:.2f}'.format(train_loss).ljust(21),
              'Test ELBO: {:.2f}'.format(valid_loss).ljust(21), 
              'Test Error: {:.4f}%'.format(current_error))
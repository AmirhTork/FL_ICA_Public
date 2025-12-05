import torch
import torch.nn as nn
import torch.optim as optim

def local_training(model, data_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    losses = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for X, y in data_loader:

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            else:
                X = X.detach().clone().float()

            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            else:
                y = y.detach().clone().long()

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)

        losses.append(epoch_loss / len(data_loader.dataset))

    return losses

import torch
import torch.nn as nn
import torch.optim as optim

from network import QuadraticNetwork


# Create a simple quadratic dataset
x = torch.randn(1000, 2)
y = x[:, 0] ** 2 + 2 * x[:, 1] ** 2


# Create the network
model = QuadraticNetwork([2, 16, 16, 1])

# Loss function
loss_fn = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train
for epoch in range(1000):
    prediction = model(x).squeeze()

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")


print("Training finished!")
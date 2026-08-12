import torch.nn as nn
from model import QuadraticLayer


class QuadraticNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(
                QuadraticLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1]
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
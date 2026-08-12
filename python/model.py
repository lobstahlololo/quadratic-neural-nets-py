import torch
import torch.nn as nn


class QuadraticLayer(nn.Module):
    """
    This is a quadratic neural-network layer.

    y = Wx + V(x^2) + b
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        # Linear weights: W
        self.linear = nn.Linear(input_size, output_size)

        # Quadratic weights: V
        self.quadratic = nn.Parameter(
            torch.empty(output_size, input_size)
        )

        # Xavier initialization for the quadratic weights
        nn.init.xavier_uniform_(self.quadratic)

    def forward(self, x):
        # x^2 — square every input feature
        x_squared = x ** 2

        # Wx + V(x^2) + b
        linear_part = self.linear(x)
        quadratic_part = x_squared @ self.quadratic.T

        return linear_part + quadratic_part
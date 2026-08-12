import math
import torch
import torch.nn as nn


class QuadraticLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)

        self.quadratic = nn.Parameter(
            torch.empty(output_size, input_size)
        )

        nn.init.xavier_uniform_(self.quadratic)

    def forward(self, x):
        return self.linear(x) + (x ** 2) @ self.quadratic.T


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dimension, max_sequence_length):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length

        self.query = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False
        )

    def forward(self, x):
        # x: [batch, sequence, embedding]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention scores
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.embedding_dimension)

        # Causal mask
        seq_len = x.shape[1]

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, -1e30)

        # Softmax
        attention = torch.softmax(scores, dim=-1)

        # Weighted values
        return attention @ v


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        feedforward_dimension,
        max_sequence_length
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dimension)

        # Four separate attention layers
        self.attention = nn.ModuleList([
            AttentionLayer(
                embedding_dimension,
                max_sequence_length
            )
            for _ in range(4)
        ])

        self.norm2 = nn.LayerNorm(embedding_dimension)

        self.feedforward1 = QuadraticLayer(
            embedding_dimension,
            feedforward_dimension
        )

        self.feedforward2 = QuadraticLayer(
            feedforward_dimension,
            embedding_dimension
        )

    def forward(self, x):
        x = self.norm1(x)

        for attention in self.attention:
            x = attention(x)

        x = self.norm2(x)

        x = self.feedforward1(x)

        x = torch.relu(x)

        x = self.feedforward2(x)

        return x


class QuadraticTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension=128,
        feedforward_dimension=512,
        transformer_layers=4,
        attention_heads=4,
        max_sequence_length=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dimension,
                feedforward_dimension,
                max_sequence_length
            )
            for _ in range(transformer_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dimension)

        self.output = QuadraticLayer(
            embedding_dimension,
            vocabulary_size
        )

    def forward(self, x):
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        return self.output(x)
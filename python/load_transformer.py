import torch
from transformer import QuadraticTransformer


EMBEDDING_DIM = 128
FEEDFORWARD_DIM = 512
NUM_LAYERS = 4
ATTENTION_HEADS = 4
MAX_SEQUENCE_LENGTH = 128


def load_vocab(path="../transformer_vocab.txt"):
    vocab = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split()

            # Ignore malformed vocabulary lines
            if not parts or not parts[-1].isdigit():
                continue

            index = int(parts[-1])
            token = " ".join(parts[:-1])

            vocab[token] = index

    return vocab


def load_weights(path="../transformer_weights.bin"):
    with open(path, "rb") as f:
        data = f.read()

    if len(data) % 4 != 0:
        raise ValueError(
            f"Weight file size {len(data)} is not divisible by 4"
        )

    weights = torch.frombuffer(
        bytearray(data),
        dtype=torch.float32
    ).clone()

    print(f"Weight file: {len(data):,} bytes")
    print(f"Float32 values: {len(weights):,}")

    return weights


def take(weights, offset, count):
    end = offset + count

    if end > len(weights):
        raise ValueError(
            f"Ran past end of weight file: "
            f"{end} > {len(weights)}"
        )

    return weights[offset:end], end


def load_quadratic(layer, weights, offset):
    input_size = layer.linear.in_features
    output_size = layer.linear.out_features

    matrix_size = input_size * output_size

    # C++:
    # [quadratic][linear][bias]

    values, offset = take(
        weights,
        offset,
        matrix_size
    )

    layer.quadratic.data.copy_(
        values.reshape(input_size, output_size).T
    )

    values, offset = take(
        weights,
        offset,
        matrix_size
    )

    layer.linear.weight.data.copy_(
        values.reshape(input_size, output_size).T
    )

    values, offset = take(
        weights,
        offset,
        output_size
    )

    layer.linear.bias.data.copy_(values)

    return offset


def load_layernorm(layer, weights, offset):
    dimension = layer.normalized_shape[0]

    gamma, offset = take(
        weights,
        offset,
        dimension
    )

    beta, offset = take(
        weights,
        offset,
        dimension
    )

    layer.weight.data.copy_(gamma)
    layer.bias.data.copy_(beta)

    return offset


def load_attention(layer, weights, offset):
    dimension = layer.embedding_dimension

    matrix_size = dimension * dimension

    # C++:
    # [Q][K][V]

    for linear in (
        layer.query,
        layer.key,
        layer.value,
    ):
        values, offset = take(
            weights,
            offset,
            matrix_size
        )

        linear.weight.data.copy_(
            values.reshape(dimension, dimension).T
        )

    return offset


def main():

    print("Loading vocabulary...")

    vocab = load_vocab()

    vocabulary_size = len(vocab)

    print(f"Vocabulary size: {vocabulary_size}")

    print("Loading weights...")

    weights = load_weights()

    print("Creating PyTorch model...")

    model = QuadraticTransformer(
        vocabulary_size=vocabulary_size,
        embedding_dimension=EMBEDDING_DIM,
        feedforward_dimension=FEEDFORWARD_DIM,
        transformer_layers=NUM_LAYERS,
        attention_heads=ATTENTION_HEADS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    offset = 0

    # --------------------------------------------------
    # Embedding
    # --------------------------------------------------

    count = vocabulary_size * EMBEDDING_DIM

    values, offset = take(
        weights,
        offset,
        count
    )

    model.embedding.weight.data.copy_(
        values.reshape(
            vocabulary_size,
            EMBEDDING_DIM
        )
    )

    print("Loaded embedding")

    # --------------------------------------------------
    # Transformer blocks
    # --------------------------------------------------

    for block_number, block in enumerate(model.blocks):

        print(f"Loading block {block_number + 1}/4")

        offset = load_layernorm(
            block.norm1,
            weights,
            offset
        )

        for attention_number, attention in enumerate(
            block.attention
        ):

            offset = load_attention(
                attention,
                weights,
                offset
            )

        offset = load_layernorm(
            block.norm2,
            weights,
            offset
        )

        offset = load_quadratic(
            block.feedforward1,
            weights,
            offset
        )

        offset = load_quadratic(
            block.feedforward2,
            weights,
            offset
        )

    # --------------------------------------------------
    # Final LayerNorm
    # --------------------------------------------------

    print("Loading final LayerNorm")

    offset = load_layernorm(
        model.final_norm,
        weights,
        offset
    )

    # --------------------------------------------------
    # Output layer
    # --------------------------------------------------

    print("Loading output layer")

    offset = load_quadratic(
        model.output,
        weights,
        offset
    )

    # --------------------------------------------------
    # Verify exact match
    # --------------------------------------------------

    print()
    print(f"Weights used: {offset:,}")
    print(f"Weights available: {len(weights):,}")

    if offset != len(weights):
        raise ValueError(
            f"WEIGHT MISMATCH: "
            f"{len(weights) - offset} floats remain"
        )

    print()
    print("================================")
    print("SUCCESS!")
    print("All C++ weights loaded.")
    print("================================")

    return model, vocab


if __name__ == "__main__":
    model, vocab = main()
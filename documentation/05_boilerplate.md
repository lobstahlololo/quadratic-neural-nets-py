# Boilerplate Layers
`boilerplate/layers.h` provides shortcut functions that create `LayerArgs` for common building blocks. Using these saves you from manually setting hooks and sizes.
## SoftmaxLayer
Creates a quadratic layer with softmax activation.
```
LayerArgs SoftmaxLayer(int model_dimension);
```
- `model_dimension` — number of classes / vocabulary size
- Creates a `Quadratic` layer with `Softmax` forward hook
Example:
```
architecture.push_back(SoftmaxLayer(10)); // 10-class classifier
```
By default uses `SoftmaxDerivative` as the backward hook. To use the combined O(n) system, manually set the backward hook:
```
LayerArgs output = SoftmaxLayer(10);
output.hook_gradients = { SoftmaxForCrossEntropyLossDerivative };
architecture.push_back(output);
```
## LearnableLayerNormLayer
Creates a parametric layer with learnable layer normalisation.
```
LayerArgs LearnableLayerNormLayer(int model_dimension);
```
- `model_dimension` — feature dimension to normalise
- Creates a `Parametric` layer with 2 weights per input (gamma scale and beta shift)
- Forward: normalises to zero mean and unit variance, then applies `gamma * x_norm + beta`
- Backward: computes gradients for gamma, beta, and the input
Example:
```
architecture.push_back(LearnableLayerNormLayer(512));
```
## RMSNormLayer
Creates a parametric layer with RMS normalisation.
```
LayerArgs RMSNormLayer(int model_dimension);
```
- `model_dimension` — feature dimension
- Creates a `Parametric` layer with 1 weight per input (gamma scale only, no bias)
- Forward: normalises by RMS, then applies `gamma * x_norm`
- Faster than layer norm (no mean subtraction)
Example:
```
architecture.push_back(RMSNormLayer(512));
```
## EmbeddingLayer
Creates an embedding lookup table.
```
LayerArgs EmbeddingLayer(int vocabulary_size, int embedding_dimension);
```
- `vocabulary_size` — total number of tokens
- `embedding_dimension` — size of each embedding vector
- Creates a `Parametric` layer with `vocabulary_size * embedding_dimension` weights
- Forward: looks up `embedding_dimension` values for each input token index
- Backward: accumulates gradients into the looked-up rows
Example:
```
architecture.push_back(EmbeddingLayer(32000, 512)); // 32k vocab, 512-dim embeddings
```
## AttentionLayer
Creates a single-head self-attention layer.
```
LayerArgs AttentionLayer(int embedding_dimension, int max_sequence_length);
```
- `embedding_dimension` — size of each token's embedding
- `max_sequence_length` — maximum number of tokens in a sequence (buffer multiplier)
- Creates a `Parametric` layer with `3 * embedding_dimension * embedding_dimension` weights (Q, K, V matrices)
- Forward: projects input into Q, K, V; computes attention scores; applies softmax; multiplies by V
- Backward: computes gradients for Q, K, V weights and input
- Requires scratch space: `sequence_length^2 + 3 * sequence_length * embedding_dimension` (plus extra for training)
Example:
```
architecture.push_back(AttentionLayer(512, 256)); // 512-dim, 256-token sequences
```
## FeedForwardLayer
Creates a linear (parametric) layer with activations.
```
LayerArgs FeedForwardLayer(
    int input_dimension,
    int output_dimension,
    std::vector<HookFunc> activations,
    std::vector<HookDerivative> activation_derivatives,
    int weights_per_input_override = 0
);
LayerArgs FeedForwardLayer(
    int input_dimension,
    int output_dimension,
    HookFunc activation,
    HookDerivative activation_derivative,
    int weights_per_input_override = 0
);
```
- `input_dimension` — number of input features
- `output_dimension` — number of output features
- `activations` — list of forward hooks applied in order
- `activation_derivatives` — matching backward hooks (same order)
- `weights_per_input_override` — if non-zero, overrides the default weight count (default = `input_dimension`)
Creates a `Parametric` layer. The hooks do the actual transformation; the weights are available to hooks via `layer.weights_begin`.
Example — ReLU feed-forward block:
```
architecture.push_back(
    FeedForwardLayer(512, 2048, ReLuHook, ReLuGradHook)
);
```
Example — two activations:
```
architecture.push_back(
    FeedForwardLayer(512, 2048,
                     { ReLuHook, SigmoidHook },
                     { ReLuGradHook, SigmoidGradHook })
);
```
Example — no activation (linear projection):
```
architecture.push_back(
    FeedForwardLayer(2048, 512, {}, {})
);
```
## Typical Transformer Block
Combining boilerplate layers into a transformer block:
```
std::vector<LayerArgs> architecture;
architecture.push_back(EmbeddingLayer(vocab_size, dim));
architecture.push_back(LearnableLayerNormLayer(dim));
architecture.push_back(AttentionLayer(dim, seq_len));
architecture.push_back(LearnableLayerNormLayer(dim));
architecture.push_back(FeedForwardLayer(dim, dim * 4, ReLuHook, ReLuGradHook));
architecture.push_back(FeedForwardLayer(dim * 4, dim, {}, {}));
architecture.push_back(LearnableLayerNormLayer(dim));
architecture.push_back(FeedForwardLayer(dim, vocab_size, {}, {}));
architecture.push_back(SoftmaxLayer(vocab_size));
```
# Quadratic Neural Network User Guide
Welcome to the quadratic NN project. This library lets you build neural networks where each neuron computes `W_quad * x^2 + W_lin * x + b` instead of just `W * x + b`. It runs on CPU without external ML frameworks.
## Quick Start
Include the necessary headers:
```
#include "model/network.h"
#include "train/train.h"
#include "math/math.h"
```
Before training, define `TRAINING_ON` (or include `train/train.h` which defines it). Compile with `-DTRAINING_ON` if you want to train, or leave it out for inference-only builds.
## The Two Layer Types
**Layer** — the quadratic layer. It stores two weight matrices (quadratic and linear) plus a bias vector. It can also hold forward/backward hooks (activation functions, normalisation, etc). Use this for standard quadratic neurons.
**ParametricLayer** — a simpler wrapper that has a weight matrix and only hooks. It has no built-in matrix multiplies; all computation is done by its hooks (embedding, attention, layer norm). Use it when you need custom weights but not the quadratic multiply.
Both are set up through the same `setupNeuralNetwork()` function using `LayerArgs`.
## Hooks (Activations, Normalisers, etc.)
Hooks are functions that modify the data flowing through a layer. Every hook must have a **forward** (`HookFunc`) and a **backward** (`HookDerivative`) partner.
### Writing a Custom Activation
A forward hook receives:
- `layer` — the layer reference (can be `Layer*` or `ParametricLayer*`)
- `sample_count` — total number of samples (batch × sequence length)
- `original_inputs` — the raw input data
- `preactivation_values` — the values before the activation
- `output_values` — the buffer to fill with the activated output
- `feature_count` — the size of one sample
Example — ReLU forward:
```
HookFunc MyReLU = [](LayerRef layer, int sample_count, float* original_inputs,
                     float* preactivation_values, float* output_values, int feature_count) {
    int total = feature_count * sample_count;
    for (int i = 0; i < total; ++i) {
        output_values[i] = preactivation_values[i] > 0.0f ? preactivation_values[i] : 0.0f;
    }
};
```
The backward hook receives additionally:
- `upstream_gradient` — the gradient from the next layer
- `output_gradient` — where to store the gradient to pass to the previous layer
- `correct_indices` — the correct class/token indices for this batch
Example — ReLU backward:
```
HookDerivative MyReLUGrad = [](LayerRef layer, int sample_count, float* original_inputs,
                                float* preactivation_values, float* upstream_gradient,
                                float* output_gradient, int feature_count,
                                const std::vector<int>& correct_indices) {
    int total = feature_count * sample_count;
    for (int i = 0; i < total; ++i) {
        output_gradient[i] = upstream_gradient[i] *
                             (preactivation_values[i] > 0.0f ? 1.0f : 0.0f);
    }
};
```
**Important:** Every `HookFunc` must have a matching `HookDerivative` that correctly computes the chain rule derivative.
## Pre-built Activations and Layers
`boilerplate/activations.h` contains ready-made hook pairs:
- `ReLuHook` / `ReLuGradHook`
- `SigmoidHook` / `SigmoidGradHook`
- `TanhHook` / `TanhGradHook`
- `Softmax` / `SoftmaxDerivative` — generic softmax, works with any loss
- `Softmax` / `SoftmaxForCrossEntropyLossDerivative` — softmax that computes the full `(softmax - one_hot)` gradient itself, for pairing with the no-op CE loss
- `Residual` / `ResidualGradHook`
- `NonLearnableLayerNorm` / `NonLearnableLayerNormDerivative`
- `LearnableLayerNorm` / `LearnableLayerNormDerivative`
- `RMSNorm` / `RMSNormDerivative`
- `EmbeddingForward` / `EmbeddingDerivative`
- `AttentionForward` / `AttentionDerivative`
`boilerplate/layers.h` gives you shortcut functions that create `LayerArgs`:
- `SoftmaxLayer(model_dimension)` — quadratic layer with softmax hooks
- `LearnableLayerNormLayer(model_dimension)` — layer norm with learnable scale and shift
- `RMSNormLayer(model_dimension)` — RMS normalisation with learnable scale
- `EmbeddingLayer(vocabulary_size, embedding_dimension)` — embedding table
- `AttentionLayer(embedding_dimension, sequence_length)` — single-head self-attention
- `FeedForwardLayer(input_dim, output_dim, activations, activation_derivatives)` — linear layer followed by activations
Example — feed-forward with two activations:
```
LayerArgs hidden_block = FeedForwardLayer(
    512, 2048,
    { ReLuHook, SigmoidHook },
    { ReLuGradHook, SigmoidGradHook }
);
```
## Building a Network
Use `setupNeuralNetwork()` with a vector of `LayerArgs` in order from input to output.
Example for MNIST (784 → 128 quadratic ReLU → 10 classes):
```
std::vector<LayerArgs> architecture;
LayerArgs input_layer;
input_layer.layer_size = 784;
input_layer.kind = Quadratic;
input_layer.outputs_per_neuron = 1;
architecture.push_back(input_layer);
architecture.push_back(
    FeedForwardLayer(784, 128, ReLuHook, ReLuGradHook)
);
LayerArgs output_layer;
output_layer.layer_size = 10;
output_layer.kind = Quadratic;
output_layer.outputs_per_neuron = 1;
architecture.push_back(output_layer);
setupNeuralNetwork(architecture, "UNDEFINED276lineosersyoujelly?");
```
The magic string `"UNDEFINED276lineosersyoujelly?"` initialises weights randomly. Pass a file path to load saved weights instead.
## Loss Functions — The Two Systems
`boilerplate/losses.h` provides two complete systems. Pick one.
### System 1: Generic (use with any output)
Use `CrossEntropyLoss` + `CrossEntropyLossDerivative` as your training loss. Your network must end with a `Softmax` layer.
**How it flows:**
1. Network outputs logits
2. `Softmax` forward hook converts logits to probabilities
3. `CrossEntropyLoss` computes `-log(prob[target])`
4. `CrossEntropyLossDerivative` computes `(prob - one_hot) / batch_size` and sends it upstream
5. `SoftmaxDerivative` receives that upstream, computes the full softmax Jacobian, and passes the gradient back to the logits
Use this when you want a standard softmax that works with any downstream loss.
### System 2: Combined O(n) (for cross-entropy specifically)
Use `CrossEntropyLossForSoftmax` + `CrossEntropyLossForSoftmaxDerivative` as your training loss. Your network must end with a `Softmax` layer whose backward hook is `SoftmaxForCrossEntropyLossDerivative`.
**How it flows:**
1. Network outputs logits
2. `Softmax` forward hook converts logits to probabilities
3. `CrossEntropyLossForSoftmax` computes `-log(prob[target])`
4. `CrossEntropyLossForSoftmaxDerivative` is a **no-op** — it just copies probabilities through unchanged
5. `SoftmaxForCrossEntropyLossDerivative` receives those probabilities, and **on its own** computes `(softmax(logits) - one_hot)` directly from the stored preactivation logits, bypassing the full Jacobian
This is faster (O(n) instead of O(n²)) because the math simplifies: when softmax is followed by cross-entropy, the combined gradient is just `softmax(logits) - one_hot`. The softmax hook does the real work, the loss derivative does nothing.
### Which one should I use?
| System | Softmax backward hook | Loss function | Speed | When |
|---|---|---|---|---|
| Generic | `SoftmaxDerivative` | `CrossEntropyLoss` + `CrossEntropyLossDerivative` | O(n²) | Flexible, works with custom losses |
| Combined | `SoftmaxForCrossEntropyLossDerivative` | `CrossEntropyLossForSoftmax` + `CrossEntropyLossForSoftmaxDerivative` | O(n) | Standard choice, faster |
For regression, use `MeanSquaredErrorLoss` + `MeanSquaredErrorLossDerivative` — no softmax needed.
## Training
Training functions are in `train/train.h`:
- `train(layers, training_data, correct_indices, required_output, learning_rate, loss_function, loss_derivative, step)` — one optimisation step
- `trainScheduler(...)` — full epoch loop with cosine-decay learning rate and Adam moments
Example training call with the combined O(n) system:
```
trainScheduler(layers, training_data, correct_indices, required_output,
               initial_learning_rate, minimum_learning_rate,
               CrossEntropyLossForSoftmax,
               CrossEntropyLossForSoftmaxDerivative,
               total_epochs, batch_size);
```
## Tokenizers
`boilerplate/tokenizers.h` contains:
- `tokenize_letter` — character-level tokenizer
- `tokenize_words` — word-level tokenizer with dynamic vocabulary
- `tokenize_bpe` — Byte-Pair Encoding (skeleton, may need completion)
Example — word tokenization:
```
std::vector<std::string> words = {"the", "hero", "leaves"};
std::vector<int> token_ids;
std::unordered_map<std::string, int> word_to_index;
tokenize_words(words, token_ids, word_to_index);
```
## Full Examples
- `documentation/example_mnist.cpp` — digit classifier on MNIST
- `documentation/example_language_model.cpp` — tiny transformer trained on Hero's Journey text
Compile with:
```
g++ -std=c++17 -DTRAINING_ON example_mnist.cpp model/network.cpp train/train.cpp math/math.cpp -o mnist
```
## Layer vs ParametricLayer in Detail
**Layer:**
- Contains `weights_begin`, `input`, `output`, `size`, `neurons`
- `size` = `neurons + input * output * 2` (quadratic weights + linear weights + biases)
- Methods `quadratic()`, `linear()`, `biases()` return pointers into the weight array
- Forward: computes `matmult(quadratic, x^2) + matmult(linear, x) + bias`, then runs hooks
- Backward: computes gradients of the quadratic form, then runs hook derivatives
**ParametricLayer:**
- Has `weights_begin`, `input`, `output`, but no built-in multiply
- All work is done by hooks; forward copies input to output and runs hooks
- Backward runs hook derivatives; hooks are responsible for updating `weight_gradients`
- Used for embeddings, attention, layer norm, etc.
## Important Tips
- Always pair every forward hook with a backward hook
- Set `batch_size` (from `model/network.h`) before calling `setupNeuralNetwork`
- Training requires `#define TRAINING_ON` or include `train/train.h`
- For `ParametricLayer` hooks, you must update `weight_gradients` inside the hook derivative if the hook has learnable parameters
- All memory buffers are managed globally — no manual allocation needed
- The math library supports cuBLAS and CLBlast via `QQ_BLAS_CUBLAS` / `QQ_BLAS_CLBLAST` defines
Happy hacking!

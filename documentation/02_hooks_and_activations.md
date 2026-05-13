# Hooks and Activations
## What are Hooks?
Hooks are functions that run during the forward and backward passes of a layer. They let you add activations (ReLU, sigmoid, tanh, softmax), normalisation (layer norm, RMS norm), and other transformations to your layers.
Every hook must come as a pair:
- **Forward hook** (`HookFunc`) — transforms data during the forward pass
- **Backward hook** (`HookDerivative`) — computes gradients during backpropagation
## Hook Function Signatures
### Forward Hook (`HookFunc`)
```
void forward_hook(
    LayerRef layer,           // pointer to the layer (Layer* or ParametricLayer*)
    int sample_count,         // total samples = batch_size * sequence_length
    float* original_inputs,   // the raw input data before any processing
    float* preactivation_values, // values before this hook (output of previous step)
    float* output_values,     // buffer to fill with this hook's output
    int feature_count         // number of features per sample
);
```
### Backward Hook (`HookDerivative`)
```
void backward_hook(
    LayerRef layer,           // pointer to the layer
    int sample_count,         // total samples
    float* original_inputs,   // raw inputs (saved from forward pass)
    float* preactivation_values, // preactivation values (saved from forward pass)
    float* upstream_gradient, // gradient coming from the next layer/loss
    float* output_gradient,   // buffer to fill with gradient for previous layer
    int feature_count,        // features per sample
    const std::vector<int>& correct_indices  // correct class/token indices
);
```
## Writing a Custom Activation
### Step 1: Write the forward function
```
HookFunc MyReLU = [](LayerRef layer, int sample_count,
                     float* original_inputs, float* preactivation_values,
                     float* output_values, int feature_count) {
    int total_elements = feature_count * sample_count;
    for (int i = 0; i < total_elements; ++i) {
        output_values[i] = preactivation_values[i] > 0.0f
                           ? preactivation_values[i]
                           : 0.0f;
    }
};
```
### Step 2: Write the backward function (derivative)
The backward function must compute `upstream_gradient * derivative_of_activation`.
For ReLU, the derivative is 1 if the input was > 0, else 0:
```
HookDerivative MyReLUGrad = [](LayerRef layer, int sample_count,
                                float* original_inputs, float* preactivation_values,
                                float* upstream_gradient, float* output_gradient,
                                int feature_count,
                                const std::vector<int>& correct_indices) {
    int total_elements = feature_count * sample_count;
    for (int i = 0; i < total_elements; ++i) {
        output_gradient[i] = upstream_gradient[i] *
                             (preactivation_values[i] > 0.0f ? 1.0f : 0.0f);
    }
};
```
### Step 3: Use your hooks in a layer
```
LayerArgs my_layer;
my_layer.layer_size = 256;
my_layer.kind = Quadratic;
my_layer.outputs_per_neuron = 1;
my_layer.hooks = { MyReLU };
my_layer.hook_gradients = { MyReLUGrad };
```
## Multiple Hooks
A layer can have multiple hooks. They run in order. For example, a residual connection followed by ReLU:
```
LayerArgs block;
block.layer_size = 512;
block.kind = Quadratic;
block.outputs_per_neuron = 1;
block.hooks = { Residual, ReLuHook };
block.hook_gradients = { ResidualGradHook, ReLuGradHook };
```
The forward pass runs: quadratic computation → Residual (adds input) → ReLU
The backward pass runs: ReLU derivative → Residual derivative (passes through)
## Pre-built Activations
`boilerplate/activations.h` provides these ready-to-use pairs:
| Forward Hook | Backward Hook | Description |
|---|---|---|
| `ReLuHook` | `ReLuGradHook` | ReLU: max(0, x) |
| `SigmoidHook` | `SigmoidGradHook` | Sigmoid: 1/(1+e^(-x)) |
| `TanhHook` | `TanhGradHook` | Tanh: tanh(x) |
| `Softmax` | `SoftmaxDerivative` | Softmax (generic, O(n²) backward) |
| `Softmax` | `SoftmaxForCrossEntropyLossDerivative` | Softmax for CE loss (O(n) backward) |
| `Residual` | `ResidualGradHook` | Adds input to output |
| `NonLearnableLayerNorm` | `NonLearnableLayerNormDerivative` | Layer norm without learnable params |
| `LearnableLayerNorm` | `LearnableLayerNormDerivative` | Layer norm with scale and shift |
| `RMSNorm` | `RMSNormDerivative` | RMS normalisation |
| `EmbeddingForward` | `EmbeddingDerivative` | Embedding lookup |
| `AttentionForward` | `AttentionDerivative` | Single-head self-attention |
## Important Rules
1. Every `HookFunc` must have exactly one matching `HookDerivative`
2. The order of hooks and hook_gradients matters — backward hooks run in reverse order
3. For `ParametricLayer` hooks that have learnable weights, update `weight_gradients` inside the backward hook
4. The `original_inputs` and `preactivation_values` in the backward hook are the same values from the forward pass (they are saved automatically)

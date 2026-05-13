# Getting Started
## What is this?
This is a quadratic neural network library. Normal neural networks compute `W * x + b` for each neuron. This library computes `W_quad * x^2 + W_lin * x + b`. The quadratic term lets the network learn curved patterns with fewer neurons.
It runs on CPU with no external ML frameworks. Optional GPU support via cuBLAS or CLBlast.
## Including the Library
```
#include "model/network.h"
#include "train/train.h"
#include "math/math.h"
```
For training, `train/train.h` defines `TRAINING_ON` automatically. Compile with:
```
g++ -std=c++17 -DTRAINING_ON your_file.cpp model/network.cpp train/train.cpp math/math.cpp -o your_program
```
If you only want inference (no training), leave out `train/train.cpp` and don't define `TRAINING_ON`.
## The Two Layer Types
### Layer (Quadratic Layer)
This is the core of the library. Each `Layer` stores:
- `quadratic()` — weights for the `x^2` term
- `linear()` — weights for the `x` term
END2763
- `biases()` — bias values
Forward pass computes:
```
output = quadratic * x^2 + linear * x + bias
```
Then runs any hooks (activations, normalisation) attached to the layer.
Key members:
- `weights_begin` — pointer to start of weight array
- `input` — number of input features
- `output` — number of output features (neurons)
- `size` — total number of weights = `neurons + input * output * 2`
- `neurons` — number of neurons in this layer
- `outputs_per_neuron` — how many outputs each neuron produces (usually 1)
- `forward_hooks` — list of activation functions to apply after the quadratic computation
- `forward_hook_derivatives` — corresponding backward functions for training
### ParametricLayer (Custom Weight Layer)
A simpler wrapper that has weights but no built-in multiply. All computation is done by hooks. Used for:
- Embeddings (lookup table)
- Attention (Q, K, V projections)
- Layer normalisation (scale and shift parameters)
Key members:
- `weights_begin` — pointer to weight array
- `input` — input feature count
- `output` — output feature count
- `weights_per_input` — how many weights per input feature
- `forward_hooks` / `forward_hook_derivatives` — same as Layer
When writing hooks for `ParametricLayer`, you are responsible for accessing `weights_begin` and updating `weight_gradients` in the backward hook if the layer has learnable parameters.
## Building a Network
Use `setupNeuralNetwork()` with a vector of `LayerArgs`. Each `LayerArgs` describes one layer:
```
struct LayerArgs {
    int layer_size;                    // number of neurons
    std::vector<HookFunc> hooks;       // forward activation hooks
    std::vector<HookDerivative> hook_gradients; // backward hooks
    LayerKind kind;                    // Quadratic or Parametric
    int outputs_per_neuron;            // usually 1
    int weights_per_input;             // for Parametric layers
    int scratch_size;                  // temporary buffer size
};
```
Example — a simple two-layer network:
```
std::vector<LayerArgs> architecture;
LayerArgs input_layer;
input_layer.layer_size = 784;
input_layer.kind = Quadratic;
input_layer.outputs_per_neuron = 1;
architecture.push_back(input_layer);
LayerArgs hidden_layer;
hidden_layer.layer_size = 256;
hidden_layer.kind = Quadratic;
hidden_layer.outputs_per_neuron = 1;
hidden_layer.hooks = { ReLuHook };
hidden_layer.hook_gradients = { ReLuGradHook };
architecture.push_back(hidden_layer);
LayerArgs output_layer;
output_layer.layer_size = 10;
output_layer.kind = Quadratic;
output_layer.outputs_per_neuron = 1;
output_layer.hooks = { Softmax };
output_layer.hook_gradients = { SoftmaxDerivative };
architecture.push_back(output_layer);
setupNeuralNetwork(architecture);
```
Calling `setupNeuralNetwork()` with no weight path argument initialises weights using Xavier initialisation by default.
## Weight Initialisation
`setupNeuralNetwork()` accepts an optional weight initialisation function:
```
void setupNeuralNetwork(
    std::vector<LayerArgs> layers,
    std::string weights_path = "",
    WeightInitFunc initialiser = xavier_initialisation
);
```
Built-in initialisers (in `model/network.h`):
- `xavier_initialisation` (default) — good for layers with sigmoid/tanh
- `he_initialisation` — good for layers with ReLU
- `uniform_random_initialisation` — random values between -0.5 and 0.5
- `zero_initialisation` — all weights set to 0
Example with custom initialisation:
```
setupNeuralNetwork(architecture, "", he_initialisation);
```
To load weights from a file:
```
setupNeuralNetwork(architecture, "saved_weights.bin");
```
The file must have exactly `network_size * sizeof(float)` bytes.
## Next Steps
- Read `02_hooks_and_activations.md` to learn how activations work
- Read `03_loss_functions.md` to understand the two loss systems
- Read `04_training.md` to learn how training works
- Read `05_boilerplate.md` for pre-built layers
- Read `06_tokenizers.md` for text processing
- Read `07_examples.md` for full working examples

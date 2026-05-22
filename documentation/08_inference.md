# Inference (Generating Text)
After training a language model, you can load the saved weights and vocabulary to generate new text. The `boilerplate/inference.h` header provides the tools for tokenization, sampling, and generation.
## Saving the model and vocabulary
During training (as shown in `example_language_model.cpp`), save the weights and vocabulary:
```
save_weights("lm_weights.bin");
save_vocabulary(vocabulary, "lm_vocab.txt");
```
## Loading for inference
In your inference program, load the vocabulary and build the same network architecture, then load the weights:
```
std::unordered_map<std::string, int> vocabulary;
load_vocabulary(vocabulary, "lm_vocab.txt");
int vocab_size = vocabulary.size();
int embedding_dim = 128;
int seq_len = 12;
std::vector<LayerArgs> architecture;
architecture.push_back(EmbeddingLayer(vocab_size, embedding_dim));
architecture.push_back(LearnableLayerNormLayer(embedding_dim));
architecture.push_back(AttentionLayer(embedding_dim, seq_len));
architecture.push_back(FeedForwardLayer(embedding_dim, 128, ReLuHook, ReLuGradHook));
architecture.push_back(LearnableLayerNormLayer(128));
LayerArgs output_layer;
output_layer.layer_size = vocab_size;
output_layer.kind = Quadratic;
output_layer.hooks = { TemperatureHook, Softmax };
output_layer.hook_gradients = { TemperatureGradHook, SoftmaxDerivative };
architecture.push_back(output_layer);
setupNeuralNetwork(architecture, "lm_weights.bin");
```
**Important:** Place `TemperatureHook` before `Softmax` in the output layer. The temperature value is passed via `extra_args` of that layer. A value of 0 means no temperature scaling.
## Setting the temperature
Before generating, set the temperature in the output layer's `extra_args`. The temperature is stored as integer thousandths (so 800 = 0.8).
```
layers.back().extra_args = { 800 }; // temperature = 0.8
```
## Generating tokens
Use `generate_tokens` from `boilerplate/inference.h`:
```
std::vector<int> seed_tokens = { tokenize_some_initial_text };
int tokens_to_generate = 50;
int max_seq_len = seq_len;
std::vector<int> generated = generate_tokens(layers, vocabulary, seed_tokens, tokens_to_generate, max_seq_len);
```
`generate_tokens` runs the model autoregressively: it feeds the current sequence, takes the last output probabilities, samples the next token, appends it, and repeats.
## Converting token IDs back to text
Iterate over `generated` and map IDs to strings using the vocabulary:
```
for (int token_id : generated) {
    for (const auto& pair : vocabulary) {
        if (pair.second == token_id) {
            std::cout << pair.first;
            break;
        }
    }
}
```
Note that the vocabulary may not have a perfect reverse map; you can build one or search linearly.
## Complete example (inference program)
```
#include "model/network.h"
#include "boilerplate/activations.h"
#include "boilerplate/layers.h"
#include "boilerplate/inference.h"
#include <iostream>
int main() {
    std::unordered_map<std::string, int> vocab;
    load_vocabulary(vocab, "lm_vocab.txt");
    int vocab_size = vocab.size();
    int dim = 128, seq_len = 12;
    std::vector<LayerArgs> arch;
    arch.push_back(EmbeddingLayer(vocab_size, dim));
    arch.push_back(LearnableLayerNormLayer(dim));
    arch.push_back(AttentionLayer(dim, seq_len));
    arch.push_back(FeedForwardLayer(dim, 128, ReLuHook, ReLuGradHook));
    arch.push_back(LearnableLayerNormLayer(128));
    LayerArgs out;
    out.layer_size = vocab_size;
    out.kind = Quadratic;
    out.hooks = { TemperatureHook, Softmax };
    out.hook_gradients = { TemperatureGradHook, SoftmaxDerivative };
    arch.push_back(out);
    setupNeuralNetwork(arch, "lm_weights.bin");
    layers.back().extra_args = { 800 }; // temperature 0.8
    std::string seed_text = "the hero";
    std::vector<int> seed_ids;
    bpe_tokenize(seed_text, vocab, seed_ids);
    auto generated = generate_tokens(layers, vocab, seed_ids, 30, seq_len);
    std::cout << seed_text;
    for (int id : generated) {
        for (auto& p : vocab) if (p.second == id) { std::cout << p.first; break; }
    }
    std::cout << std::endl;
    return 0;
}
```
Compile with all necessary source files (without -DTRAINING_ON, as inference doesn't need training code). Include `model/network.cpp`, `math/math.cpp`.
## Custom training loops
If you need a training loop not covered by `trainScheduler`, use `train()` directly. See `documentation/04_training.md` for the function signature and step-by-step breakdown.
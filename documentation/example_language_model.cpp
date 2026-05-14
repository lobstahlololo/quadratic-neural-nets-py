#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include "../model/network.h"
#include "../train/train.h"
#include "../math/math.h"
#include "../boilerplate/activations.h"
#include "../boilerplate/layers.h"
#include "../boilerplate/tokenizers.h"
#include "../boilerplate/losses.h"
int main() {
    const std::string story =
        "The hero leaves home. The hero faces trials. The hero meets a mentor. "
        "The hero gains a sword. The hero enters the dark cave. The hero confronts the shadow. "
        "The hero claims the treasure. The hero returns home. The hero shares the treasure. "
        "The hero becomes a legend. ";
    std::vector<std::string> words;
    std::string current_word;
    for (char character : story) {
        if (character == ' ' || character == '.' || character == '?' || character == '!') {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
        } else {
            current_word += character;
        }
    }
    std::unordered_map<std::string, int> word_to_index;
    std::vector<int> token_ids;
    tokenize_words(words, token_ids, word_to_index);
    int vocabulary_size = word_to_index.size() + 2;
    int embedding_dimension = 64;
    int sequence_length = 8;
    std::vector<float> training_data;
    std::vector<float> targets;
    std::vector<std::vector<int>> correct_indices;
    for (size_t position = 0; position + sequence_length < token_ids.size(); ++position) {
        for (int step = 0; step < sequence_length; ++step) {
            training_data.push_back(static_cast<float>(token_ids[position + step]));
        }
        int next_word = token_ids[position + sequence_length];
        std::vector<float> one_hot(vocabulary_size, 0.0f);
        one_hot[next_word] = 1.0f;
        targets.insert(targets.end(), one_hot.begin(), one_hot.end());
        correct_indices.push_back({next_word});
    }
    batch_size = 1;
    std::vector<LayerArgs> architecture;
    architecture.push_back(EmbeddingLayer(vocabulary_size, embedding_dimension));
    architecture.push_back(LearnableLayerNormLayer(embedding_dimension));
    architecture.push_back(AttentionLayer(embedding_dimension, sequence_length));
    architecture.push_back(FeedForwardLayer(embedding_dimension, 128, ReLuHook, ReLuGradHook));
    architecture.push_back(LearnableLayerNormLayer(128));
    LayerArgs output_layer;
    output_layer.layer_size = vocabulary_size;
    output_layer.kind = Quadratic;
    output_layer.outputs_per_neuron = 1;
    output_layer.hooks = { Softmax };
    output_layer.hook_gradients = { SoftmaxForCrossEntropyLossDerivative };
    architecture.push_back(output_layer);
    setupNeuralNetwork(architecture, "", he_initialisation);
    int total_epochs = 50;
    float learning_rate = 0.01f;
    float min_learning_rate = 0.0001f;
    trainScheduler(layers, training_data, correct_indices, targets,
                   learning_rate, min_learning_rate,
                   CrossEntropyLossForSoftmax,
                   CrossEntropyLossForSoftmaxDerivative,
                   total_epochs, batch_size);
    std::cout << "Language model training finished.\n";
    return 0;
}
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
    std::ifstream file("heroes_journey.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open heroes_journey.txt. Please ensure the file exists.\n";
        return 1;
    }
    std::string story((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::vector<std::string> words;
    std::string current_word;
    for (char character : story) {
        if (std::isspace(character) || std::ispunct(character)) {
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
    int embedding_dimension = 128;
    int sequence_length = 12;
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
    batch_size = 16;
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
    int total_epochs = 150;
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
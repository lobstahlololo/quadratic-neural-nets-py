#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "../model/network.h"
#include "../train/train.h"
#include "../math/math.h"
#include "../boilerplate/activations.h"
#include "../boilerplate/layers.h"
#include "../boilerplate/losses.h"
std::string read_text_file(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    return content;
}
std::vector<std::string> get_txt_files(const std::string& path) {
    std::vector<std::string> files;
    files.push_back(path);
    return files;
}
int main() {
    std::string dataset_path = "training_data.txt";
    int maximum_sequence_length = 128;
    int embedding_dimension = 128;
    int feedforward_dimension = 512;
    int transformer_layers = 4;
    int attention_heads = 4;
    int epochs = 100;
    float learning_rate = 0.001f;
    float min_learning_rate = 0.00001f;
    batch_size = 8;
    std::string full_text;
    auto txt_files = get_txt_files(dataset_path);
    for (const auto& filepath : txt_files) {
        full_text += read_text_file(filepath);
    }
    if (full_text.empty()) {
        full_text = "The hero leaves home. The hero faces trials. The hero meets a mentor. "
                    "The hero gains a sword. The hero enters the dark cave. The hero confronts the shadow. "
                    "The hero claims the treasure. The hero returns home. The hero becomes a legend. ";
        std::cout << "No dataset found. Using built-in Hero's Journey text.\n";
    }
    std::unordered_map<char, int> character_to_index;
    std::vector<char> unique_characters;
    for (char character : full_text) {
        if (character_to_index.find(character) == character_to_index.end()) {
            int new_index = character_to_index.size();
            character_to_index[character] = new_index;
            unique_characters.push_back(character);
        }
    }
    int vocabulary_size = character_to_index.size();
    std::vector<int> token_ids;
    for (char character : full_text) {
        token_ids.push_back(character_to_index[character]);
    }
    int stride = maximum_sequence_length / 2;
    std::vector<float> training_data;
    std::vector<float> targets;
    std::vector<std::vector<int>> correct_indices;
    for (size_t position = 0; position + maximum_sequence_length < token_ids.size(); position += stride) {
        for (int step = 0; step < maximum_sequence_length; ++step) {
            training_data.push_back(static_cast<float>(token_ids[position + step]));
        }
        int next_token = token_ids[position + maximum_sequence_length];
        std::vector<float> one_hot(vocabulary_size, 0.0f);
        one_hot[next_token] = 1.0f;
        targets.insert(targets.end(), one_hot.begin(), one_hot.end());
        correct_indices.push_back({next_token});
    }
    std::cout << "Vocabulary size: " << vocabulary_size << "\n";
    std::cout << "Training sequences: " << correct_indices.size() << "\n";
    std::cout << "Characters in dataset: " << full_text.size() << "\n";
    std::vector<LayerArgs> architecture;
    architecture.push_back(EmbeddingLayer(vocabulary_size, embedding_dimension));
    for (int layer = 0; layer < transformer_layers; ++layer) {
        architecture.push_back(LearnableLayerNormLayer(embedding_dimension));
        for (int head = 0; head < attention_heads; ++head) {
            int head_dimension = embedding_dimension / attention_heads;
            architecture.push_back(AttentionLayer(embedding_dimension, maximum_sequence_length));
        }
        architecture.push_back(LearnableLayerNormLayer(embedding_dimension));
        architecture.push_back(FeedForwardLayer(embedding_dimension, feedforward_dimension, ReLuHook, ReLuGradHook));
        architecture.push_back(FeedForwardLayer(feedforward_dimension, embedding_dimension, {}, {}));
    }
    architecture.push_back(LearnableLayerNormLayer(embedding_dimension));
    LayerArgs output_layer;
    output_layer.layer_size = vocabulary_size;
    output_layer.kind = Quadratic;
    output_layer.outputs_per_neuron = 1;
    output_layer.hooks = { Softmax };
    output_layer.hook_gradients = { SoftmaxForCrossEntropyLossDerivative };
    architecture.push_back(output_layer);
    setupNeuralNetwork(architecture, "", xavier_initialisation);
    int network_parameters = 0;
    for (const auto& layer : layers) {
        network_parameters += layer.size;
    }
    std::cout << "Total parameters: " << network_parameters << "\n";
    trainScheduler(layers, training_data, correct_indices, targets,
                   learning_rate, min_learning_rate,
                   CrossEntropyLossForSoftmax,
                   CrossEntropyLossForSoftmaxDerivative,
                   epochs, batch_size);
    std::cout << "Transformer training finished.\n";
    return 0;
}
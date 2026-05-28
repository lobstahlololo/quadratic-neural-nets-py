#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include "../model/network.h"
#include "../train/train.h"
#include "../math/math.h"
#include "../boilerplate/activations.h"
#include "../boilerplate/layers.h"
#include "../boilerplate/tokenizers.h"
#include "../boilerplate/inference.h"
#include "../boilerplate/losses.h"
#include "../boilerplate/weight_inits.h"
#include "../train/train.h"
int main() {
    int target_vocab_size = 256;
    int embedding_dimension = 64;
    int sequence_length = 16;
    int stride = 16;
    int sequences_per_batch = 8;
    int total_epochs = 1000;
    float learning_rate = 0.1f;
    float min_learning_rate = 0.0001f;
    int tokens_to_generate = 100;

    batch_size = sequences_per_batch;

    std::ifstream file("heroes_journey.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open heroes_journey.txt.\n";
        return 1;
    }
    std::string story((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::vector<char> text_chars(story.begin(), story.end());
    std::unordered_map<std::string, int> vocabulary;
    std::cout << "Training BPE vocabulary...\n";
    train_bpe_vocabulary(text_chars, vocabulary, target_vocab_size);
    int vocabulary_size = vocabulary.size();
    std::cout << "Vocabulary size: " << vocabulary_size << "\n";
    std::vector<int> token_ids;
    bpe_tokenize(story, vocabulary, token_ids);
    std::vector<float> training_data;
    std::vector<int> correct_indices;
    std::vector<float> required_output;
    for (size_t pos = 0; pos + sequence_length < token_ids.size(); pos += stride) {
        for (int i = 0; i < sequence_length; ++i) {
            training_data.push_back(static_cast<float>(token_ids[pos + i]));
        }
        for (int i = 0; i < sequence_length; ++i) {
            correct_indices.push_back(token_ids[pos + i + 1]);
        }
    }
    int total_sequences = training_data.size() / sequence_length;
    int total_rows = total_sequences * sequence_length;
    required_output.resize(total_rows * vocabulary_size, 0.0f);
    std::cout << "Training sequences: " << total_sequences << "\n";
    std::cout << "Total prediction rows: " << total_rows << "\n";
    std::vector<LayerArgs> architecture;
    architecture.push_back(InputLayer(1));
    architecture.push_back(EmbeddingLayer(vocabulary_size, embedding_dimension));
    architecture.push_back(LearnableLayerNormLayer(embedding_dimension));
    architecture.push_back(AttentionLayer(embedding_dimension, sequence_length));
    architecture.push_back(FeedForwardLayer(embedding_dimension, embedding_dimension, ReLuHook, ReLuGradHook));
    architecture.push_back(LearnableLayerNormLayer(embedding_dimension));
    LayerArgs output_layer;
    output_layer.layer_size = vocabulary_size;
    output_layer.kind = Quadratic;
    output_layer.hooks = { Softmax };
    output_layer.hook_gradients = { SoftmaxForCrossEntropyLossDerivative };
    architecture.push_back(output_layer);

    setupNeuralNetwork(architecture, "", he_initialisation, sequence_length);

    std::vector<int> sequence_lengths(sequences_per_batch, sequence_length);

    trainScheduler(layers, training_data, correct_indices, required_output,
                   learning_rate, min_learning_rate,
                   CrossEntropyLossForSoftmax,
                   CrossEntropyLossForSoftmaxDerivative,
                   total_epochs, batch_size, sequence_lengths);
    save_weights("transformer_weights.bin");
    save_vocabulary(vocabulary, "transformer_vocab.txt");
    std::cout << "Training finished. Weights and vocabulary saved.\n";
    std::cout << "\nGenerating text...\n";
    std::vector<int> seed_tokens(token_ids.begin(), token_ids.begin() + std::min(sequence_length, (int)token_ids.size()));
    std::vector<int> generated = generate_tokens(layers, vocabulary, seed_tokens, tokens_to_generate, sequence_length);
    std::cout << "Generated sequence: ";
    for (int id : generated) {
        for (const auto& pair : vocabulary) {
            if (pair.second == id) {
                std::cout << pair.first;
                break;
            }
        }
    }
    std::cout << "\n";
    return 0;
}

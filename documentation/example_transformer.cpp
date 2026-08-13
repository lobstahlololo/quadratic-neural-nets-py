#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>

#include "../boilerplate/weight_inits.h"
#include "../model/network.h"
#include "../train/train.h"
#include "../math/math.h"
#include "../boilerplate/activations.h"
#include "../boilerplate/layers.h"
#include "../boilerplate/tokenizers.h"
#include "../boilerplate/inference.h"
#include "../boilerplate/losses.h"

std::string read_text_file(const std::string &filepath)
{
    std::ifstream file(filepath);

    std::string content;
    std::string line;

    while (std::getline(file, line))
    {
        content += line + "\n";
    }

    return content;
}

std::vector<std::string> get_txt_files(const std::string &path)
{
    std::vector<std::string> files;
    files.push_back(path);
    return files;
}

int main()
{
    std::string dataset_path = "training_data.txt";

    int maximum_sequence_length = 128;
    int embedding_dimension = 128;
    int feedforward_dimension = 512;
    int transformer_layers = 4;
    int attention_heads = 4;
    int epochs = 100;

    float learning_rate = 0.001f;
    float min_learning_rate = 0.00001f;

    // Quadratic weights are |x| times more sensitive than linear weights in
    // function space (d y/d W_quad = x^2 vs d y/d W_lin = x), so a slightly
    // reduced LR/weight-decay on the quadratic block equalises the effective
    // step size and improves stability (see train_functions.h). Linear weights,
    // biases, gamma/beta and the embedding keep the default 1.0 scaling.
    quad_lr_scale = 0.7f;
    quad_wd_scale = 0.7f;

    batch_size = 8;

    std::string full_text;

    auto txt_files = get_txt_files(dataset_path);

    for (const auto &filepath : txt_files)
    {
        full_text += read_text_file(filepath);
    }

    if (full_text.empty())
    {
        std::string base_text =
            "The hero leaves home. "
            "The hero faces trials. "
            "The hero meets a mentor. "
            "The hero gains a sword. "
            "The hero enters the dark cave. "
            "The hero confronts the shadow. "
            "The hero claims the treasure. "
            "The hero returns home. "
            "The hero becomes a legend. ";

        for (int i = 0; i < 100; ++i)
        {
            full_text += base_text;
        }

        std::cout
            << "No dataset found. Using built-in Hero's Journey text.\n";
    }

    std::vector<char> text_chars(
        full_text.begin(),
        full_text.end());

    std::unordered_map<std::string, int> vocabulary;

    int target_vocab_size = 128;

    std::cout
        << "Training BPE vocabulary...\n";

    train_bpe_vocabulary(
        text_chars,
        vocabulary,
        target_vocab_size);

    int vocabulary_size = vocabulary.size();

    std::vector<int> token_ids;

    bpe_tokenize(
        full_text,
        vocabulary,
        token_ids);

    std::cout
        << "Token count: "
        << token_ids.size()
        << "\n";

    int stride = maximum_sequence_length / 2;

    std::vector<float> training_data;
    std::vector<float> targets;
    std::vector<int> correct_indices;
    std::vector<int> sequence_lengths;

    std::cout
        << "DEBUG token_ids.size(): "
        << token_ids.size()
        << "\n";

    std::cout
        << "DEBUG maximum_sequence_length: "
        << maximum_sequence_length
        << "\n";

    for (
        size_t position = 0;
        position + maximum_sequence_length < token_ids.size();
        position += stride)
    {
        for (
            int step = 0;
            step < maximum_sequence_length;
            ++step)
        {
            training_data.push_back(
                static_cast<float>(
                    token_ids[position + step]));

            int next_token =
                token_ids[position + step + 1];

            std::vector<float> one_hot(
                vocabulary_size,
                0.0f);

            one_hot[next_token] = 1.0f;

            targets.insert(
                targets.end(),
                one_hot.begin(),
                one_hot.end());

            correct_indices.push_back(
                next_token);
        }

        sequence_lengths.push_back(
            maximum_sequence_length);
    }

    std::cout
        << "Vocabulary size: "
        << vocabulary_size
        << "\n";

    std::cout
        << "Training sequences: "
        << sequence_lengths.size()
        << "\n";

    std::cout
        << "Correct token count: "
        << correct_indices.size()
        << "\n";

    std::cout
        << "Characters in dataset: "
        << full_text.size()
        << "\n";

    std::vector<LayerArgs> architecture;

    architecture.push_back(
        EmbeddingLayer(
            vocabulary_size,
            embedding_dimension));

    for (
        int layer = 0;
        layer < transformer_layers;
        ++layer)
    {
        LayerArgs norm1;

        norm1.layer_size = embedding_dimension;
        norm1.kind = Quadratic;

        norm1.hooks = {
            LearnableLayerNorm(norm1)};

        norm1.hook_gradients = {
            LearnableLayerNormDerivative(norm1)};

        architecture.push_back(norm1);

        for (
            int head = 0;
            head < attention_heads;
            ++head)
        {
            architecture.push_back(
                AttentionLayer(
                    embedding_dimension,
                    maximum_sequence_length));
        }

        LayerArgs norm2;

        norm2.layer_size = embedding_dimension;
        norm2.kind = Quadratic;

        norm2.hooks = {
            LearnableLayerNorm(norm2)};

        norm2.hook_gradients = {
            LearnableLayerNormDerivative(norm2)};

        architecture.push_back(norm2);

        architecture.push_back(
            FeedForwardLayer(
                embedding_dimension,
                feedforward_dimension,
                ReLuHook,
                ReLuGradHook));

        architecture.push_back(
            FeedForwardLayer(
                feedforward_dimension,
                embedding_dimension,
                std::vector<HookFunc>{},
                std::vector<HookDerivative>{}));
    }

    LayerArgs final_norm;

    final_norm.layer_size = embedding_dimension;
    final_norm.kind = Quadratic;

    final_norm.hooks = {
        LearnableLayerNorm(final_norm)};

    final_norm.hook_gradients = {
        LearnableLayerNormDerivative(final_norm)};

    architecture.push_back(final_norm);

    LayerArgs output_layer;

    output_layer.layer_size = vocabulary_size;
    output_layer.kind = Quadratic;

    output_layer.hooks = {
        Softmax};

    output_layer.hook_gradients = {
        SoftmaxForCrossEntropyLossDerivative};

    architecture.push_back(output_layer);

    setupNeuralNetwork(
        architecture,
        "",
        xavier_initialisation,
        // Buffers must hold batch_size x sequence_length rows per training step
        // (total_rows = 8 sequences x 128 tokens = 1024).
        maximum_sequence_length);

    int parameter_count = 0;

    for (const auto &layer : layers)
    {
        std::visit(
            [&](const auto &actual_layer)
            {
                if constexpr (
                    std::is_same_v<
                        std::decay_t<
                            decltype(actual_layer)>,
                        Layer>)
                {
                    parameter_count +=
                        actual_layer.size;

                    parameter_count +=
                        actual_layer.extra_weights_size;
                }
                else
                {
                    parameter_count +=
                        actual_layer.input *
                        actual_layer.weights_per_input;

                    parameter_count +=
                        actual_layer.extra_weights_size;
                }
            },
            layer);
    }

    std::cout
        << "Total parameters: "
        << parameter_count
        << "\n";

    trainScheduler(
        layers,
        training_data,
        correct_indices,
        targets,
        learning_rate,
        min_learning_rate,
        CrossEntropyLossForSoftmax,
        CrossEntropyLossForSoftmaxDerivative,
        epochs,
        batch_size,
        // One batch is batch_size sequences of maximum_sequence_length tokens;
        // trainScheduler slices training_data into batches of these sizes.
        std::vector<int>(batch_size, maximum_sequence_length));

    save_weights(
        "transformer_weights.bin");

    save_vocabulary(
        vocabulary,
        "transformer_vocab.txt");

    std::cout
        << "Transformer training finished. "
        << "Weights saved to transformer_weights.bin, "
        << "vocabulary saved to transformer_vocab.txt\n";

    return 0;
}

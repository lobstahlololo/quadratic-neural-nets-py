// python/robust_harness.cpp
// Stage-9 robustness harness (NOT part of the C++ library; no existing C++ file
// is touched). Builds the SAME architecture as documentation/example_transformer.cpp
// but with configurable dims/seed, and can:
//   mode=fwd   run a forward + backward pass on a VARIABLE-LENGTH batch and dump
//              per-layer outputs/preactivations, gradients, loss, plus a one-step
//              train_adams update (post-step weights + moments);
//   mode=train run the exact trainScheduler loop over a multi-batch corpus
//              (steps_per_epoch > 1) and dump per-step loss/LR/bias-correction
//              powers and weight snapshots.
//
// The corpus is the built-in hero text repeated `repeats` times (deterministic;
// does not depend on training_data.txt). Weights are always freshly generated
// with std::srand(seed) + xavier_initialisation.
//
// This sandbox's std::fstream segfaults, so all file I/O uses C-style fopen.
//
// Usage:
//   robust_harness.exe <outdir> <seed> <vocab> <emb> <ff> <blocks> <heads>
//                      <max_seq> <repeats> <mode> <mode_arg> [batch_size]
//     mode=fwd:    <mode_arg> = comma-separated sequence lengths, e.g. "4,8,4";
//                  batch_size defaults to the number of lengths.
//     mode=train:  <mode_arg> = number of epochs; batch_size defaults to 8.
//
// Build (from repo root):
//   g++ -std=c++17 -O2 -DTRAINING_ON python/robust_harness.cpp \
//       model/network.cpp train/train.cpp math/math.cpp -o python/robust_harness.exe

#define TRAINING_ON
#include "../model/network.h"
#include "../train/train.h"
#include "../math/math.h"
#include "../boilerplate/activations.h"
#include "../boilerplate/layers.h"
#include "../boilerplate/tokenizers.h"
#include "../boilerplate/train_functions.h"
#include "../boilerplate/inference.h"
#include "../boilerplate/losses.h"
#include "../boilerplate/weight_inits.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

static void dbg(const char *msg) { std::cerr << "[harness] " << msg << "\n"; }

static void write_bytes(const char *path, const void *data, size_t nbytes) {
    FILE *f = fopen(path, "wb");
    if (!f) { std::cerr << "[harness] cannot open for write: " << path << "\n"; std::exit(1); }
    if (nbytes > 0 && fwrite(data, 1, nbytes, f) != nbytes) {
        std::cerr << "[harness] short write: " << path << "\n"; std::exit(1);
    }
    fclose(f);
}

static void dump_floats(const std::string &path, const float *data, size_t count) {
    write_bytes(path.c_str(), data, count * sizeof(float));
}
static void dump_ints(const std::string &path, const int *data, size_t count) {
    write_bytes(path.c_str(), data, count * sizeof(int));
}

static std::string pad2(int i) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%02d", i);
    return std::string(buf);
}

// Serialize the flat weight buffer exactly as save_weights() would.
static void save_flat_weights(const std::string &path) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "[harness] cannot open for write: " << path << "\n"; std::exit(1); }
    size_t total = 0;
    for (const auto &l : layers) {
        std::visit([&](auto &layer) {
            size_t main_count = 0;
            if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, Layer>)
                main_count = layer.size;
            else
                main_count = layer.input * layer.weights_per_input;
            fwrite(layer.weights_begin, sizeof(float), main_count, f);
            fwrite(layer.extra_weights_begin, sizeof(float),
                   static_cast<size_t>(layer.extra_weights_size), f);
            total += main_count + static_cast<size_t>(layer.extra_weights_size);
        }, l);
    }
    fclose(f);
    std::cerr << "[harness] serialized weight floats: " << total
              << " (network_size=" << network_size << ")\n";
    if (total != static_cast<size_t>(network_size)) {
        std::cerr << "[harness] FATAL: serialized count != network_size\n";
        std::exit(1);
    }
}

static void dump_layer_gradients(const std::string &outdir) {
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        std::visit([&](auto &layer) {
            size_t main_count = 0;
            if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, Layer>)
                main_count = layer.size;
            else
                main_count = layer.input * layer.weights_per_input;
            char fn[160];
            std::snprintf(fn, sizeof fn, "%sref_layer%02zu_wgrad.bin", outdir.c_str(), layer_index);
            dump_floats(fn, layer.weight_gradients, main_count);
            if (layer.extra_weights_size > 0) {
                std::snprintf(fn, sizeof fn, "%sref_layer%02zu_extra_wgrad.bin", outdir.c_str(), layer_index);
                dump_floats(fn, layer.extra_weight_gradients, layer.extra_weights_size);
            }
        }, layers[layer_index]);
    }
}

int main(int argc, char **argv) {
    if (argc < 11) {
        std::cerr << "usage: robust_harness <outdir> <seed> <vocab> <emb> <ff> <blocks> "
                     "<heads> <max_seq> <repeats> <mode=fwd|train> <mode_arg> [batch_size]\n";
        return 1;
    }
    const std::string outdir = argv[1];
    const int seed = std::atoi(argv[2]);
    const int vocab_target = std::atoi(argv[3]);
    const int embedding_dimension = std::atoi(argv[4]);
    const int feedforward_dimension = std::atoi(argv[5]);
    const int transformer_layers = std::atoi(argv[6]);
    const int attention_heads = std::atoi(argv[7]);
    const int maximum_sequence_length = std::atoi(argv[8]);
    const int repeats = std::atoi(argv[9]);
    const std::string mode = argv[10];
    const int batch_size_arg = (argc > 12) ? std::atoi(argv[12]) : 8;

    // Reference behavior kept for exactness with example_transformer.cpp.
    quad_lr_scale = 0.7f;
    quad_wd_scale = 0.7f;

    // ---- corpus: built-in hero text repeated `repeats` times ----
    std::string full_text;
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
        for (int i = 0; i < repeats; ++i) full_text += base_text;
    }

    std::vector<char> text_chars(full_text.begin(), full_text.end());
    std::unordered_map<std::string, int> vocabulary;
    train_bpe_vocabulary(text_chars, vocabulary, vocab_target);
    const int vocabulary_size = static_cast<int>(vocabulary.size());
    std::cerr << "Vocabulary size: " << vocabulary_size << "\n";

    std::vector<int> token_ids;
    bpe_tokenize(full_text, vocabulary, token_ids);
    std::cerr << "Token count: " << token_ids.size() << "\n";

    const int stride = maximum_sequence_length / 2;
    std::vector<float> training_data;
    std::vector<float> targets;
    std::vector<int> correct_indices;
    std::vector<int> window_lengths;
    for (size_t position = 0; position + maximum_sequence_length < token_ids.size(); position += stride) {
        for (int step = 0; step < maximum_sequence_length; ++step) {
            training_data.push_back(static_cast<float>(token_ids[position + step]));
            const int next_token = token_ids[position + step + 1];
            std::vector<float> one_hot(vocabulary_size, 0.0f);
            one_hot[next_token] = 1.0f;
            targets.insert(targets.end(), one_hot.begin(), one_hot.end());
            correct_indices.push_back(next_token);
        }
        window_lengths.push_back(maximum_sequence_length);
    }
    std::cerr << "Training tokens: " << training_data.size() << " windows: "
              << window_lengths.size() << "\n";

    // ---- architecture (identical structure to example_transformer.cpp) ----
    std::vector<LayerArgs> architecture;
    architecture.push_back(EmbeddingLayer(vocabulary_size, embedding_dimension));
    for (int layer = 0; layer < transformer_layers; ++layer) {
        LayerArgs norm1;
        norm1.layer_size = embedding_dimension;
        norm1.kind = Quadratic;
        norm1.hooks = {LearnableLayerNorm(norm1)};
        norm1.hook_gradients = {LearnableLayerNormDerivative(norm1)};
        architecture.push_back(norm1);
        for (int head = 0; head < attention_heads; ++head) {
            architecture.push_back(AttentionLayer(embedding_dimension, maximum_sequence_length));
        }
        LayerArgs norm2;
        norm2.layer_size = embedding_dimension;
        norm2.kind = Quadratic;
        norm2.hooks = {LearnableLayerNorm(norm2)};
        norm2.hook_gradients = {LearnableLayerNormDerivative(norm2)};
        architecture.push_back(norm2);
        architecture.push_back(FeedForwardLayer(embedding_dimension, feedforward_dimension,
                                                ReLuHook, ReLuGradHook));
        architecture.push_back(FeedForwardLayer(feedforward_dimension, embedding_dimension,
                                                std::vector<HookFunc>{},
                                                std::vector<HookDerivative>{}));
    }
    LayerArgs final_norm;
    final_norm.layer_size = embedding_dimension;
    final_norm.kind = Quadratic;
    final_norm.hooks = {LearnableLayerNorm(final_norm)};
    final_norm.hook_gradients = {LearnableLayerNormDerivative(final_norm)};
    architecture.push_back(final_norm);
    LayerArgs output_layer;
    output_layer.layer_size = vocabulary_size;
    output_layer.kind = Quadratic;
    output_layer.hooks = {Softmax};
    output_layer.hook_gradients = {SoftmaxForCrossEntropyLossDerivative};
    architecture.push_back(output_layer);

    // ---- batch size / buffers ----
    std::vector<int> batch_seq_lengths;
    int epochs = 0;
    if (mode == "fwd") {
        std::string csv = argv[11];
        size_t pos = 0;
        while (pos < csv.size()) {
            size_t comma = csv.find(',', pos);
            batch_seq_lengths.push_back(std::atoi(csv.substr(pos, comma - pos).c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        batch_size = static_cast<int>(batch_seq_lengths.size());
    } else {
        epochs = std::atoi(argv[11]);
        batch_size = batch_size_arg;
        batch_seq_lengths.assign(batch_size, maximum_sequence_length);
    }
    int buffer_multiplier = 0;
    for (int s : batch_seq_lengths) buffer_multiplier = std::max(buffer_multiplier, s);
    std::cerr << "batch_size=" << batch_size << " buffer_multiplier=" << buffer_multiplier << "\n";

    std::srand(static_cast<unsigned>(seed));
    setupNeuralNetwork(architecture, "", xavier_initialisation, buffer_multiplier);
    std::cerr << "network_size (total params): " << network_size << "\n";

    // ---- dump fresh weights + full corpus token data ----
    save_flat_weights(outdir + "fresh_transformer_weights.bin");
    dump_floats(outdir + "training_data.bin", training_data.data(), training_data.size());
    dump_ints(outdir + "correct_indices.bin", correct_indices.data(), correct_indices.size());
    dump_floats(outdir + "targets.bin", targets.data(), targets.size());
    dump_ints(outdir + "window_lengths.bin", window_lengths.data(), window_lengths.size());

    int total_rows = 0;
    for (int s : batch_seq_lengths) total_rows += s;
    std::cerr << "mode=" << mode << " total_rows=" << total_rows << "\n";

    if (mode == "fwd") {
        // Batch = first sum(lengths) rows of the corpus (compacted, var-length).
        std::vector<float> batch_inputs(training_data.begin(), training_data.begin() + total_rows);
        std::vector<float> batch_targets(targets.begin(),
                                         targets.begin() + total_rows * vocabulary_size);
        std::vector<int> batch_correct(correct_indices.begin(), correct_indices.begin() + total_rows);
        dump_floats(outdir + "batch_inputs.bin", batch_inputs.data(), batch_inputs.size());
        dump_floats(outdir + "batch_targets.bin", batch_targets.data(), batch_targets.size());
        dump_ints(outdir + "batch_correct.bin", batch_correct.data(), batch_correct.size());
        dump_ints(outdir + "batch_seq_lengths.bin", batch_seq_lengths.data(), batch_seq_lengths.size());

        // forward + per-layer output/preact dumps
        float *input_pointer = batch_inputs.data();
        FILE *manifest = fopen((outdir + "manifest.txt").c_str(), "w");
        fprintf(manifest, "total_rows %d\n", total_rows);
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            if (auto *layer_pointer = std::get_if<Layer>(&layers[layer_index])) {
                input_pointer = layer_pointer->forward(input_pointer, batch_size, batch_seq_lengths);
            } else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers[layer_index])) {
                input_pointer = parametric_layer_pointer->forward(input_pointer, batch_size, batch_seq_lengths);
            }
            std::visit([&](auto &actual_layer) {
                const int out = static_cast<int>(actual_layer.output);
                const int count = out * total_rows;
                char fn[160];
                std::snprintf(fn, sizeof fn, "%sref_layer%02zu_output.bin", outdir.c_str(), layer_index);
                dump_floats(fn, actual_layer.output_pointer, count);
                const bool has_hooks = !actual_layer.forward_hooks.empty();
                std::snprintf(fn, sizeof fn, "%sref_layer%02zu_preact.bin", outdir.c_str(), layer_index);
                if (has_hooks) dump_floats(fn, actual_layer.previous_preactivations, count);
                fprintf(manifest, "%zu output=%d preact=%d\n", layer_index, count, has_hooks ? count : 0);
            }, layers[layer_index]);
        }
        fclose(manifest);

        // loss (C++ reporting: mean over rows of -log(p_target + 1e-7))
        int last_layer_output = 0;
        std::visit([&](auto &current_layer) { last_layer_output = static_cast<int>(current_layer.output); },
                   layers.back());
        float batch_loss = 0.0f;
        for (int row_index = 0; row_index < total_rows; ++row_index) {
            std::vector<int> single_target = {batch_correct[row_index]};
            batch_loss += CrossEntropyLossForSoftmax(
                input_pointer + last_layer_output * row_index,
                batch_targets.data() + last_layer_output * row_index,
                single_target, last_layer_output);
        }
        batch_loss /= total_rows;
        std::cerr << "batch_loss " << batch_loss << "\n";
        float one = batch_loss;
        dump_floats(outdir + "batch_loss.bin", &one, 1);

        // backward + gradient dumps
        {
            std::vector<float> downstream_gradient(static_cast<size_t>(last_layer_output) * total_rows);
            for (int row_index = 0; row_index < total_rows; ++row_index) {
                std::vector<int> single_target = {batch_correct[row_index]};
                CrossEntropyLossForSoftmaxDerivative(
                    0.0f,
                    input_pointer + last_layer_output * row_index,
                    batch_targets.data() + last_layer_output * row_index,
                    single_target,
                    downstream_gradient.data() + last_layer_output * row_index,
                    last_layer_output);
            }
            std::vector<int> mutable_lengths = batch_seq_lengths;
            float *gradient_pointer = downstream_gradient.data();
            for (int layer_index = static_cast<int>(layers.size()) - 1; layer_index >= 0; --layer_index) {
                if (auto *layer_pointer = std::get_if<Layer>(&layers[layer_index])) {
                    gradient_pointer = layer_pointer->backward(gradient_pointer, batch_size,
                                                               mutable_lengths, batch_correct);
                } else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers[layer_index])) {
                    gradient_pointer = parametric_layer_pointer->backward(gradient_pointer, batch_size,
                                                                          mutable_lengths, batch_correct);
                }
                int input_count = 0;
                std::visit([&](auto &l) { input_count = static_cast<int>(l.input) * total_rows; },
                           layers[layer_index]);
                dump_floats(outdir + "ref_layer" + pad2(layer_index) + "_gradin.bin",
                            gradient_pointer, input_count);
            }
            dump_layer_gradients(outdir);
        }
        std::cerr << "[harness] fwd/back dumps done\n";

        // one-step optimizer update (train_adams, step 0, moments zeroed)
        {
            first_moment_buffer.resize(network_size);
            second_moment_buffer.resize(network_size);
            std::fill(first_moment_buffer.begin(), first_moment_buffer.end(), 0.0f);
            std::fill(second_moment_buffer.begin(), second_moment_buffer.end(), 0.0f);
            const float one_step_loss = train_adams(
                layers, batch_inputs, batch_correct, batch_targets, 0.001f,
                CrossEntropyLossForSoftmax, CrossEntropyLossForSoftmaxDerivative,
                0, batch_size, batch_seq_lengths);
            std::cerr << "one_step_loss " << one_step_loss << "\n";
            save_flat_weights(outdir + "post_step_weights.bin");
            dump_floats(outdir + "post_step_first_moments.bin",
                        first_moment_buffer.data(), first_moment_buffer.size());
            dump_floats(outdir + "post_step_second_moments.bin",
                        second_moment_buffer.data(), second_moment_buffer.size());
            const float d1 = std::pow(0.9f, 1.0f);
            const float d2 = std::pow(0.999f, 1.0f);
            float dp[2] = {d1, d2};
            dump_floats(outdir + "decay_powers.bin", dp, 2);
        }
        std::cerr << "[harness] one-step dumps done\n";
    } else {
        // mode=train: exact trainScheduler loop over the multi-batch corpus.
        const int total_input_rows = batch_size * maximum_sequence_length;
        const int steps_per_epoch = static_cast<int>(training_data.size()) / total_input_rows;
        const int num_steps = epochs * steps_per_epoch;
        std::cerr << "steps_per_epoch=" << steps_per_epoch << " num_steps=" << num_steps << "\n";
        first_moment_buffer.resize(network_size);
        second_moment_buffer.resize(network_size);
        std::fill(first_moment_buffer.begin(), first_moment_buffer.end(), 0.0f);
        std::fill(second_moment_buffer.begin(), second_moment_buffer.end(), 0.0f);

        FILE *lossfile = fopen((outdir + "multistep_losses.txt").c_str(), "w");
        FILE *lrfile = fopen((outdir + "multistep_lrs.txt").c_str(), "w");
        FILE *dpfile = fopen((outdir + "multistep_decay_powers.bin").c_str(), "wb");
        const float learning_rate = 0.001f;
        const float minimum_learning_rate = 0.00001f;
        const int output_size = vocabulary_size;
        int global_step = 0;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int j = 0; j < steps_per_epoch; ++j, ++global_step) {
                float current_learning_rate = minimum_learning_rate +
                    (learning_rate - minimum_learning_rate) *
                    (1 + std::cos(3.14159265f * j / steps_per_epoch)) / 2;
                std::vector<float> batch_inputs(training_data.begin() + j * total_input_rows,
                                                training_data.begin() + (j + 1) * total_input_rows);
                std::vector<float> batch_targets(targets.begin() + j * total_input_rows * output_size,
                                                 targets.begin() + (j + 1) * total_input_rows * output_size);
                std::vector<int> batch_correct(correct_indices.begin() + j * total_input_rows,
                                               correct_indices.begin() + (j + 1) * total_input_rows);
                std::vector<int> mutable_lengths = batch_seq_lengths;
                float batch_loss = train_adams(
                    layers, batch_inputs, batch_correct, batch_targets, current_learning_rate,
                    CrossEntropyLossForSoftmax, CrossEntropyLossForSoftmaxDerivative,
                    global_step, batch_size, mutable_lengths);
                fprintf(lossfile, "%d %.9g\n", global_step + 1, batch_loss);
                fprintf(lrfile, "%d %.9g\n", global_step + 1, current_learning_rate);
                const float d1 = std::pow(0.9f, static_cast<float>(global_step + 1));
                const float d2 = std::pow(0.999f, static_cast<float>(global_step + 1));
                float dp[2] = {d1, d2};
                fwrite(dp, sizeof(float), 2, dpfile);
                std::cerr << "[harness] step " << (global_step + 1) << "/" << num_steps
                          << " epoch " << (epoch + 1) << " j " << j
                          << " lr " << current_learning_rate << " loss " << batch_loss << "\n";
                const int after = global_step + 1;
                if (after == 1 || after == 2 || after == 5 || after == 10 ||
                    after == 20 || after == 50 || after == num_steps) {
                    save_flat_weights(outdir + "post_step_" + pad2(after) + "_weights.bin");
                }
            }
        }
        fclose(lossfile);
        fclose(lrfile);
        fclose(dpfile);
        dump_floats(outdir + "final_first_moments.bin",
                    first_moment_buffer.data(), first_moment_buffer.size());
        dump_floats(outdir + "final_second_moments.bin",
                    second_moment_buffer.data(), second_moment_buffer.size());
        std::cerr << "[harness] train loop done\n";
    }

    {
        FILE *meta = fopen((outdir + "meta.txt").c_str(), "w");
        fprintf(meta, "seed %d\n", seed);
        fprintf(meta, "vocab_size %d\n", vocabulary_size);
        fprintf(meta, "emb %d\n", embedding_dimension);
        fprintf(meta, "ff %d\n", feedforward_dimension);
        fprintf(meta, "blocks %d\n", transformer_layers);
        fprintf(meta, "heads %d\n", attention_heads);
        fprintf(meta, "max_seq %d\n", maximum_sequence_length);
        fprintf(meta, "batch_size %d\n", batch_size);
        fprintf(meta, "total_tokens %zu\n", training_data.size());
        fprintf(meta, "num_windows %zu\n", window_lengths.size());
        fprintf(meta, "total_rows %d\n", total_rows);
        fprintf(meta, "expected_params %d\n", network_size);
        fprintf(meta, "mode %s\n", mode.c_str());
        fprintf(meta, "epochs %d\n", epochs);
        fprintf(meta, "steps_per_epoch %d\n", mode == "train"
            ? static_cast<int>(training_data.size()) / (batch_size * maximum_sequence_length) : 0);
        fclose(meta);
    }

    dbg("done");
    return 0;
}

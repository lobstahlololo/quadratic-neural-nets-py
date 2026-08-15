// python/export_reference.cpp
// Comparison helper (NOT part of the C++ library; no existing C++ file is
// touched). It replicates the exact data pipeline + architecture of
// documentation/example_transformer.cpp, saves a FRESH (untrained) weight file,
// and dumps the first training batch plus every forward intermediate so that
// python/compare_forward.py can verify the PyTorch replica against the C++
// reference. Training semantics are not modified.
//
// NOTE: this sandbox's std::fstream runtime segfaults, so all file I/O uses
// C-style fopen/fwrite. The flat weight buffer is serialized by concatenating
// each layer's weight block + extra-weight block in architecture order, which is
// byte-identical to what save_weights() would produce from the flat `weights`
// vector (verified by asserting the sum equals network_size).
//
// Build (from repo root):
//   g++ -std=c++17 -O2 -DTRAINING_ON python/export_reference.cpp \
//       model/network.cpp train/train.cpp math/math.cpp -o python/export_reference.exe

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

static void dbg(const char *msg) { std::cerr << "[ref] " << msg << "\n"; }

static void write_bytes(const char *path, const void *data, size_t nbytes) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        std::cerr << "[ref] cannot open for write: " << path << "\n";
        std::exit(1);
    }
    if (nbytes > 0 && fwrite(data, 1, nbytes, f) != nbytes) {
        std::cerr << "[ref] short write: " << path << "\n";
        std::exit(1);
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
    char buf[8];
    std::snprintf(buf, sizeof buf, "%02d", i);
    return std::string(buf);
}

static std::string read_text_file(const std::string &filepath) {
    std::string content;
    FILE *f = fopen(filepath.c_str(), "rb");
    if (!f) return content;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) content.append(buf, n);
    fclose(f);
    return content;
}

static std::vector<std::string> get_txt_files(const std::string &path) {
    std::vector<std::string> files;
    files.push_back(path);
    return files;
}

// Serialize the flat weight buffer exactly as save_weights() would: each layer's
// main weight block followed by its extra-weight block, in architecture order.
// Single file handle (reopening per block would truncate).
static void save_flat_weights(const std::string &path) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "[ref] cannot open for write: " << path << "\n";
        std::exit(1);
    }
    size_t total = 0;
    for (const auto &l : layers) {
        std::visit(
            [&](auto &layer) {
                size_t main_count = 0;
                if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, Layer>)
                    main_count = layer.size;
                else
                    main_count = layer.input * layer.weights_per_input;
                fwrite(layer.weights_begin, sizeof(float), main_count, f);
                fwrite(layer.extra_weights_begin, sizeof(float),
                       static_cast<size_t>(layer.extra_weights_size), f);
                total += main_count + static_cast<size_t>(layer.extra_weights_size);
            },
            l);
    }
    fclose(f);
    std::cerr << "[ref] serialized weight floats: " << total << " (network_size=" << network_size
              << ")\n";
    if (total != static_cast<size_t>(network_size)) {
        std::cerr << "[ref] FATAL: serialized count != network_size\n";
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    dbg("main start");
    const bool quick = (argc > 2) && std::string(argv[2]) == "quick";
    const std::string outdir = "python/ref_data/";
    // Optional seed argv[2] (when not the "quick" flag): fresh initialization
    // for the second-seed robustness check. Without it, behaviour is unchanged
    // (default unseeded state).
    const bool seeded = argc > 2 && std::string(argv[2]) != "quick";
    if (seeded) std::srand(static_cast<unsigned>(std::atoi(argv[2])));
    const int maximum_sequence_length = 128;
    const int embedding_dimension = 128;
    const int feedforward_dimension = 512;
    const int transformer_layers = 4;
    const int attention_heads = 4;

    // Reference behavior: quadratic optimizer scaling (only affects training,
    // kept for exactness with example_transformer.cpp).
    quad_lr_scale = 0.7f;
    quad_wd_scale = 0.7f;
    batch_size = 8;

    // ---- data pipeline (identical to example_transformer.cpp) ----
    std::string full_text;
    auto txt_files = get_txt_files("training_data.txt");
    for (const auto &filepath : txt_files) {
        full_text += read_text_file(filepath);
    }
    if (full_text.empty()) {
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
        for (int i = 0; i < 100; ++i) {
            full_text += base_text;
        }
        std::cerr << "No dataset found. Using built-in Hero's Journey text.\n";
    }

    std::vector<char> text_chars(full_text.begin(), full_text.end());
    std::unordered_map<std::string, int> vocabulary;
    train_bpe_vocabulary(text_chars, vocabulary, 128);
    const int vocabulary_size = static_cast<int>(vocabulary.size());
    std::cerr << "Vocabulary size: " << vocabulary_size << "\n";

    std::vector<int> token_ids;
    bpe_tokenize(full_text, vocabulary, token_ids);
    std::cerr << "Token count: " << token_ids.size() << "\n";

    const int stride = maximum_sequence_length / 2;
    std::vector<float> training_data;
    std::vector<float> targets;
    std::vector<int> correct_indices;
    std::vector<int> sequence_lengths;
    for (size_t position = 0; position + maximum_sequence_length < token_ids.size(); position += stride) {
        for (int step = 0; step < maximum_sequence_length; ++step) {
            training_data.push_back(static_cast<float>(token_ids[position + step]));
            const int next_token = token_ids[position + step + 1];
            std::vector<float> one_hot(vocabulary_size, 0.0f);
            one_hot[next_token] = 1.0f;
            targets.insert(targets.end(), one_hot.begin(), one_hot.end());
            correct_indices.push_back(next_token);
        }
        sequence_lengths.push_back(maximum_sequence_length);
    }
    std::cerr << "Training tokens: " << training_data.size() << "\n";

    // ---- architecture (identical to example_transformer.cpp) ----
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
        architecture.push_back(
            FeedForwardLayer(embedding_dimension, feedforward_dimension, ReLuHook, ReLuGradHook));
        architecture.push_back(
            FeedForwardLayer(feedforward_dimension, embedding_dimension,
                             std::vector<HookFunc>{}, std::vector<HookDerivative>{}));
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

    setupNeuralNetwork(architecture, "", xavier_initialisation, maximum_sequence_length);
    std::cerr << "network_size (total params): " << network_size << "\n";

    // ---- fresh weights (serialized manually; save_weights uses broken fstream here) ----
    save_flat_weights(outdir + "fresh_transformer_weights.bin");
    dbg("saved fresh weights");

    // ---- dump token data ----
    dump_floats(outdir + "training_data.bin", training_data.data(), training_data.size());
    dump_ints(outdir + "correct_indices.bin", correct_indices.data(), correct_indices.size());
    dump_floats(outdir + "targets.bin", targets.data(), targets.size());
    dump_ints(outdir + "sequence_lengths.bin", sequence_lengths.data(), sequence_lengths.size());

    // ---- forward pass on batch 0 + per-layer dumps ----
    const int total_rows = batch_size * maximum_sequence_length;
    std::vector<int> seq_lengths(batch_size, maximum_sequence_length);

    float *input_pointer = training_data.data();
    std::string manifest_path = outdir + "manifest.txt";
    FILE *manifest = quick ? nullptr : fopen(manifest_path.c_str(), "w");
    if (!quick && !manifest) {
        std::cerr << "[ref] cannot open manifest\n";
        return 1;
    }
    if (manifest) fprintf(manifest, "total_rows %d\n", total_rows);
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        if (auto *layer_pointer = std::get_if<Layer>(&layers[layer_index])) {
            input_pointer = layer_pointer->forward(input_pointer, batch_size, seq_lengths);
        } else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers[layer_index])) {
            input_pointer = parametric_layer_pointer->forward(input_pointer, batch_size, seq_lengths);
        }

        std::visit(
            [&](auto &actual_layer) {
                if (!quick) {
                    const int out = static_cast<int>(actual_layer.output);
                    const int count = out * total_rows;
                    char fn[160];
                    std::snprintf(fn, sizeof fn, "%sref_layer%02zu_output.bin", outdir.c_str(), layer_index);
                    dump_floats(fn, actual_layer.output_pointer, count);
                    // 'preact' is the pre-hook value; valid only for layers that HAVE
                    // hooks (FF2 layers have none and never fill previous_preactivations).
                    const bool has_hooks = !actual_layer.forward_hooks.empty();
                    std::snprintf(fn, sizeof fn, "%sref_layer%02zu_preact.bin", outdir.c_str(), layer_index);
                    if (has_hooks) {
                        dump_floats(fn, actual_layer.previous_preactivations, count);
                    }
                    fprintf(manifest, "%zu output=%d preact=%d\n", layer_index, count,
                            has_hooks ? count : 0);
                }
            },
            layers[layer_index]);

    }
    if (manifest) fclose(manifest);
    dbg("forward + dumps done");

    // ---- batch-0 loss using the exact C++ loss function ----
    int last_layer_output = 0;
    std::visit([&](auto &current_layer) { last_layer_output = static_cast<int>(current_layer.output); },
               layers.back());
    float batch_loss = 0.0f;
    for (int row_index = 0; row_index < total_rows; ++row_index) {
        std::vector<int> single_target = {correct_indices[row_index]};
        batch_loss += CrossEntropyLossForSoftmax(
            input_pointer + last_layer_output * row_index,
            targets.data() + last_layer_output * row_index,
            single_target, last_layer_output);
    }
    batch_loss /= total_rows;
    std::cerr << "batch0_loss " << batch_loss << "\n";

    // ---- backward pass + gradient dumps (mirrors train_adams) ----
    if (!quick)
    {
        std::vector<float> downstream_gradient(static_cast<size_t>(last_layer_output) * total_rows);
        for (int row_index = 0; row_index < total_rows; ++row_index) {
            std::vector<int> single_target = {correct_indices[row_index]};
            CrossEntropyLossForSoftmaxDerivative(
                0.0f,  // loss value is unused by this derivative
                input_pointer + last_layer_output * row_index,
                targets.data() + last_layer_output * row_index,
                single_target,
                downstream_gradient.data() + last_layer_output * row_index,
                last_layer_output);
        }
        std::vector<int> mutable_sequence_lengths = seq_lengths;
        float *gradient_pointer = downstream_gradient.data();
        for (int layer_index = static_cast<int>(layers.size()) - 1; layer_index >= 0; --layer_index) {
            if (auto *layer_pointer = std::get_if<Layer>(&layers[layer_index])) {
                gradient_pointer = layer_pointer->backward(
                    gradient_pointer, batch_size, mutable_sequence_lengths, correct_indices);
                dump_floats(outdir + "ref_layer" + pad2(layer_index) + "_wgrad.bin",
                            layer_pointer->weight_gradients, layer_pointer->size);
                if (layer_pointer->extra_weights_size > 0)
                    dump_floats(outdir + "ref_layer" + pad2(layer_index) + "_extra_wgrad.bin",
                                layer_pointer->extra_weight_gradients,
                                layer_pointer->extra_weights_size);
            } else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers[layer_index])) {
                gradient_pointer = parametric_layer_pointer->backward(
                    gradient_pointer, batch_size, mutable_sequence_lengths, correct_indices);
                const size_t wcount =
                    parametric_layer_pointer->input * parametric_layer_pointer->weights_per_input;
                dump_floats(outdir + "ref_layer" + pad2(layer_index) + "_wgrad.bin",
                            parametric_layer_pointer->weight_gradients, wcount);
                if (parametric_layer_pointer->extra_weights_size > 0)
                    dump_floats(outdir + "ref_layer" + pad2(layer_index) + "_extra_wgrad.bin",
                                parametric_layer_pointer->extra_weight_gradients,
                                parametric_layer_pointer->extra_weights_size);
            }
            // gradient wrt THIS layer's input (= previous layer's output)
            int input_count = 0;
            std::visit([&](auto &l) { input_count = static_cast<int>(l.input) * total_rows; },
                       layers[layer_index]);
            dump_floats(outdir + "ref_layer" + pad2(layer_index) + "_gradin.bin",
                        gradient_pointer, input_count);
        }
    }
    dbg("backward + gradient dumps done");

    // ---- per-sequence attention intermediates ----
    // The forward hook reuses its scratch buffer per sequence, so after the
    // batch forward only the LAST sequence's Q/K/V/scores survive. Re-run each
    // attention layer one sequence at a time (identical matmuls -> identical
    // values) to capture every sequence. Process in REVERSE layer order so that
    // re-running layer i (which overwrites layer i's own output region = layer
    // i+1's input) never corrupts a layer whose intermediates are still needed.
    if (!quick)
    {
        std::vector<size_t> attention_indices;
        for (size_t i = 0; i < layers.size(); ++i) {
            auto *plp = std::get_if<ParametricLayer>(&layers[i]);
            if (plp && !plp->extra_args.empty() && plp->scratch_pointer && i != 0)
                attention_indices.push_back(i);
        }
        std::vector<int> single_lengths(1, maximum_sequence_length);
        for (auto rit = attention_indices.rbegin(); rit != attention_indices.rend(); ++rit) {
            const size_t layer_index = *rit;
            auto *plp = std::get_if<ParametricLayer>(&layers[layer_index]);
            const int emb = static_cast<int>(plp->output);
            const int max_seq = plp->extra_args[0];
            const int n = max_seq * emb;
            float *prev_out = nullptr;
            std::visit([&](auto &l) { prev_out = l.output_pointer; }, layers[layer_index - 1]);
            for (int s = 0; s < batch_size; ++s) {
                plp->forward(prev_out + s * n, 1, single_lengths);
                char fn[160];
                std::snprintf(fn, sizeof fn, "%sref_layer%02zu_q_seq%02d.bin", outdir.c_str(), layer_index, s);
                dump_floats(fn, plp->scratch_pointer, n);
                std::snprintf(fn, sizeof fn, "%sref_layer%02zu_k_seq%02d.bin", outdir.c_str(), layer_index, s);
                dump_floats(fn, plp->scratch_pointer + n, n);
                std::snprintf(fn, sizeof fn, "%sref_layer%02zu_v_seq%02d.bin", outdir.c_str(), layer_index, s);
                dump_floats(fn, plp->scratch_pointer + 2 * n, n);
                std::snprintf(fn, sizeof fn, "%sref_layer%02zu_scores_seq%02d.bin", outdir.c_str(), layer_index, s);
                dump_floats(fn, plp->scratch_pointer + 3 * n, max_seq * max_seq);
            }
        }
    }
    dbg("per-sequence attention dumps done");

    // ---- multi-step training loop (exact trainScheduler semantics) ----
    // With this corpus steps_per_epoch = training_data.size() / total_input_rows
    // = 1024 / 1024 = 1, so every epoch is ONE step on the SAME 1024-token batch
    // and the cosine LR at j=0 equals the max LR (0.001f) for every step.
    // num_steps = argv[1] (default 20) -> the controlled Stage-8 run.
    {
        const int num_steps = (argc > 1) ? std::atoi(argv[1]) : 20;
        first_moment_buffer.resize(network_size);
        second_moment_buffer.resize(network_size);
        std::fill(first_moment_buffer.begin(), first_moment_buffer.end(), 0.0f);
        std::fill(second_moment_buffer.begin(), second_moment_buffer.end(), 0.0f);

        std::vector<int> batch_seq_lengths(batch_size, maximum_sequence_length);
        std::vector<float> batch_inputs(training_data.begin(),
                                        training_data.begin() + total_rows);
        std::vector<float> batch_targets(targets.begin(),
                                         targets.begin() + total_rows * vocabulary_size);
        std::vector<int> batch_correct(correct_indices.begin(),
                                       correct_indices.begin() + total_rows);

        FILE *lossfile = fopen((outdir + "multistep_losses.txt").c_str(), "w");
        FILE *dpfile = fopen((outdir + "multistep_decay_powers.bin").c_str(), "wb");
        std::cerr << "[ref] training for " << num_steps << " steps\n";
        for (int t = 0; t < num_steps; ++t) {
            // trainScheduler: j=0, steps_per_epoch=1 -> cosine schedule = max LR.
            const float step_lr = 0.001f;
            const float step_loss = train_adams(
                layers, batch_inputs, batch_correct, batch_targets, step_lr,
                CrossEntropyLossForSoftmax, CrossEntropyLossForSoftmaxDerivative,
                t /* training_step */, batch_size, batch_seq_lengths);
            fprintf(lossfile, "%d %.9g\n", t + 1, step_loss);
            const float d1 = std::pow(0.9f, static_cast<float>(t + 1));
            const float d2 = std::pow(0.999f, static_cast<float>(t + 1));
            float dp[2] = {d1, d2};
            fwrite(dp, sizeof(float), 2, dpfile);
            std::cerr << "[ref] step " << (t + 1) << " loss " << step_loss
                      << " decay_powers " << d1 << " " << d2 << "\n";

            const int after = t + 1;
            if (after == 1 || after == 2 || after == 5 || after == 10 ||
                after == 20 || after == 50 || after == num_steps) {
                save_flat_weights(outdir + "post_step_" + pad2(after) + "_weights.bin");
            }
            if (after == 1) {
                // Legacy one-step dumps for optimizer_step.py (step-0 results).
                {
                    FILE *f = fopen((outdir + "post_step_wgrads.bin").c_str(), "wb");
                    size_t total = 0;
                    for (const auto &l : layers) {
                        std::visit([&](auto &layer) {
                            size_t main_count = 0;
                            if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, Layer>)
                                main_count = layer.size;
                            else
                                main_count = layer.input * layer.weights_per_input;
                            fwrite(layer.weight_gradients, sizeof(float), main_count, f);
                            fwrite(layer.extra_weight_gradients, sizeof(float),
                                   static_cast<size_t>(layer.extra_weights_size), f);
                            total += main_count + static_cast<size_t>(layer.extra_weights_size);
                        }, l);
                    }
                    fclose(f);
                    std::cerr << "[ref] post_step_wgrads floats: " << total << "\n";
                }
                save_flat_weights(outdir + "post_step_weights.bin");
                dump_floats(outdir + "post_step_first_moments.bin",
                            first_moment_buffer.data(), first_moment_buffer.size());
                dump_floats(outdir + "post_step_second_moments.bin",
                            second_moment_buffer.data(), second_moment_buffer.size());
                float dp0[2] = {d1, d2};
                dump_floats(outdir + "decay_powers.bin", dp0, 2);
                std::cerr << "decay_powers " << d1 << " " << d2 << "\n";
            }
            if (after == num_steps) {
                dump_floats(outdir + "final_first_moments.bin",
                            first_moment_buffer.data(), first_moment_buffer.size());
                dump_floats(outdir + "final_second_moments.bin",
                            second_moment_buffer.data(), second_moment_buffer.size());
            }
        }
        fclose(lossfile);
        fclose(dpfile);
    }
    dbg("multi-step training dumps done");

    {
        FILE *meta = fopen((outdir + "meta.txt").c_str(), "w");
        fprintf(meta, "vocab_size %d\n", vocabulary_size);
        fprintf(meta, "total_tokens %zu\n", training_data.size());
        fprintf(meta, "num_windows %zu\n", sequence_lengths.size());
        fprintf(meta, "max_seq %d\n", maximum_sequence_length);
        fprintf(meta, "emb %d\n", embedding_dimension);
        fprintf(meta, "ff %d\n", feedforward_dimension);
        fprintf(meta, "transformer_layers %d\n", transformer_layers);
        fprintf(meta, "attention_heads %d\n", attention_heads);
        fprintf(meta, "batch_size %d\n", batch_size);
        fprintf(meta, "total_rows %d\n", total_rows);
        fprintf(meta, "expected_params %d\n", network_size);
        fprintf(meta, "batch0_loss %.9g\n", batch_loss);
        fclose(meta);
    }

    dbg("done");
    return 0;
}

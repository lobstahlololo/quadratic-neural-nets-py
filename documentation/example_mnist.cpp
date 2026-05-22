#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "../model/network.h"
#include "../train/train.h"
#include "../math/math.h"
#include "../boilerplate/activations.h"
#include "../boilerplate/layers.h"
#include "../boilerplate/losses.h"
std::vector<float> read_mnist_images(const std::string& filepath, int& image_count, int& rows, int& cols) {
    std::ifstream file(filepath, std::ios::binary);
    int magic, count, r, c;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&count), 4);
    image_count = __builtin_bswap32(count);
    file.read(reinterpret_cast<char*>(&r), 4);
    rows = __builtin_bswap32(r);
    file.read(reinterpret_cast<char*>(&c), 4);
    cols = __builtin_bswap32(c);
    std::vector<float> images(image_count * rows * cols);
    for (int i = 0; i < image_count * rows * cols; ++i) {
        unsigned char pixel = 0;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        images[i] = pixel / 255.0f;
    }
    return images;
}
std::vector<float> read_mnist_labels(const std::string& filepath, int& label_count) {
    std::ifstream file(filepath, std::ios::binary);
    int magic, count;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&count), 4);
    label_count = __builtin_bswap32(count);
    std::vector<float> labels(label_count * 10, 0.0f);
    for (int i = 0; i < label_count; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i * 10 + label] = 1.0f;
    }
    return labels;
}
int main() {
    batch_size = 64;
    int image_count, rows, cols, label_count;
    auto images = read_mnist_images("train-images-idx3-ubyte", image_count, rows, cols);
    auto labels = read_mnist_labels("train-labels-idx1-ubyte", label_count);
    std::vector<int> correct_indices(label_count);
    for (int i = 0; i < label_count; ++i) {
        for (int j = 0; j < 10; ++j) {
            if (labels[i * 10 + j] == 1.0f) {
                correct_indices[i] = j;
                break;
            }
        }
    }
    int input_size = rows * cols;
    int hidden_size = 256;
    int output_size = 10;
    std::vector<LayerArgs> architecture;
    LayerArgs input_layer;
    input_layer.layer_size = input_size;
    input_layer.kind = Quadratic;
    architecture.push_back(input_layer);
    architecture.push_back(FeedForwardLayer(input_size, hidden_size, ReLuHook, ReLuGradHook));
    LayerArgs output_layer;
    output_layer.layer_size = output_size;
    output_layer.kind = Quadratic;
    output_layer.hooks = { Softmax };
    output_layer.hook_gradients = { SoftmaxForCrossEntropyLossDerivative };
    architecture.push_back(output_layer);
    setupNeuralNetwork(architecture, "", he_initialisation);
    int total_epochs = 20;
    float learning_rate = 0.01f;
    float min_learning_rate = 0.0001f;
    trainScheduler(layers, images, correct_indices, labels,
                   learning_rate, min_learning_rate,
                   CrossEntropyLossForSoftmax,
                   CrossEntropyLossForSoftmaxDerivative,
                   total_epochs, batch_size);
    save_weights("mnist_model.bin");
    std::cout << "MNIST training finished. Weights saved to mnist_model.bin\n";
    return 0;
}
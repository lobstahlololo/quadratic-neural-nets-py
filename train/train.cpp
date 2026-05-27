#include "train.h"
#include "../model/network.h"
#include <iostream>
#include <cmath>
#include <variant>
#include <iostream>
using std::vector;
vector<float> first_moment_buffer;
vector<float> second_moment_buffer;
float train(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& batch_sizes) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	float* allocated_input_pointer = output_buffers.first.data(); 
	for (int i = 0; i < layers.size(); ++i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			allocated_input_pointer = layer_ptr->forward(allocated_input_pointer, batch_count, batch_sizes);
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			allocated_input_pointer = param_ptr->forward(allocated_input_pointer, batch_count, batch_sizes);
		}
	}
    int last_layer_output = 0;
    std::visit([&](auto& l) { last_layer_output = l.output; }, layers.back());
    vector<float> sample_losses(total_rows);
    for (int i = 0; i < total_rows; ++i) {
        std::vector<int> single_target = { correct_indices[i] };
        sample_losses[i] = loss_function(output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, last_layer_output);
    }
    float batch_loss = 0.0f;
    for (int i = 0; i < total_rows; ++i) batch_loss += sample_losses[i];
    batch_loss /= total_rows;
    vector<float> downstream_gradient(last_layer_output * total_rows);
    for (int i = 0; i < total_rows; ++i) {
        std::vector<int> single_target = { correct_indices[i] };
        loss_derivative(sample_losses[i], output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, downstream_gradient.data() + last_layer_output * i, last_layer_output);
    }
	std::vector<int> mutable_batch_sizes = batch_sizes;
	float* grad_ptr = downstream_gradient.data();
	for (int i = layers.size() - 1; i >= 0; --i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			grad_ptr = layer_ptr->backward(grad_ptr, batch_count, mutable_batch_sizes, correct_indices);
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			grad_ptr = param_ptr->backward(grad_ptr, batch_count, mutable_batch_sizes, correct_indices);
		}
	}
    float scaled_learning_rate = learning_rate / batch_size;
    float first_moment_decay_power = pow(0.9f, step + 1);
    float second_moment_decay_power = pow(0.999f, step + 1);
    size_t moment_offset = 0;
    for (int i = 0; i < layers.size(); ++i) {
        if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
            for (int j = 0; j < layer_ptr->size; ++j) {
                float grad = layer_ptr->weight_gradients[j];
                first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
                second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
                float first_bias_correction = first_moment_buffer[moment_offset + j] / (1 - first_moment_decay_power);
                float second_bias_correction = second_moment_buffer[moment_offset + j] / (1 - second_moment_decay_power);
                layer_ptr->weights_begin[j] -= scaled_learning_rate * first_bias_correction / (sqrt(second_bias_correction) + 1e-8f);
            }
            moment_offset += layer_ptr->size;
        } else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
            int weight_count = param_ptr->input * param_ptr->weights_per_input;
            for (int j = 0; j < weight_count; ++j) {
                float grad = param_ptr->weight_gradients[j];
                first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
                second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
                float first_bias_correction = first_moment_buffer[moment_offset + j] / (1 - first_moment_decay_power);
                float second_bias_correction = second_moment_buffer[moment_offset + j] / (1 - second_moment_decay_power);
                param_ptr->weights_begin[j] -= scaled_learning_rate * first_bias_correction / (sqrt(second_bias_correction) + 1e-8f);
            }
            moment_offset += weight_count;
        }
    }
	return batch_loss;
}
void trainScheduler(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, std::vector<float> required_output, float learning_rate, float minimum_learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int total_epochs, int batch_size_arg, std::vector<int> initial_batch_sizes) {
	first_moment_buffer.resize(network_size);
	second_moment_buffer.resize(network_size);
	batch_size = batch_size_arg;
	if (initial_batch_sizes.empty()) {
		initial_batch_sizes.resize(batch_size, 1);
	}
	std::cout << "batch size: " << batch_size << "\n";
	std::cout << "initial batch size size:" << initial_batch_sizes.size() << "\n";
	
        int input_size = 0;
        std::visit([&](auto& l) { input_size = l.input; }, layers[0]);
        int output_size = 0;
        std::visit([&](auto& l) { output_size = l.output; }, layers.back());
	int total_input_rows = 0;
	for (int s : initial_batch_sizes) {
		total_input_rows += s;
		std::cout << "Batch size: " << s << "\n";
	}
	
	int steps_per_epoch = training_data.size() / (total_input_rows * input_size);
	for (int epoch = 0; epoch < total_epochs; ++epoch) {
		float epoch_loss = 0.0f;
		for (int j = 0; j < steps_per_epoch; ++j) {
			float current_learning_rate = minimum_learning_rate + (learning_rate - minimum_learning_rate) * (1 + cos(3.14159265f * j / steps_per_epoch)) / 2;
			std::vector<float> batch_inputs(training_data.begin() + j * total_input_rows * input_size, training_data.begin() + (j + 1) * total_input_rows * input_size);
			std::vector<float> batch_targets(required_output.begin() + j * total_input_rows * output_size, required_output.begin() + (j + 1) * total_input_rows * output_size);
			std::vector<int> batch_correct_indices(correct_indices.begin() + j * total_input_rows, correct_indices.begin() + (j + 1) * total_input_rows);
			int step = epoch * steps_per_epoch + j;
			std::vector<int> mutable_batch_sizes = initial_batch_sizes;
			float batch_loss = train(layers, batch_inputs, batch_correct_indices, batch_targets, current_learning_rate, loss_function, loss_derivative, step, batch_size, mutable_batch_sizes);
			epoch_loss += batch_loss;
		}
		epoch_loss /= steps_per_epoch;
		std::cout << "Epoch " << epoch + 1 << "/" << total_epochs << " - Loss: " << epoch_loss << "\n";
	}
}

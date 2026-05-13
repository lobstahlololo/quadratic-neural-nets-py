#include "train.h"
#include "../model/network.h"
#include <cmath>
using std::vector;
vector<float> first_moment_buffer;
vector<float> second_moment_buffer;
void train(std::vector<Layer>& layers, const std::vector<float>& training_data, const std::vector<std::vector<int>>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step) {
	float* allocated_input_pointer = output_buffers.first.data(); 
	for (int i = 0; i < layers.size(); ++i) {
		allocated_input_pointer = layers[i].forward(allocated_input_pointer);
	}
	vector<float> sample_losses(batch_size);
	for (int i = 0; i < batch_size; ++i) {
		sample_losses[i] = loss_function(output_buffers.first.data() + layers.back().output * i, required_output.data() + layers.back().output * i, correct_indices[i], layers.back().output);
	}
	vector<float> downstream_gradient(layers.back().output * batch_size);
	for (int i = 0; i < batch_size; ++i) {
		loss_derivative(sample_losses[i], output_buffers.first.data() + layers.back().output * i, required_output.data() + layers.back().output * i, correct_indices[i], downstream_gradient.data() + layers.back().output * i, layers.back().output);
	}
	for (int i = 0; i < layers.size(); ++i) {
		std::fill(layers[i].weight_gradients, layers[i].weight_gradients + layers[i].size, 0.0f);
	}
	float* downstream_pointer = downstream_gradient.data();
	for (int b = 0; b < batch_size; ++b) {
		int temporary_batch_size = correct_indices[b].size();
		float* downstream_sample_pointer = downstream_pointer + b * layers.back().output;
		const std::vector<int>& sample_correct_indices = correct_indices[b];
		float* sample_pointer = downstream_sample_pointer;
		for (int i = layers.size() - 1; i >= 0; --i) {
			sample_pointer = layers[i].backward(sample_pointer, sample_correct_indices, temporary_batch_size);
		}
	}
	float scaled_learning_rate = learning_rate / batch_size;
	float first_moment_decay_power = pow(0.9f, step + 1);
	float second_moment_decay_power = pow(0.999f, step + 1);
	size_t moment_offset = 0;
	for (int i = 0; i < layers.size(); ++i) {
		const Layer& layer = layers[i];
		for (int j = 0; j < layer.size; ++j) {
			float grad = layer.weight_gradients[j];
			first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
			second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
			float first_bias_correction = first_moment_buffer[moment_offset + j] / (1 - first_moment_decay_power);
			float second_bias_correction = second_moment_buffer[moment_offset + j] / (1 - second_moment_decay_power);
			layer.weights_begin[j] -= scaled_learning_rate * first_bias_correction / (sqrt(second_bias_correction) + 1e-8f);
		}
		moment_offset += layer.size;
	}
}
void trainScheduler(std::vector<Layer>& layers, const std::vector<float>& training_data, const std::vector<std::vector<int>>& correct_indices, std::vector<float> required_output, float learning_rate, float minimum_learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int total_epochs, int batch_size_arg) {
	first_moment_buffer.resize(network_size);
	second_moment_buffer.resize(network_size);
	batch_size = batch_size_arg;
	int input_size = layers[0].input;
	int output_size = layers.back().output;
	int steps_per_epoch = training_data.size() / (batch_size * input_size);
	for (int epoch = 0; epoch < total_epochs; ++epoch) {
		for (int j = 0; j < steps_per_epoch; ++j) {
			float current_learning_rate = minimum_learning_rate + (learning_rate - minimum_learning_rate) * (1 + cos(3.14159265f * j / steps_per_epoch)) / 2;
			std::vector<float> batch_inputs(training_data.begin() + j * batch_size * input_size, training_data.begin() + (j + 1) * batch_size * input_size);
			std::vector<float> batch_targets(required_output.begin() + j * batch_size * output_size, required_output.begin() + (j + 1) * batch_size * output_size);
			std::vector<std::vector<int>> batch_correct_indices(correct_indices.begin() + j * batch_size, correct_indices.begin() + (j + 1) * batch_size);
			int step = epoch * steps_per_epoch + j;
			train(layers, batch_inputs, batch_correct_indices, batch_targets, current_learning_rate, loss_function, loss_derivative, step);
		}
	}
}

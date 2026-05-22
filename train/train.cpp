#include "train.h"
#include "../model/network.h"
#include <cmath>
using std::vector;
vector<float> first_moment_buffer;
vector<float> second_moment_buffer;
void train(std::vector<Layer>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& batch_sizes) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	float* allocated_input_pointer = output_buffers.first.data(); 
	for (int i = 0; i < layers.size(); ++i) {
		allocated_input_pointer = layers[i].forward(allocated_input_pointer, batch_count, batch_sizes);
	}
	vector<float> sample_losses(total_rows);
	for (int i = 0; i < total_rows; ++i) {
		std::vector<int> single_target = { correct_indices[i] };
		sample_losses[i] = loss_function(output_buffers.first.data() + layers.back().output * i, required_output.data() + layers.back().output * i, single_target, layers.back().output);
	}
	vector<float> downstream_gradient(layers.back().output * total_rows);
	for (int i = 0; i < total_rows; ++i) {
		std::vector<int> single_target = { correct_indices[i] };
		loss_derivative(sample_losses[i], output_buffers.first.data() + layers.back().output * i, required_output.data() + layers.back().output * i, single_target, downstream_gradient.data() + layers.back().output * i, layers.back().output);
	}
	std::vector<int> mutable_batch_sizes = batch_sizes;
	float* grad_ptr = downstream_gradient.data();
	for (int i = layers.size() - 1; i >= 0; --i) {
		grad_ptr = layers[i].backward(grad_ptr, batch_count, mutable_batch_sizes, correct_indices);
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
void trainScheduler(std::vector<Layer>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, std::vector<float> required_output, float learning_rate, float minimum_learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int total_epochs, int batch_size_arg, std::vector<int> initial_batch_sizes) {
	first_moment_buffer.resize(network_size);
	second_moment_buffer.resize(network_size);
	batch_size = batch_size_arg;
	if (initial_batch_sizes.empty()) {
		initial_batch_sizes.resize(batch_size, 1);
	}
	int input_size = layers[0].input;
	int output_size = layers.back().output;
	int total_input_rows = 0;
	for (int s : initial_batch_sizes) total_input_rows += s;
	int steps_per_epoch = training_data.size() / (total_input_rows * input_size);
	for (int epoch = 0; epoch < total_epochs; ++epoch) {
		for (int j = 0; j < steps_per_epoch; ++j) {
			float current_learning_rate = minimum_learning_rate + (learning_rate - minimum_learning_rate) * (1 + cos(3.14159265f * j / steps_per_epoch)) / 2;
			std::vector<float> batch_inputs(training_data.begin() + j * total_input_rows * input_size, training_data.begin() + (j + 1) * total_input_rows * input_size);
			std::vector<float> batch_targets(required_output.begin() + j * total_input_rows * output_size, required_output.begin() + (j + 1) * total_input_rows * output_size);
			std::vector<int> batch_correct_indices(correct_indices.begin() + j * total_input_rows, correct_indices.begin() + (j + 1) * total_input_rows);
			int step = epoch * steps_per_epoch + j;
			std::vector<int> mutable_batch_sizes = initial_batch_sizes;
			train(layers, batch_inputs, batch_correct_indices, batch_targets, current_learning_rate, loss_function, loss_derivative, step, batch_size, mutable_batch_sizes);
		}
	}
}
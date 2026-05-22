#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include "network.h"
#include "../math/math.h"
bool is_setup = false;
int batch_size = 1;
int network_size = 0;
std::pair<std::vector<float>, std::vector<float>> output_buffers;
std::vector<float> squared_inputs_buffer;
std::vector<float> weights;
std::vector<float> global_scratch_buffer;
std::vector<Layer> layers;
std::vector<int> layer_sizes;
#ifdef TRAINING_ON
std::vector<float> global_inputs;
std::vector<float> global_preactivations;
std::vector<float> global_grads;
#endif
float* Layer::forward(float* inputs, int batch_count, std::vector<int>& batch_sizes) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	if (size == 0) {
		#ifdef TRAINING_ON
		previous_inputs = inputs;
		previous_preactivations = inputs;
		#endif
		if (forward_hooks.empty()) {
			return inputs;
		}
		LayerRef self(this);
		std::copy(inputs, inputs + output * total_rows, output_buffers.second.data());
		for (auto& hook : forward_hooks) {
			hook(self, batch_count, batch_sizes, inputs, output_buffers.second.data(), output_buffers.second.data(), output);
		}
		std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_buffers.first.data());
		if (output_pointer) {
			std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_pointer);
		}
		std::fill(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, 0.0f);
		return output_buffers.first.data();
	}
	float* in_ptr = inputs;
	for (std::size_t i = 0; i < input * total_rows; ++i) {
		squared_inputs_buffer[i] = in_ptr[i] * in_ptr[i];
	}
	matmult(linear(), inputs, output_buffers.second.data(), output, input, total_rows);
	matmult(quadratic(), squared_inputs_buffer.data(), output_buffers.second.data(), output, input, total_rows);
	for (std::size_t i = 0; i < output * total_rows; ++i) {
		output_buffers.second[i] += biases()[i % output];
	}
	if (!forward_hooks.empty()) {
		LayerRef self(this);
		std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, previous_preactivations);
		for (auto& hook : forward_hooks) {
			hook(self, batch_count, batch_sizes, inputs, output_buffers.second.data(), output_buffers.second.data(), output);
		}
	}
	#ifdef TRAINING_ON
	std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_pointer);
	#endif
	std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_buffers.first.data());
	std::fill(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, 0.0f);
	return output_buffers.first.data();
}
float* ParametricLayer::forward(float* inputs, int batch_count, std::vector<int>& batch_sizes) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	if (forward_hooks.empty()) {
		return inputs;
	}
	LayerRef self(this);
	std::copy(inputs, inputs + input * total_rows, output_buffers.first.data());
	for (auto& hook : forward_hooks) {
		hook(self, batch_count, batch_sizes, inputs, output_buffers.first.data(), output_buffers.first.data(), input);
	}
	if (output_pointer) {
		std::copy(output_buffers.first.data(), output_buffers.first.data() + input * total_rows, output_pointer);
	}
	return output_buffers.first.data();
}
#ifdef TRAINING_ON
float* Layer::backward(float* upstream_gradient, int batch_count, std::vector<int>& batch_sizes, const std::vector<int>& correct_indices) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	if (!forward_hook_derivatives.empty()) {
		LayerRef self(this);
		for (auto& derivHook : forward_hook_derivatives) {
			derivHook(self, batch_count, batch_sizes, previous_inputs, previous_preactivations, upstream_gradient, output_buffers.first.data(), output, correct_indices);
		}
	}
	size_t total_weights = weight_count();
	size_t bias_offset = total_weights;
	std::fill(weight_gradients + bias_offset, weight_gradients + size, 0.0f);
	for (size_t b = 0; b < total_rows; ++b) {
		for (size_t n = 0; n < neurons; ++n) {
			weight_gradients[bias_offset + n] += upstream_gradient[b * neurons + n];
		}
	}
	float* post_activation_gradient = output_buffers.first.data();
	float* scratch_data = output_buffers.second.data();
	matmult(post_activation_gradient, previous_inputs, weight_gradients, output, total_rows, input);
	matmult(post_activation_gradient, squared_inputs_buffer.data(), weight_gradients + input * output, output, total_rows, input);
	matmult(post_activation_gradient, linear(), scratch_data, total_rows, output, input);
	for (size_t i = 0; i < input * total_rows; ++i) {
		output_buffers.first.data()[i] = scratch_data[i];
		scratch_data[i] = 0.0f;
	}
	matmult(post_activation_gradient, quadratic(), scratch_data, total_rows, output, input);
	for (size_t i = 0; i < input * total_rows; ++i) {
		output_buffers.first.data()[i] += 2.0f * previous_inputs[i] * scratch_data[i];
	}
	return output_buffers.first.data();
}
float* ParametricLayer::backward(float* upstream_gradient, int batch_count, std::vector<int>& batch_sizes, const std::vector<int>& correct_indices) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	size_t weight_count = input * weights_per_input;
	std::fill(weight_gradients, weight_gradients + weight_count, 0.0f);
	if (!forward_hook_derivatives.empty()) {
		LayerRef self(this);
		for (auto& derivHook : forward_hook_derivatives) {
			derivHook(self, batch_count, batch_sizes, previous_inputs, upstream_gradient, output_buffers.first.data(), input, correct_indices);
		}
	}
	return output_buffers.first.data();
}
#endif

void setupNeuralNetwork(std::vector<LayerArgs> layersAdd, std::string weights_path, WeightInitFunc initialiser, int max_sequence_length) {
	layers.clear();
	weights.clear();
	layer_sizes.clear();
	int maximum_neurons = 0;
	network_size = 0;
	int max_token_count = batch_size * max_sequence_length;
	int maximum_buffer_size = 0;
	LayerArgs* prev = nullptr;
	for (auto& args : layersAdd) {
		maximum_neurons = std::max(maximum_neurons, args.layer_size);
		int token_capacity = args.layer_size * max_token_count;
		if (token_capacity > maximum_buffer_size) maximum_buffer_size = token_capacity;
		if (prev == nullptr) {
			layer_sizes.push_back(args.layer_size);
			prev = &args;
			continue;
		}
		if (args.kind == Parametric) {
			network_size += prev->layer_size * args.weights_per_input;
		} else {
			network_size += args.layer_size + args.layer_size * prev->layer_size * 2;
		}
		layer_sizes.push_back(args.layer_size);
		prev = &args;
	}
	auto resizeBuffers = [&](size_t size) {
		output_buffers.first.resize(size);
		output_buffers.second.resize(size);
	};
	resizeBuffers(maximum_buffer_size);
	size_t maximum_squared_input_size = 0;
	for (size_t i = 0; i < layersAdd.size(); ++i) {
		if (i > 0 && layersAdd[i].kind != Parametric) {
			size_t input_size = layersAdd[i-1].layer_size;
			size_t needed = input_size * max_token_count;
			if (needed > maximum_squared_input_size) maximum_squared_input_size = needed;
		}
	}
	squared_inputs_buffer.resize(maximum_squared_input_size);
	size_t total_scratch = 0;
	for (auto& args : layersAdd) total_scratch += args.scratch_size;
	global_scratch_buffer.resize(total_scratch);
	#ifdef TRAINING_ON
	global_inputs.resize(maximum_neurons * max_token_count);
	global_preactivations.resize(maximum_neurons * max_token_count);
	global_grads.resize(network_size);
	#endif
	weights.resize(network_size);

	if (weights_path.empty()) {
		initialiser(weights.data(), network_size, layersAdd);
	} else {
		std::ifstream file(weights_path, std::ios::binary);
		if (!file) {
			throw std::runtime_error("wrong filepath");
		}
		file.seekg(0, std::ios::end);
		std::streampos fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		if (static_cast<long>(network_size * sizeof(float)) != fileSize) {
			throw std::runtime_error("wrong file size relation to weights");
		}
		file.read(reinterpret_cast<char*>(weights.data()), network_size * sizeof(float));
	}

	float* weight_start = weights.data();
	std::size_t previous_output_offset = 0;
	std::size_t accumulated_weights = 0;
	std::size_t gradient_offset = 0;
	std::size_t scratch_offset = 0;
	for (std::size_t i = 0; i < layersAdd.size(); ++i) {
		Layer layer;
		for (auto& h : layersAdd[i].hooks) {
			layer.forward_hooks.push_back(h);
		}
		for (auto& hd : layersAdd[i].hook_gradients) {
			layer.forward_hook_derivatives.push_back(hd);
		}
		layer.neurons = layersAdd[i].layer_size;
		if (i == 0) {
			layer.input = layersAdd[i].layer_size;
			layer.output = layersAdd[i].layer_size;
			layer.size = 0;
			layer.weights_begin = weight_start;
		} else {
			layer.input = layersAdd[i-1].layer_size;
			layer.output = layersAdd[i].layer_size;
			if (layersAdd[i].kind == Parametric) {
				layer.size = layer.input * layersAdd[i].weights_per_input;
				layer.weights_begin = weight_start + accumulated_weights;
			} else {
				layer.size = layersAdd[i].layer_size + layersAdd[i].layer_size * layersAdd[i-1].layer_size * 2;
				layer.weights_begin = weight_start + accumulated_weights;
			}
		}
		size_t output_data_size = layer.output * max_token_count;
		layer.extra_args = layersAdd[i].extra_args;
		layer.scratch_size = layersAdd[i].scratch_size;
		layer.scratch_pointer = layer.scratch_size > 0 ? global_scratch_buffer.data() + scratch_offset : nullptr;
		scratch_offset += layer.scratch_size;
		#ifdef TRAINING_ON
		layer.previous_preactivations = global_preactivations.data() + previous_output_offset;
		layer.weight_gradients = global_grads.data() + gradient_offset;
		layer.output_pointer = global_inputs.data() + previous_output_offset;
		if (i > 0) {
			layer.previous_inputs = layers.back().output_pointer;
		} else {
			layer.previous_inputs = nullptr;
		}
		#endif
		layers.push_back(layer);
		previous_output_offset += output_data_size;
		if (i > 0) {
			gradient_offset += layer.size;
			accumulated_weights += layer.size;
		}
	}
	is_setup = true;
}

void save_weights(std::string path) {
	
	std::ofstream file(path, std::ios::binary);
	if (!file) {
		throw std::runtime_error("wrong filepath");
	}
	file.write(reinterpret_cast<const char*>(weights), network_size * sizeof(float));
	

}

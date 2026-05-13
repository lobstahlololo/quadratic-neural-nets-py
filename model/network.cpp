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
float* Layer::forward(float* inputs, int temporary_batch_size) {
	int effective_batch_size = batch * temporary_batch_size;
	if (size == 0) {
		#ifdef TRAINING_ON
		previous_inputs = inputs;
		previous_preactivations = inputs;
		#endif
		if (forward_hooks.empty()) {
			return inputs;
		}
		LayerRef self(this);
		std::copy(inputs, inputs + output * effective_batch_size, output_buffers.second.data());
		for (auto& hook : forward_hooks) {
			hook(self, effective_batch_size, inputs, output_buffers.second.data(), output_buffers.second.data(), output);
		}
		std::copy(output_buffers.second.data(), output_buffers.second.data() + output * effective_batch_size, output_buffers.first.data());
		if (output_pointer) {
			std::copy(output_buffers.second.data(), output_buffers.second.data() + output * effective_batch_size, output_pointer);
		}
		std::fill(output_buffers.second.data(), output_buffers.second.data() + output * effective_batch_size, 0.0f);
		return output_buffers.first.data();
	}
	float* in_ptr = inputs;
	for (std::size_t i = 0; i < input * effective_batch_size; ++i) {
		squared_inputs_buffer[i] = in_ptr[i] * in_ptr[i];
	}
	matmult(linear(), inputs, output_buffers.second.data(), output, input, effective_batch_size);
	matmult(quadratic(), squared_inputs_buffer.data(), output_buffers.second.data(), output, input, effective_batch_size);
	for (std::size_t i = 0; i < output * effective_batch_size; ++i) {
		output_buffers.second[i] += biases()[i % output];
	}
	if (!forward_hooks.empty()) {
		LayerRef self(this);
		std::copy(output_buffers.second.data(), output_buffers.second.data() + output * effective_batch_size, previous_preactivations);
		for (auto& hook : forward_hooks) {
			hook(self, effective_batch_size, inputs, output_buffers.second.data(), output_buffers.second.data(), output);
		}
	}
	#ifdef TRAINING_ON
	std::copy(output_buffers.second.data(), output_buffers.second.data() + output * effective_batch_size, output_pointer);
	#endif
	std::copy(output_buffers.second.data(), output_buffers.second.data() + output * effective_batch_size, output_buffers.first.data());
	std::fill(output_buffers.second.data(), output_buffers.second.data() + output * effective_batch_size, 0.0f);
	return output_buffers.first.data();
}
float* ParametricLayer::forward(float* inputs, int temporary_batch_size) {
	int effective_batch_size = batch * temporary_batch_size;
	if (forward_hooks.empty()) {
		return inputs;
	}
	LayerRef self(this);
		std::copy(inputs, inputs + input * effective_batch_size, output_buffers.first.data());
		for (auto& hook : forward_hooks) {
			hook(self, effective_batch_size, inputs, output_buffers.first.data(), output_buffers.first.data(), input);
		}
		if (output_pointer) {
			std::copy(output_buffers.first.data(), output_buffers.first.data() + input * effective_batch_size, output_pointer);
		}
		return output_buffers.first.data();
}
#ifdef TRAINING_ON
float* Layer::backward(float* upstream_gradient, const std::vector<int>& correct_indices, int temporary_batch_size) {
	int effective_batch_size = temporary_batch_size;
	if (!forward_hook_derivatives.empty()) {
		LayerRef self(this);
		for (auto& derivHook : forward_hook_derivatives) {
			derivHook(self, effective_batch_size, previous_inputs, previous_preactivations, upstream_gradient, output_buffers.first.data(), output, correct_indices);
		}
	}
	size_t total_weights = weight_count();
	size_t bias_offset = total_weights;
	std::fill(weight_gradients + bias_offset, weight_gradients + size, 0.0f);
	for (size_t b = 0; b < effective_batch_size; ++b) {
		for (size_t n = 0; n < neurons; ++n) {
			weight_gradients[bias_offset + n] += upstream_gradient[b * neurons + n];
		}
	}
	float* scratch_data = output_buffers.second.data();
	matmult(upstream_gradient, previous_inputs, weight_gradients, output, effective_batch_size, input);
	matmult(upstream_gradient, squared_inputs_buffer.data(), weight_gradients + input * output, output, effective_batch_size, input);
	matmult(upstream_gradient, linear(), scratch_data, effective_batch_size, output, input);
	for (size_t i = 0; i < input * effective_batch_size; ++i) {
		output_buffers.first.data()[i] = scratch_data[i];
		scratch_data[i] = 0.0f;
	}
	matmult(upstream_gradient, quadratic(), scratch_data, effective_batch_size, output, input);
	for (size_t i = 0; i < input * effective_batch_size; ++i) {
		output_buffers.first.data()[i] += 2.0f * previous_inputs[i] * scratch_data[i];
	}
	return output_buffers.first.data();
}
float* ParametricLayer::backward(float* upstream_gradient, const std::vector<int>& correct_indices, int temporary_batch_size) {
	int effective_batch_size = temporary_batch_size;
	size_t weight_count = input * weights_per_input;
	std::fill(weight_gradients, weight_gradients + weight_count, 0.0f);
	if (!forward_hook_derivatives.empty()) {
		LayerRef self(this);
		for (auto& derivHook : forward_hook_derivatives) {
			derivHook(self, effective_batch_size, previous_inputs, upstream_gradient, output_buffers.first.data(), input, correct_indices);
		}
	}
	return output_buffers.first.data();
}
#endif
void xavier_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	int weight_index = 0;
	for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
		const LayerArgs& current = layers[layer_idx];
		const LayerArgs& previous = layers[layer_idx - 1];
		int input_dimension = previous.layer_size;
		int output_dimension = current.layer_size;
		float scale = std::sqrt(6.0f / (input_dimension + output_dimension));
		int layer_weights = 0;
		if (current.kind == Parametric) {
			layer_weights = input_dimension * current.weights_per_input;
		} else {
			layer_weights = current.layer_size + current.layer_size * input_dimension * 2 * current.outputs_per_neuron;
		}
		for (int i = 0; i < layer_weights; ++i) {
			float random_value = static_cast<float>(rand()) / RAND_MAX;
			weights[weight_index + i] = (random_value * 2.0f - 1.0f) * scale;
		}
		weight_index += layer_weights;
	}
}

void he_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	int weight_index = 0;
	for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
		const LayerArgs& current = layers[layer_idx];
		const LayerArgs& previous = layers[layer_idx - 1];
		int input_dimension = previous.layer_size;
		float scale = std::sqrt(2.0f / input_dimension);
		int layer_weights = 0;
		if (current.kind == Parametric) {
			layer_weights = input_dimension * current.weights_per_input;
		} else {
			layer_weights = current.layer_size + current.layer_size * input_dimension * 2 * current.outputs_per_neuron;
		}
		for (int i = 0; i < layer_weights; ++i) {
			float random_value = static_cast<float>(rand()) / RAND_MAX;
			weights[weight_index + i] = (random_value * 2.0f - 1.0f) * scale;
		}
		weight_index += layer_weights;
	}
}

void uniform_random_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	for (int i = 0; i < total_size; ++i) {
		float random_value = static_cast<float>(rand()) / RAND_MAX;
		weights[i] = random_value - 0.5f;
	}
}

void zero_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	std::fill(weights, weights + total_size, 0.0f);
}

void setupNeuralNetwork(std::vector<LayerArgs> layersAdd, std::string weights_path, WeightInitFunc initialiser) {
	layers.clear();
	weights.clear();
	layer_sizes.clear();
	int maximum_neurons = 0;
	network_size = 0;
	int accumulated_batch_size = batch_size;
	int maximum_accumulated_batch = batch_size;
	int maximum_buffer_size = 0;
	LayerArgs* prev = nullptr;
	for (auto& args : layersAdd) {
		maximum_neurons = std::max(maximum_neurons, args.layer_size);
		int layer_batch = accumulated_batch_size;
		maximum_buffer_size = std::max(maximum_buffer_size, args.layer_size * layer_batch);
		if (prev == nullptr) {
			layer_sizes.push_back(args.layer_size);
			prev = &args;
			accumulated_batch_size *= args.outputs_per_neuron;
			maximum_accumulated_batch = std::max(maximum_accumulated_batch, accumulated_batch_size);
			continue;
		}
		accumulated_batch_size *= args.outputs_per_neuron;
		maximum_accumulated_batch = std::max(maximum_accumulated_batch, accumulated_batch_size);
		if (args.kind == Parametric) {
			network_size += prev->layer_size * args.weights_per_input;
		} else {
			network_size += args.layer_size + args.layer_size * prev->layer_size * 2 * args.outputs_per_neuron;
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
	accumulated_batch_size = batch_size;
	for (size_t i = 0; i < layersAdd.size(); ++i) {
		if (i > 0 && layersAdd[i].kind != Parametric) {
			size_t input_size = layersAdd[i-1].layer_size;
			size_t needed = input_size * accumulated_batch_size;
			if (needed > maximum_squared_input_size) maximum_squared_input_size = needed;
		}
		accumulated_batch_size *= layersAdd[i].outputs_per_neuron;
	}
	squared_inputs_buffer.resize(maximum_squared_input_size);
	size_t total_scratch = 0;
	for (auto& args : layersAdd) total_scratch += args.scratch_size;
	global_scratch_buffer.resize(total_scratch);
	#ifdef TRAINING_ON
	global_inputs.resize(maximum_neurons * maximum_neurons * maximum_accumulated_batch);
	global_preactivations.resize(maximum_neurons * maximum_neurons * maximum_accumulated_batch);
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
	int accumulated_batch = batch_size;
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
				layer.size = layersAdd[i].layer_size + layersAdd[i].layer_size * layersAdd[i-1].layer_size * 2 * layersAdd[i].outputs_per_neuron;
				layer.weights_begin = weight_start + accumulated_weights;
			}
		}
		layer.batch = accumulated_batch;
		accumulated_batch *= layersAdd[i].outputs_per_neuron;
		size_t output_data_size = layer.output * layer.batch * layersAdd[i].outputs_per_neuron;
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

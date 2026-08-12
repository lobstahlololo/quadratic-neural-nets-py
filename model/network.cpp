#include <cstdint>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "network.h"
#include "../math/math.h"
bool is_setup = false;
int batch_size = 1;
int network_size = 0;
std::pair<std::vector<float>, std::vector<float>> output_buffers;
std::vector<float> squared_inputs_buffer;
std::vector<float> weights;
std::vector<float> global_scratch_buffer;
std::vector<std::variant<Layer, ParametricLayer>> layers;
std::vector<int> layer_sizes;
#ifdef TRAINING_ON
std::vector<float> global_inputs;
std::vector<float> global_preactivations;
std::vector<float> global_grads;
#endif
float *Layer::forward(float *inputs, int batch_count, std::vector<int> &sequence_lengths)
{
	int total_rows = 0;
	for (int s : sequence_lengths)
		total_rows += s;
	if (size == 0)
	{
#ifdef TRAINING_ON
		previous_inputs = inputs;
		previous_preactivations = inputs;
#endif
		if (forward_hooks.empty())
		{
			return inputs;
		}
		LayerRef self(this);
		std::copy(inputs, inputs + output * total_rows, output_buffers.second.data());
		for (auto &hook : forward_hooks)
		{
			hook(self, batch_count, sequence_lengths, inputs, output_buffers.second.data(), output_buffers.second.data(), output);
		}
		std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_buffers.first.data());
#ifdef TRAINING_ON
		std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_pointer);
#endif
		std::fill(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, 0.0f);
		return output_buffers.first.data();
	}
	float *in_ptr = inputs;
	for (std::size_t i = 0; i < input * total_rows; ++i)
	{
		squared_inputs_buffer[i] = in_ptr[i] * in_ptr[i];
	}
	float max = 0.0f;
	matmult(inputs, linear(), output_buffers.second.data(), total_rows, input, output, false, true, 1.0f, 0.0f);
	matmult(squared_inputs_buffer.data(), quadratic(), output_buffers.second.data(), total_rows, input, output, false, true, 1.0f, 1.0f);
	for (std::size_t i = 0; i < total_rows; ++i)
	{
		for (std::size_t j = 0; j < output; ++j)
		{
			output_buffers.second[i * output + j] += biases()[j];
			max = std::max(max, output_buffers.second[i * output + j]);
		}
	}
	// std::cout << "Max pre-activation: " << max << "\n";
	if (!forward_hooks.empty())
	{
		LayerRef self(this);
#ifdef TRAINING_ON
		std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, previous_preactivations);
#endif
		for (auto &hook : forward_hooks)
		{
			hook(self, batch_count, sequence_lengths, inputs, output_buffers.second.data(), output_buffers.second.data(), output);
		}
	}
	// debug
	float max_postact = 0.0f;
	for (std::size_t i = 0; i < output * total_rows; ++i)
	{
		max_postact = std::max(max_postact, output_buffers.second[i]);
	}
// std::cout << "Max post-activation: " << max_postact << "\n";
#ifdef TRAINING_ON
	std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_pointer);
#endif
	std::copy(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, output_buffers.first.data());
	std::fill(output_buffers.second.data(), output_buffers.second.data() + output * total_rows, 0.0f);
	return output_buffers.first.data();
}
float *ParametricLayer::forward(float *inputs, int batch_count, std::vector<int> &sequence_lengths)
{
	int total_rows = 0;
	for (int s : sequence_lengths)
		total_rows += s;
#ifdef TRAINING_ON
	previous_preactivations = output_buffers.first.data();
#endif
	if (forward_hooks.empty())
	{
		return inputs;
	}
	LayerRef self(this);
	std::copy(inputs, inputs + input * total_rows, output_buffers.first.data());
	for (auto &hook : forward_hooks)
	{
		hook(self, batch_count, sequence_lengths, inputs, output_buffers.first.data(), output_buffers.first.data(), input);
	}
	float max = 0.0f;
	for (std::size_t i = 0; i < input * total_rows; ++i)
	{
		if (std::abs(output_buffers.first[i]) > max)
			max = std::abs(output_buffers.first[i]);
	}
// std::cout << "Max post-activation: " << max << "\n";
#ifdef TRAINING_ON
	std::copy(output_buffers.first.data(), output_buffers.first.data() + input * total_rows, output_pointer);
#endif
	return output_buffers.first.data();
}
#ifdef TRAINING_ON
float *Layer::backward(float *upstream_gradient, int batch_count, std::vector<int> &sequence_lengths, const std::vector<int> &correct_indices)
{
	int total_rows = 0;
	for (int s : sequence_lengths)
		total_rows += s;
	float *post_activation_gradient = upstream_gradient;
	if (!forward_hook_derivatives.empty())
	{
		LayerRef self(this);
		for (auto &derivHook : forward_hook_derivatives)
		{
			derivHook(self, batch_count, sequence_lengths, previous_inputs, previous_preactivations, upstream_gradient, output_buffers.first.data(), output, correct_indices);
		}
		post_activation_gradient = output_buffers.first.data();
	}
	if (size == 0)
	{
		return post_activation_gradient;
	}
	size_t total_weights = weight_count();
	size_t bias_offset = total_weights;
	std::fill(weight_gradients, weight_gradients + size, 0.0f);
	for (size_t b = 0; b < total_rows; ++b)
	{
		for (size_t n = 0; n < neurons; ++n)
		{
			weight_gradients[bias_offset + n] += post_activation_gradient[b * neurons + n];
		}
	}
	float *scratch_data = output_buffers.second.data();
	matmult(post_activation_gradient, previous_inputs, weight_gradients, output, total_rows, input, true, false, 1.0f, 1.0f);
	matmult(post_activation_gradient, squared_inputs_buffer.data(), weight_gradients + input * output, output, total_rows, input, true, false, 1.0f, 1.0f);
	matmult(post_activation_gradient, linear(), squared_inputs_buffer.data(), total_rows, output, input, false, false, 1.0f, 0.0f);
	matmult(post_activation_gradient, quadratic(), scratch_data, total_rows, output, input, false, false, 1.0f, 0.0f);
	for (size_t i = 0; i < input * total_rows; ++i)
	{
		output_buffers.first.data()[i] = squared_inputs_buffer[i] + 2.0f * previous_inputs[i] * scratch_data[i];
	}
	return output_buffers.first.data();
}
float *ParametricLayer::backward(float *upstream_gradient, int batch_count, std::vector<int> &sequence_lengths, const std::vector<int> &correct_indices)
{
	int total_rows = 0;
	for (int s : sequence_lengths)
		total_rows += s;
	size_t weight_count = input * weights_per_input;
	std::fill(weight_gradients, weight_gradients + weight_count, 0.0f);
	if (!forward_hook_derivatives.empty())
	{
		LayerRef self(this);
		for (auto &derivHook : forward_hook_derivatives)
		{
			derivHook(self, batch_count, sequence_lengths, previous_inputs, previous_preactivations, upstream_gradient, output_buffers.first.data(), input, correct_indices);
		}
		return output_buffers.first.data();
	}
	return upstream_gradient;
}
#endif
void setupNeuralNetwork(std::vector<LayerArgs> layersAdd, std::string weights_path, WeightInitFunc initialiser, int buffer_multiplier, setupNNHookFunction setup_hook)
{
	layers.clear();
	weights.clear();
	layer_sizes.clear();
	int maximum_neurons = 0;
	network_size = 0;
	int max_token_count = batch_size * buffer_multiplier;
	int maximum_buffer_size = 0;
	int total_output_tokens = 0;
	LayerArgs *prev = nullptr;
	for (auto &args : layersAdd)
	{
		maximum_neurons = std::max(maximum_neurons, args.layer_size);
		int token_capacity = args.layer_size * max_token_count;
		total_output_tokens += token_capacity;
		if (token_capacity > maximum_buffer_size)
			maximum_buffer_size = token_capacity;
		if (prev == nullptr)
		{
			layer_sizes.push_back(args.layer_size);
			prev = &args;
			continue;
		}
		if (args.kind == Parametric)
		{
			network_size += prev->layer_size * args.weights_per_input + args.extra_weights;
		}
		else
		{
			network_size += args.layer_size + args.layer_size * prev->layer_size * 2 + args.extra_weights;
		}
		layer_sizes.push_back(args.layer_size);
		prev = &args;
	}
	if (!layersAdd.empty() && layersAdd[0].extra_weights > 0)
	{
		network_size += layersAdd[0].extra_weights;
	}
	auto resizeBuffers = [&](size_t size)
	{
		output_buffers.first.resize(size);
		output_buffers.second.resize(size);
	};
	resizeBuffers(maximum_buffer_size);
	squared_inputs_buffer.resize(maximum_buffer_size);
	size_t total_scratch = 0;
	for (auto &args : layersAdd)
		total_scratch += args.scratch_size;
	global_scratch_buffer.resize(total_scratch);
#ifdef TRAINING_ON
	global_inputs.resize(total_output_tokens);
	global_preactivations.resize(total_output_tokens);
	global_grads.resize(network_size);
#endif
	weights.resize(network_size);

	if (weights_path.empty())
	{
		initialiser(weights.data(), network_size, layersAdd);
	}
	else
	{
		std::ifstream file(weights_path, std::ios::binary);
		if (!file)
		{
			throw std::runtime_error("wrong filepath");
		}
		char magic[4];
		if (file.read(magic, 4) && std::string(magic, 4) == "Q2CP")
		{
			uint8_t *raw = reinterpret_cast<uint8_t *>(weights.data());
			size_t raw_size = network_size * sizeof(float);
			size_t i = 0;
			while (i < raw_size && file)
			{
				uint8_t count, val;
				if (!file.read(reinterpret_cast<char *>(&count), 1))
					break;
				if (!file.read(reinterpret_cast<char *>(&val), 1))
					break;
				for (size_t c = 0; c < count && i < raw_size; ++c)
				{
					raw[i++] = val;
				}
			}
		}
		else
		{
			file.seekg(0, std::ios::end);
			std::streampos fileSize = file.tellg();
			file.seekg(0, std::ios::beg);
			if (static_cast<long>(network_size * sizeof(float)) != fileSize)
			{
				throw std::runtime_error("wrong file size relation to weights");
			}
			file.read(reinterpret_cast<char *>(weights.data()), network_size * sizeof(float));
		}
	}

	float *weight_start = weights.data();
	std::size_t previous_output_offset = 0;
	std::size_t accumulated_weights = 0;
	std::size_t gradient_offset = 0;
	std::size_t scratch_offset = 0;
	for (std::size_t i = 0; i < layersAdd.size(); ++i)
	{
		if (layersAdd[i].kind == Parametric)
		{
			ParametricLayer layer;
			for (auto &h : layersAdd[i].hooks)
				layer.forward_hooks.push_back(h);
			for (auto &hd : layersAdd[i].hook_gradients)
				layer.forward_hook_derivatives.push_back(hd);
			layer.extra_args = layersAdd[i].extra_args;
			layer.input = (i == 0) ? layersAdd[i].layer_size : layersAdd[i - 1].layer_size;
			layer.output = layersAdd[i].layer_size;
			layer.weights_per_input = layersAdd[i].weights_per_input;
			layer.weights_begin = (i == 0) ? weight_start : weight_start + accumulated_weights;
			layer.extra_weights_size = layersAdd[i].extra_weights;
			layer.extra_weights_begin = layer.weights_begin + layer.input * layer.weights_per_input;
			layer.scratch_size = layersAdd[i].scratch_size;
			layer.scratch_pointer = layer.scratch_size > 0 ? global_scratch_buffer.data() + scratch_offset : nullptr;
			scratch_offset += layer.scratch_size;
			size_t output_data_size = layer.output * max_token_count;
#ifdef TRAINING_ON
			layer.previous_preactivations = global_preactivations.data() + previous_output_offset;
			layer.weight_gradients = global_grads.data() + gradient_offset;
			layer.extra_weight_gradients = layer.weight_gradients + layer.input * layer.weights_per_input;
			layer.output_pointer = global_inputs.data() + previous_output_offset;
			if (i > 0)
			{
				layer.previous_inputs = std::visit([](auto &l) -> float *
												   { return l.output_pointer; }, layers.back());
			}
			else
			{
				layer.previous_inputs = nullptr;
			}
#endif
			layers.push_back(layer);
			previous_output_offset += output_data_size;
			size_t param_size = layer.input * layer.weights_per_input + layer.extra_weights_size;
			if (i > 0 || layer.extra_weights_size > 0)
			{
				gradient_offset += param_size;
				accumulated_weights += param_size;
			}
		}
		else
		{
			Layer layer;
			for (auto &h : layersAdd[i].hooks)
				layer.forward_hooks.push_back(h);
			for (auto &hd : layersAdd[i].hook_gradients)
				layer.forward_hook_derivatives.push_back(hd);
			layer.neurons = layersAdd[i].layer_size;
			layer.extra_args = layersAdd[i].extra_args;
			if (i == 0)
			{
				layer.input = layersAdd[i].layer_size;
				layer.output = layersAdd[i].layer_size;
				layer.size = 0;
				layer.neurons = layersAdd[i].layer_size;
				layer.weights_begin = weight_start;
				layer.extra_weights_size = layersAdd[i].extra_weights;
				layer.extra_weights_begin = layer.weights_begin;
				layer.scratch_size = 0;
				layer.scratch_pointer = nullptr;
			}
			else
			{
				layer.input = layersAdd[i - 1].layer_size;
				layer.output = layersAdd[i].layer_size;
				layer.size = layersAdd[i].layer_size + layersAdd[i].layer_size * layersAdd[i - 1].layer_size * 2;
				layer.weights_begin = weight_start + accumulated_weights;
				layer.extra_weights_size = layersAdd[i].extra_weights;
				layer.extra_weights_begin = layer.weights_begin + layer.size;
				layer.scratch_size = layersAdd[i].scratch_size;
				layer.scratch_pointer = layer.scratch_size > 0 ? global_scratch_buffer.data() + scratch_offset : nullptr;
				scratch_offset += layer.scratch_size;
			}
			size_t output_data_size = layer.output * max_token_count;
			layer.scratch_size = layersAdd[i].scratch_size;
			layer.scratch_pointer = layer.scratch_size > 0 ? global_scratch_buffer.data() + scratch_offset : nullptr;
			scratch_offset += layer.scratch_size;
#ifdef TRAINING_ON
			layer.previous_preactivations = global_preactivations.data() + previous_output_offset;
			layer.weight_gradients = global_grads.data() + gradient_offset;
			layer.extra_weight_gradients = layer.weight_gradients + layer.size;
			layer.output_pointer = global_inputs.data() + previous_output_offset;
			if (i > 0)
			{
				layer.previous_inputs = std::visit([](auto &l) -> float *
												   { return l.output_pointer; }, layers.back());
			}
			else
			{
				layer.previous_inputs = nullptr;
			}
#endif
			layers.push_back(layer);
			previous_output_offset += output_data_size;
			if (i > 0 || layer.extra_weights_size > 0)
			{
				gradient_offset += layer.size + layer.extra_weights_size;
				accumulated_weights += layer.size + layer.extra_weights_size;
			}
		}
	}
	is_setup = true;
	if (setup_hook)
		setup_hook();
}

void save_weights(std::string path, bool compress)
{
	std::ofstream file(path, std::ios::binary);
	if (!file)
	{
		throw std::runtime_error("wrong filepath");
	}
	if (compress)
	{
		file.write("Q2CP", 4);
		const uint8_t *raw = reinterpret_cast<const uint8_t *>(weights.data());
		size_t raw_size = network_size * sizeof(float);
		for (size_t i = 0; i < raw_size;)
		{
			uint8_t val = raw[i];
			size_t count = 1;
			while (i + count < raw_size && count < 255 && raw[i + count] == val)
				count++;
			uint8_t c = static_cast<uint8_t>(count);
			file.write(reinterpret_cast<const char *>(&c), 1);
			file.write(reinterpret_cast<const char *>(&val), 1);
			i += count;
		}
	}
	else
	{
		file.write(reinterpret_cast<const char *>(weights.data()), network_size * sizeof(float));
	}
}
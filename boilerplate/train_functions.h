#ifndef BOILERPLATE_TRAIN_FUNCTIONS_H
#define BOILERPLATE_TRAIN_FUNCTIONS_H
#include "../model/network.h"
#include "../train/train.h"
#include <vector>
#include <variant>
#include <cmath>
#include <algorithm>
#include <iostream>
inline float global_weight_decay = 0.01f;
inline float global_max_trust_ratio = 10.0f;
inline float train_adams(std::vector<std::variant<Layer, ParametricLayer>> &layers_list, const std::vector<float> &training_data, const std::vector<int> &correct_indices, const std::vector<float> &required_output, float learning_rate, const LossFunc &loss_function, const LossDerivative &loss_derivative, int training_step, int batch_count, std::vector<int> &sequence_lengths)
{
	int total_rows = 0;
	for (int sequence_length : sequence_lengths)
		total_rows += sequence_length;
	float *allocated_input_pointer = const_cast<float *>(training_data.data());
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = parametric_layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
	}
	int last_layer_output = 0;
	std::visit([&](auto &current_layer)
			   { last_layer_output = current_layer.output; }, layers_list.back());
	std::cout << "DEBUG: train_adams reached output check. "
			  << "rows=" << total_rows
			  << " output=" << last_layer_output << "\n";
	for (int i = 0; i < last_layer_output * total_rows; ++i)
	{
		if (!std::isfinite(output_buffers.first[i]))
		{
			std::cout << "DEBUG: NaN/Inf in network output at index "
					  << i << "\n";
			return NAN;
		}
	}
	std::vector<float> sample_losses(total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		sample_losses[row_index] = loss_function(output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, last_layer_output);
		if (!std::isfinite(sample_losses[row_index]))
		{
			std::cout << "DEBUG: NaN/Inf sample loss at row "
					  << row_index << "\n";
			std::cout << "  target index: " << correct_indices[row_index] << "\n";
			std::cout << "  output size: " << last_layer_output << "\n";
			return NAN;
		}
	}
	float batch_loss = 0.0f;
	for (int row_index = 0; row_index < total_rows; ++row_index)
		batch_loss += sample_losses[row_index];
	batch_loss /= total_rows;
	if (!std::isfinite(batch_loss))
	{
		std::cout << "NAN/INF IN LOSS: " << batch_loss << "\n";
		return NAN;
	}
	std::vector<float> downstream_gradient(last_layer_output * total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		loss_derivative(sample_losses[row_index], output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, downstream_gradient.data() + last_layer_output * row_index, last_layer_output);
	}
	std::vector<int> mutable_sequence_lengths = sequence_lengths;
	float *gradient_pointer = downstream_gradient.data();
	for (int layer_index = layers_list.size() - 1; layer_index >= 0; --layer_index)
	{
		std::cout << "DEBUG: backward layer " << layer_index << "\n";

		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			gradient_pointer = layer_pointer->backward(
				gradient_pointer,
				batch_count,
				mutable_sequence_lengths,
				correct_indices);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			gradient_pointer = parametric_layer_pointer->backward(
				gradient_pointer,
				batch_count,
				mutable_sequence_lengths,
				correct_indices);
		}

		if (gradient_pointer == nullptr)
		{
			std::cout << "DEBUG: NULL gradient after layer "
					  << layer_index << "\n";
			return NAN;
		}

		std::cout << "DEBUG: finished backward layer "
				  << layer_index << "\n";
	}
	float scaled_learning_rate = learning_rate / batch_size;
	for (float g : downstream_gradient)
	{
		if (!std::isfinite(g))
		{
			std::cout << "NAN/INF IN GRADIENT\n";
			return NAN;
		}
	}
	float first_moment_decay_power = std::pow(0.9f, training_step + 1);
	float second_moment_decay_power = std::pow(0.999f, training_step + 1);
	size_t moment_offset = 0;
	float maximum_gradient_value = 0.0f;
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			for (int weight_index = 0; weight_index < layer_pointer->size; ++weight_index)
			{
				float gradient_value = layer_pointer->weight_gradients[weight_index];
				maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
				first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
				second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
				float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
				float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float weight_value = layer_pointer->weights_begin[weight_index];
				layer_pointer->weights_begin[weight_index] -= scaled_learning_rate * (adam_step_value + global_weight_decay * weight_value);
			}
			moment_offset += layer_pointer->size;
			for (int weight_index = 0; weight_index < layer_pointer->extra_weights_size; ++weight_index)
			{
				float gradient_value = layer_pointer->extra_weight_gradients[weight_index];
				maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
				first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
				second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
				float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
				float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float weight_value = layer_pointer->extra_weights_begin[weight_index];
				layer_pointer->extra_weights_begin[weight_index] -= scaled_learning_rate * (adam_step_value + global_weight_decay * weight_value);
			}
			moment_offset += layer_pointer->extra_weights_size;
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			int total_weights_count = parametric_layer_pointer->input * parametric_layer_pointer->weights_per_input;
			for (int weight_index = 0; weight_index < total_weights_count; ++weight_index)
			{
				float gradient_value = parametric_layer_pointer->weight_gradients[weight_index];
				maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
				first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
				second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
				float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
				float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float weight_value = parametric_layer_pointer->weights_begin[weight_index];
				parametric_layer_pointer->weights_begin[weight_index] -= scaled_learning_rate * (adam_step_value + global_weight_decay * weight_value);
			}
			moment_offset += total_weights_count;
			for (int weight_index = 0; weight_index < parametric_layer_pointer->extra_weights_size; ++weight_index)
			{
				float gradient_value = parametric_layer_pointer->extra_weight_gradients[weight_index];
				maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
				first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
				second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
				float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
				float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float weight_value = parametric_layer_pointer->extra_weights_begin[weight_index];
				parametric_layer_pointer->extra_weights_begin[weight_index] -= scaled_learning_rate * (adam_step_value + global_weight_decay * weight_value);
			}
			moment_offset += parametric_layer_pointer->extra_weights_size;
		}
	}
	if (trainHook)
		trainHook();
	return batch_loss;
}
inline float train_lamb(std::vector<std::variant<Layer, ParametricLayer>> &layers_list, const std::vector<float> &training_data, const std::vector<int> &correct_indices, const std::vector<float> &required_output, float learning_rate, const LossFunc &loss_function, const LossDerivative &loss_derivative, int training_step, int batch_count, std::vector<int> &sequence_lengths)
{
	int total_rows = 0;
	for (int sequence_length : sequence_lengths)
		total_rows += sequence_length;
	float *allocated_input_pointer = const_cast<float *>(training_data.data());
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = parametric_layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
	}
	int last_layer_output = 0;
	std::visit([&](auto &current_layer)
			   { last_layer_output = current_layer.output; }, layers_list.back());
	std::vector<float> sample_losses(total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		sample_losses[row_index] = loss_function(output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, last_layer_output);
	}
	float batch_loss = 0.0f;
	for (int row_index = 0; row_index < total_rows; ++row_index)
		batch_loss += sample_losses[row_index];
	batch_loss /= total_rows;
	std::vector<float> downstream_gradient(last_layer_output * total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		loss_derivative(sample_losses[row_index], output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, downstream_gradient.data() + last_layer_output * row_index, last_layer_output);
	}
	std::vector<int> mutable_sequence_lengths = sequence_lengths;
	float *gradient_pointer = downstream_gradient.data();
	for (int layer_index = layers_list.size() - 1; layer_index >= 0; --layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			gradient_pointer = layer_pointer->backward(gradient_pointer, batch_count, mutable_sequence_lengths, correct_indices);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			gradient_pointer = parametric_layer_pointer->backward(gradient_pointer, batch_count, mutable_sequence_lengths, correct_indices);
		}
	}
	float scaled_learning_rate = learning_rate / batch_size;
	float first_moment_decay_power = std::pow(0.9f, training_step + 1);
	float second_moment_decay_power = std::pow(0.999f, training_step + 1);
	size_t moment_offset = 0;
	float maximum_gradient_value = 0.0f;
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			if (layer_pointer->size > 0)
			{
				std::vector<float> update_temporary(layer_pointer->size);
				float weight_square_sum = 0.0f;
				float update_square_sum = 0.0f;
				for (int weight_index = 0; weight_index < layer_pointer->size; ++weight_index)
				{
					float gradient_value = layer_pointer->weight_gradients[weight_index];
					maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
					first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
					second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
					float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
					float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float weight_value = layer_pointer->weights_begin[weight_index];
					float update_value = adam_step_value + global_weight_decay * weight_value;
					update_temporary[weight_index] = update_value;
					weight_square_sum += weight_value * weight_value;
					update_square_sum += update_value * update_value;
				}
				float weights_l2_norm = std::sqrt(weight_square_sum);
				float updates_l2_norm = std::sqrt(update_square_sum);
				float trust_ratio = 1.0f;
				if (weights_l2_norm > 0.0f && updates_l2_norm > 0.0f)
				{
					trust_ratio = weights_l2_norm / updates_l2_norm;
					if (trust_ratio > global_max_trust_ratio)
					{
						trust_ratio = global_max_trust_ratio;
					}
				}
				for (int weight_index = 0; weight_index < layer_pointer->size; ++weight_index)
				{
					layer_pointer->weights_begin[weight_index] -= scaled_learning_rate * trust_ratio * update_temporary[weight_index];
				}
			}
			moment_offset += layer_pointer->size;
			if (layer_pointer->extra_weights_size > 0)
			{
				std::vector<float> update_temporary(layer_pointer->extra_weights_size);
				float weight_square_sum = 0.0f;
				float update_square_sum = 0.0f;
				for (int weight_index = 0; weight_index < layer_pointer->extra_weights_size; ++weight_index)
				{
					float gradient_value = layer_pointer->extra_weight_gradients[weight_index];
					maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
					first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
					second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
					float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
					float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float weight_value = layer_pointer->extra_weights_begin[weight_index];
					float update_value = adam_step_value + global_weight_decay * weight_value;
					update_temporary[weight_index] = update_value;
					weight_square_sum += weight_value * weight_value;
					update_square_sum += update_value * update_value;
				}
				float weights_l2_norm = std::sqrt(weight_square_sum);
				float updates_l2_norm = std::sqrt(update_square_sum);
				float trust_ratio = 1.0f;
				if (weights_l2_norm > 0.0f && updates_l2_norm > 0.0f)
				{
					trust_ratio = weights_l2_norm / updates_l2_norm;
					if (trust_ratio > global_max_trust_ratio)
						trust_ratio = global_max_trust_ratio;
				}
				for (int weight_index = 0; weight_index < layer_pointer->extra_weights_size; ++weight_index)
				{
					layer_pointer->extra_weights_begin[weight_index] -= scaled_learning_rate * trust_ratio * update_temporary[weight_index];
				}
			}
			moment_offset += layer_pointer->extra_weights_size;
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			int total_weights_count = parametric_layer_pointer->input * parametric_layer_pointer->weights_per_input;
			if (total_weights_count > 0)
			{
				std::vector<float> update_temporary(total_weights_count);
				float weight_square_sum = 0.0f;
				float update_square_sum = 0.0f;
				for (int weight_index = 0; weight_index < total_weights_count; ++weight_index)
				{
					float gradient_value = parametric_layer_pointer->weight_gradients[weight_index];
					maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
					first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
					second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
					float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
					float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float weight_value = parametric_layer_pointer->weights_begin[weight_index];
					float update_value = adam_step_value + global_weight_decay * weight_value;
					update_temporary[weight_index] = update_value;
					weight_square_sum += weight_value * weight_value;
					update_square_sum += update_value * update_value;
				}
				float weights_l2_norm = std::sqrt(weight_square_sum);
				float updates_l2_norm = std::sqrt(update_square_sum);
				float trust_ratio = 1.0f;
				if (weights_l2_norm > 0.0f && updates_l2_norm > 0.0f)
				{
					trust_ratio = weights_l2_norm / updates_l2_norm;
					if (trust_ratio > global_max_trust_ratio)
					{
						trust_ratio = global_max_trust_ratio;
					}
				}
				for (int weight_index = 0; weight_index < total_weights_count; ++weight_index)
				{
					parametric_layer_pointer->weights_begin[weight_index] -= scaled_learning_rate * trust_ratio * update_temporary[weight_index];
				}
			}
			moment_offset += total_weights_count;
			if (parametric_layer_pointer->extra_weights_size > 0)
			{
				std::vector<float> update_temporary(parametric_layer_pointer->extra_weights_size);
				float weight_square_sum = 0.0f;
				float update_square_sum = 0.0f;
				for (int weight_index = 0; weight_index < parametric_layer_pointer->extra_weights_size; ++weight_index)
				{
					float gradient_value = parametric_layer_pointer->extra_weight_gradients[weight_index];
					maximum_gradient_value = std::max(maximum_gradient_value, std::abs(gradient_value));
					first_moment_buffer[moment_offset + weight_index] = 0.9f * first_moment_buffer[moment_offset + weight_index] + 0.1f * gradient_value;
					second_moment_buffer[moment_offset + weight_index] = 0.999f * second_moment_buffer[moment_offset + weight_index] + 0.001f * gradient_value * gradient_value;
					float first_bias_correction = first_moment_buffer[moment_offset + weight_index] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + weight_index] / (1.0f - second_moment_decay_power);
					float adam_step_value = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float weight_value = parametric_layer_pointer->extra_weights_begin[weight_index];
					float update_value = adam_step_value + global_weight_decay * weight_value;
					update_temporary[weight_index] = update_value;
					weight_square_sum += weight_value * weight_value;
					update_square_sum += update_value * update_value;
				}
				float weights_l2_norm = std::sqrt(weight_square_sum);
				float updates_l2_norm = std::sqrt(update_square_sum);
				float trust_ratio = 1.0f;
				if (weights_l2_norm > 0.0f && updates_l2_norm > 0.0f)
				{
					trust_ratio = weights_l2_norm / updates_l2_norm;
					if (trust_ratio > global_max_trust_ratio)
						trust_ratio = global_max_trust_ratio;
				}
				for (int weight_index = 0; weight_index < parametric_layer_pointer->extra_weights_size; ++weight_index)
				{
					parametric_layer_pointer->extra_weights_begin[weight_index] -= scaled_learning_rate * trust_ratio * update_temporary[weight_index];
				}
			}
			moment_offset += parametric_layer_pointer->extra_weights_size;
		}
	}
	if (trainHook)
		trainHook();
	return batch_loss;
}
inline float regular_gd(std::vector<std::variant<Layer, ParametricLayer>> &layers_list, const std::vector<float> &training_data, const std::vector<int> &correct_indices, const std::vector<float> &required_output, float learning_rate, const LossFunc &loss_function, const LossDerivative &loss_derivative, int training_step, int batch_count, std::vector<int> &sequence_lengths)
{
	int total_rows = 0;
	for (int sequence_length : sequence_lengths)
		total_rows += sequence_length;
	float *allocated_input_pointer = const_cast<float *>(training_data.data());
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = parametric_layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
	}
	int last_layer_output = 0;
	std::visit([&](auto &current_layer)
			   { last_layer_output = current_layer.output; }, layers_list.back());
	std::vector<float> sample_losses(total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		sample_losses[row_index] = loss_function(output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, last_layer_output);
	}
	float batch_loss = 0.0f;
	for (int row_index = 0; row_index < total_rows; ++row_index)
		batch_loss += sample_losses[row_index];
	batch_loss /= total_rows;
	std::vector<float> downstream_gradient(last_layer_output * total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		loss_derivative(sample_losses[row_index], output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, downstream_gradient.data() + last_layer_output * row_index, last_layer_output);
	}
	std::vector<int> mutable_sequence_lengths = sequence_lengths;
	float *gradient_pointer = downstream_gradient.data();
	for (int layer_index = layers_list.size() - 1; layer_index >= 0; --layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			gradient_pointer = layer_pointer->backward(gradient_pointer, batch_count, mutable_sequence_lengths, correct_indices);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			gradient_pointer = parametric_layer_pointer->backward(gradient_pointer, batch_count, mutable_sequence_lengths, correct_indices);
		}
	}
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			for (int weight_index = 0; weight_index < layer_pointer->size; ++weight_index)
			{
				float gradient_value = layer_pointer->weight_gradients[weight_index];
				layer_pointer->weights_begin[weight_index] -= learning_rate * gradient_value;
			}
			for (int weight_index = 0; weight_index < layer_pointer->extra_weights_size; ++weight_index)
			{
				float gradient_value = layer_pointer->extra_weight_gradients[weight_index];
				layer_pointer->extra_weights_begin[weight_index] -= learning_rate * gradient_value;
			}
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			int total_weights_count = parametric_layer_pointer->input * parametric_layer_pointer->weights_per_input;
			for (int weight_index = 0; weight_index < total_weights_count; ++weight_index)
			{
				float gradient_value = parametric_layer_pointer->weight_gradients[weight_index];
				parametric_layer_pointer->weights_begin[weight_index] -= learning_rate * gradient_value;
			}
			for (int weight_index = 0; weight_index < parametric_layer_pointer->extra_weights_size; ++weight_index)
			{
				float gradient_value = parametric_layer_pointer->extra_weight_gradients[weight_index];
				parametric_layer_pointer->extra_weights_begin[weight_index] -= learning_rate * gradient_value;
			}
		}
	}
	if (trainHook)
		trainHook();
	return batch_loss;
}
inline float stochastic_gd(std::vector<std::variant<Layer, ParametricLayer>> &layers_list, const std::vector<float> &training_data, const std::vector<int> &correct_indices, const std::vector<float> &required_output, float learning_rate, const LossFunc &loss_function, const LossDerivative &loss_derivative, int training_step, int batch_count, std::vector<int> &sequence_lengths)
{
	int total_rows = 0;
	for (int sequence_length : sequence_lengths)
		total_rows += sequence_length;
	float *allocated_input_pointer = const_cast<float *>(training_data.data());
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			allocated_input_pointer = parametric_layer_pointer->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
	}
	int last_layer_output = 0;
	std::visit([&](auto &current_layer)
			   { last_layer_output = current_layer.output; }, layers_list.back());
	std::vector<float> sample_losses(total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		sample_losses[row_index] = loss_function(output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, last_layer_output);
	}
	float batch_loss = 0.0f;
	for (int row_index = 0; row_index < total_rows; ++row_index)
		batch_loss += sample_losses[row_index];
	batch_loss /= total_rows;
	std::vector<float> downstream_gradient(last_layer_output * total_rows);
	for (int row_index = 0; row_index < total_rows; ++row_index)
	{
		std::vector<int> single_target = {correct_indices[row_index]};
		loss_derivative(sample_losses[row_index], output_buffers.first.data() + last_layer_output * row_index, required_output.data() + last_layer_output * row_index, single_target, downstream_gradient.data() + last_layer_output * row_index, last_layer_output);
	}
	std::vector<int> mutable_sequence_lengths = sequence_lengths;
	float *gradient_pointer = downstream_gradient.data();
	for (int layer_index = layers_list.size() - 1; layer_index >= 0; --layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			gradient_pointer = layer_pointer->backward(gradient_pointer, batch_count, mutable_sequence_lengths, correct_indices);
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			gradient_pointer = parametric_layer_pointer->backward(gradient_pointer, batch_count, mutable_sequence_lengths, correct_indices);
		}
	}
	float scaled_learning_rate = learning_rate / batch_size;
	for (int layer_index = 0; layer_index < layers_list.size(); ++layer_index)
	{
		if (auto *layer_pointer = std::get_if<Layer>(&layers_list[layer_index]))
		{
			for (int weight_index = 0; weight_index < layer_pointer->size; ++weight_index)
			{
				float gradient_value = layer_pointer->weight_gradients[weight_index];
				layer_pointer->weights_begin[weight_index] -= scaled_learning_rate * gradient_value;
			}
			for (int weight_index = 0; weight_index < layer_pointer->extra_weights_size; ++weight_index)
			{
				float gradient_value = layer_pointer->extra_weight_gradients[weight_index];
				layer_pointer->extra_weights_begin[weight_index] -= scaled_learning_rate * gradient_value;
			}
		}
		else if (auto *parametric_layer_pointer = std::get_if<ParametricLayer>(&layers_list[layer_index]))
		{
			int total_weights_count = parametric_layer_pointer->input * parametric_layer_pointer->weights_per_input;
			for (int weight_index = 0; weight_index < total_weights_count; ++weight_index)
			{
				float gradient_value = parametric_layer_pointer->weight_gradients[weight_index];
				parametric_layer_pointer->weights_begin[weight_index] -= scaled_learning_rate * gradient_value;
			}
			for (int weight_index = 0; weight_index < parametric_layer_pointer->extra_weights_size; ++weight_index)
			{
				float gradient_value = parametric_layer_pointer->extra_weight_gradients[weight_index];
				parametric_layer_pointer->extra_weights_begin[weight_index] -= scaled_learning_rate * gradient_value;
			}
		}
	}
	if (trainHook)
		trainHook();
	return batch_loss;
}
#endif

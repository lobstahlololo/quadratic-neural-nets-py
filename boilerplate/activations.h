#ifndef ACTIVATIONS_IMPORTED
#define ACTIVATIONS_IMPORTED
#include "../model/network.h"
#include <cmath>
#include <algorithm>
#include <variant>
#include <vector>
#include <iostream>
#ifndef TRAINING_ON
std::vector<std::vector<float>> kv_cache_pool;
#endif
HookFunc Residual = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		output_values[element_index] = preactivation_values[element_index] + original_inputs[element_index];
	}
};
HookDerivative ResidualGradHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		output_gradient[element_index] = upstream_gradient[element_index];
	}
};
HookFunc NonLearnableLayerNorm = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	int current_offset = 0;
	for (int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
		int sequence_rows = sequence_lengths[batch_index];
		for (int row_index = 0; row_index < sequence_rows; ++row_index)
		{
			float *input_sample = preactivation_values + current_offset;
			float *output_sample = output_values + current_offset;
			float mean_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				mean_value += input_sample[feature_index];
			mean_value /= feature_count;
			float variance_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float difference = input_sample[feature_index] - mean_value;
				variance_value += difference * difference;
			}
			variance_value /= feature_count;
			float inverse_standard_deviation = 1.0f / std::sqrt(variance_value + 1e-5f);
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				output_sample[feature_index] = (input_sample[feature_index] - mean_value) * inverse_standard_deviation;
			}
			current_offset += feature_count;
		}
	}
};
HookDerivative NonLearnableLayerNormDerivative = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	int current_offset = 0;
	for (int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
		int sequence_rows = sequence_lengths[batch_index];
		for (int row_index = 0; row_index < sequence_rows; ++row_index)
		{
			// The forward hooks normalize preactivation_values (the saved pre-activation),
			// so the derivative must recompute the statistics from the same values.
			// (original_inputs is the layer's raw input and is NOT what was normalized.)
			float *raw_input_sample = preactivation_values + current_offset;
			float *upstream_sample = upstream_gradient + current_offset;
			float *output_gradient_sample = output_gradient + current_offset;
			float mean_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				mean_value += raw_input_sample[feature_index];
			mean_value /= feature_count;
			float variance_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float difference = raw_input_sample[feature_index] - mean_value;
				variance_value += difference * difference;
			}
			variance_value /= feature_count;
			float inverse_standard_deviation = 1.0f / std::sqrt(variance_value + 1e-5f);
			float mean_upstream_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				mean_upstream_value += upstream_sample[feature_index];
			mean_upstream_value /= feature_count;
			float sum_upstream_normalized = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float normalized_input = (raw_input_sample[feature_index] - mean_value) * inverse_standard_deviation;
				sum_upstream_normalized += upstream_sample[feature_index] * normalized_input;
			}
			sum_upstream_normalized /= feature_count;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float normalized_input = (raw_input_sample[feature_index] - mean_value) * inverse_standard_deviation;
				output_gradient_sample[feature_index] = (upstream_sample[feature_index] - mean_upstream_value - normalized_input * sum_upstream_normalized) * inverse_standard_deviation;
			}
			current_offset += feature_count;
		}
	}
};
HookFunc LearnableLayerNormImpl = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	float *extra_weights = std::visit([](auto *current_layer)
									  { return current_layer->extra_weights_begin; }, layer);
	float *gamma_parameters = extra_weights;
	float *beta_parameters = gamma_parameters + feature_count;
	int current_offset = 0;
	for (int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
		int sequence_rows = sequence_lengths[batch_index];
		for (int row_index = 0; row_index < sequence_rows; ++row_index)
		{
			float *input_sample = preactivation_values + current_offset;
			float *output_sample = output_values + current_offset;
			float mean_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				mean_value += input_sample[feature_index];
			mean_value /= feature_count;
			float variance_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float difference = input_sample[feature_index] - mean_value;
				variance_value += difference * difference;
			}
			variance_value /= feature_count;
			float inverse_standard_deviation = 1.0f / std::sqrt(variance_value + 1e-5f);
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				output_sample[feature_index] = (input_sample[feature_index] - mean_value) * inverse_standard_deviation * gamma_parameters[feature_index] + beta_parameters[feature_index];
			}
			current_offset += feature_count;
		}
	}
};
HookDerivative LearnableLayerNormDerivativeImpl = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	float *extra_weights = std::visit([](auto *current_layer)
									  { return current_layer->extra_weights_begin; }, layer);
	float *extra_gradients = std::visit([](auto *current_layer)
										{ return current_layer->extra_weight_gradients; }, layer);
	float *gamma_parameters = extra_weights;
	float *beta_parameters = gamma_parameters + feature_count;
	float *gamma_gradients = extra_gradients;
	float *beta_gradients = extra_gradients + feature_count;
	int current_offset = 0;
	for (int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
		int sequence_rows = sequence_lengths[batch_index];
		for (int row_index = 0; row_index < sequence_rows; ++row_index)
		{
			// The forward hooks normalize preactivation_values (the saved pre-activation),
			// so the derivative must recompute the statistics from the same values.
			// (original_inputs is the layer's raw input and is NOT what was normalized.)
			float *raw_input_sample = preactivation_values + current_offset;
			float *upstream_sample = upstream_gradient + current_offset;
			float *output_gradient_sample = output_gradient + current_offset;
			float mean_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				mean_value += raw_input_sample[feature_index];
			mean_value /= feature_count;
			float variance_value = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float difference = raw_input_sample[feature_index] - mean_value;
				variance_value += difference * difference;
			}
			variance_value /= feature_count;
			float inverse_standard_deviation = 1.0f / std::sqrt(variance_value + 1e-5f);
			float sum_gamma_upstream = 0.0f;
			float sum_gamma_upstream_normalized = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float normalized_input = (raw_input_sample[feature_index] - mean_value) * inverse_standard_deviation;
				float gamma_upstream_product = gamma_parameters[feature_index] * upstream_sample[feature_index];
				sum_gamma_upstream += gamma_upstream_product;
				sum_gamma_upstream_normalized += gamma_upstream_product * normalized_input;
			}
			float inverse_feature_count = 1.0f / feature_count;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				float normalized_input = (raw_input_sample[feature_index] - mean_value) * inverse_standard_deviation;
				// upstream_gradient and output_gradient can alias (Layer::backward passes
				// output_buffers.first for both), so capture the true upstream value
				// before output_gradient_sample overwrites it in-place.
				float true_upstream = upstream_sample[feature_index];
				float input_gradients = (gamma_parameters[feature_index] * true_upstream - sum_gamma_upstream * inverse_feature_count - normalized_input * sum_gamma_upstream_normalized * inverse_feature_count) * inverse_standard_deviation;
				output_gradient_sample[feature_index] = input_gradients;
				gamma_gradients[feature_index] += true_upstream * normalized_input;
				beta_gradients[feature_index] += true_upstream;
			}
			current_offset += feature_count;
		}
	}
};
HookFunc RMSNormImpl = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	float *gamma_parameters = std::visit([](auto *current_layer)
										 { return current_layer->extra_weights_begin; }, layer);
	int current_offset = 0;
	for (int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
		int sequence_rows = sequence_lengths[batch_index];
		for (int row_index = 0; row_index < sequence_rows; ++row_index)
		{
			float *input_sample = preactivation_values + current_offset;
			float *output_sample = output_values + current_offset;
			float rms = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				rms += input_sample[feature_index] * input_sample[feature_index];
			rms = std::sqrt(rms / feature_count + 1e-5f);
			float inverse_rms = 1.0f / rms;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				output_sample[feature_index] = input_sample[feature_index] * inverse_rms * gamma_parameters[feature_index];
			current_offset += feature_count;
		}
	}
};
HookDerivative RMSNormDerivativeImpl = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	float *gamma_parameters = std::visit([](auto *current_layer)
										 { return current_layer->extra_weights_begin; }, layer);
	float *gamma_gradients = std::visit([](auto *current_layer)
										{ return current_layer->extra_weight_gradients; }, layer);
	int current_offset = 0;
	for (int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
		int sequence_rows = sequence_lengths[batch_index];
		for (int row_index = 0; row_index < sequence_rows; ++row_index)
		{
			// The forward hooks normalize preactivation_values (the saved pre-activation),
			// so the derivative must recompute the statistics from the same values.
			// (original_inputs is the layer's raw input and is NOT what was normalized.)
			float *raw_input_sample = preactivation_values + current_offset;
			float *upstream_sample = upstream_gradient + current_offset;
			float *output_gradient_sample = output_gradient + current_offset;
			float sum_squares = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				sum_squares += raw_input_sample[feature_index] * raw_input_sample[feature_index];
			float rms = std::sqrt(sum_squares / feature_count + 1e-5f);
			float inverse_rms = 1.0f / rms;
			float inverse_rms_cubed = inverse_rms * inverse_rms * inverse_rms;
			float sum_gamma_upstream_x = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				sum_gamma_upstream_x += gamma_parameters[feature_index] * upstream_sample[feature_index] * raw_input_sample[feature_index];
			float inverse_feature_count = 1.0f / feature_count;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				// upstream_gradient and output_gradient can alias (Layer::backward passes
				// output_buffers.first for both), so capture the true upstream value
				// before output_gradient_sample overwrites it in-place.
				float true_upstream = upstream_sample[feature_index];
				float input_gradients = gamma_parameters[feature_index] * true_upstream * inverse_rms - raw_input_sample[feature_index] * sum_gamma_upstream_x * inverse_rms_cubed * inverse_feature_count;
				output_gradient_sample[feature_index] = input_gradients;
				gamma_gradients[feature_index] += true_upstream * raw_input_sample[feature_index] * inverse_rms;
			}
			current_offset += feature_count;
		}
	}
};
HookFunc TemperatureHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	const std::vector<int> &extra_arguments = std::visit([](auto *current_layer) -> const std::vector<int> &
														 { return current_layer->extra_args; }, layer);
	float temperature_value = extra_arguments.size() > 0 ? static_cast<float>(extra_arguments[0]) / 1000.0f : 0.0f;
	if (temperature_value <= 0.0f)
	{
		int total_feature_elements = 0;
		for (int length : sequence_lengths)
			total_feature_elements += length * feature_count;
		for (int element_index = 0; element_index < total_feature_elements; ++element_index)
			output_values[element_index] = preactivation_values[element_index];
		return;
	}
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	float inverse_temperature = 1.0f / temperature_value;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
		output_values[element_index] = preactivation_values[element_index] * inverse_temperature;
};
HookDerivative TemperatureGradHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	const std::vector<int> &extra_arguments = std::visit([](auto *current_layer) -> const std::vector<int> &
														 { return current_layer->extra_args; }, layer);
	float temperature_value = extra_arguments.size() > 0 ? static_cast<float>(extra_arguments[0]) / 1000.0f : 0.0f;
	if (temperature_value <= 0.0f)
	{
		int total_feature_elements = 0;
		for (int length : sequence_lengths)
			total_feature_elements += length * feature_count;
		for (int element_index = 0; element_index < total_feature_elements; ++element_index)
			output_gradient[element_index] = upstream_gradient[element_index];
		return;
	}
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	float inverse_temperature = 1.0f / temperature_value;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
		output_gradient[element_index] = upstream_gradient[element_index] * inverse_temperature;
};
HookFunc ReLuHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		output_values[element_index] = preactivation_values[element_index] > 0.0f ? preactivation_values[element_index] : 0.0f;
	}
};
HookDerivative ReLuGradHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		output_gradient[element_index] = upstream_gradient[element_index] * (preactivation_values[element_index] > 0.0f ? 1.0f : 0.0f);
	}
};
HookFunc SigmoidHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		output_values[element_index] = 1.0f / (1.0f + std::exp(-preactivation_values[element_index]));
	}
};
HookDerivative SigmoidGradHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		float sigmoid_activation = 1.0f / (1.0f + std::exp(-preactivation_values[element_index]));
		output_gradient[element_index] = upstream_gradient[element_index] * sigmoid_activation * (1.0f - sigmoid_activation);
	}
};
HookFunc TanhHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		output_values[element_index] = std::tanh(preactivation_values[element_index]);
	}
};
HookDerivative TanhGradHook = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	int total_feature_elements = 0;
	for (int length : sequence_lengths)
		total_feature_elements += length * feature_count;
	for (int element_index = 0; element_index < total_feature_elements; ++element_index)
	{
		float tanh_activation = std::tanh(preactivation_values[element_index]);
		output_gradient[element_index] = upstream_gradient[element_index] * (1.0f - tanh_activation * tanh_activation);
	}
};
HookFunc Softmax = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	int current_offset = 0;
	for (int batch_index = 0; batch_index < batch_count; ++batch_index)
	{
		int sequence_rows = sequence_lengths[batch_index];
		for (int row_index = 0; row_index < sequence_rows; ++row_index)
		{
			float *input_sample = preactivation_values + current_offset;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			{
				if (!std::isfinite(input_sample[feature_index]))
				{
					std::cout << "SOFTMAX GOT NON-FINITE VALUE: "
							  << input_sample[feature_index]
							  << " at feature "
							  << feature_index
							  << "\n";
					return;
				}
			}
			float *output_sample = output_values + current_offset;
			float maximum_value = input_sample[0];
			for (int feature_index = 1; feature_index < feature_count; ++feature_index)
			{
				if (input_sample[feature_index] > maximum_value)
					maximum_value = input_sample[feature_index];
			}
			float exponential_sum = 0.0f;
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				exponential_sum += std::exp(input_sample[feature_index] - maximum_value);
			for (int feature_index = 0; feature_index < feature_count; ++feature_index)
				output_sample[feature_index] = std::exp(input_sample[feature_index] - maximum_value) / exponential_sum;
			current_offset += feature_count;
		}
	}
};
HookDerivative SoftmaxDerivative = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	int total_rows_count = 0;
	for (int length : sequence_lengths)
		total_rows_count += length;
	std::vector<float> softmax_values_buffer(total_rows_count * feature_count);
	for (int row_index = 0; row_index < total_rows_count; ++row_index)
	{
		float *input_sample = preactivation_values + row_index * feature_count;
		float *output_sample = softmax_values_buffer.data() + row_index * feature_count;
		float maximum_value = input_sample[0];
		for (int feature_index = 1; feature_index < feature_count; ++feature_index)
		{
			if (input_sample[feature_index] > maximum_value)
				maximum_value = input_sample[feature_index];
		}
		float exponential_sum = 0.0f;
		for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			exponential_sum += std::exp(input_sample[feature_index] - maximum_value);
		for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			output_sample[feature_index] = std::exp(input_sample[feature_index] - maximum_value) / exponential_sum;
	}
	for (int row_index = 0; row_index < total_rows_count; ++row_index)
	{
		int correct_token = correct_indices[row_index];
		float *softmax_sample = softmax_values_buffer.data() + row_index * feature_count;
		float *output_gradient_sample = output_gradient + row_index * feature_count;
		float *upstream_sample = upstream_gradient + row_index * feature_count;
		float gradient_at_correct = upstream_sample[correct_token];
		for (int feature_index = 0; feature_index < feature_count; ++feature_index)
		{
			output_gradient_sample[feature_index] = upstream_sample[feature_index] * softmax_sample[feature_index] - softmax_sample[feature_index] * softmax_sample[correct_token] * gradient_at_correct;
		}
	}
};
HookDerivative SoftmaxForCrossEntropyLossDerivative = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	int total_rows_count = 0;
	for (int length : sequence_lengths)
		total_rows_count += length;
	for (int row_index = 0; row_index < total_rows_count; ++row_index)
	{
		int correct_token = correct_indices[row_index];
		float *logit_sample = preactivation_values + row_index * feature_count;
		float *output_gradient_sample = output_gradient + row_index * feature_count;
		float maximum_value = logit_sample[0];
		for (int feature_index = 1; feature_index < feature_count; ++feature_index)
		{
			if (logit_sample[feature_index] > maximum_value)
				maximum_value = logit_sample[feature_index];
		}
		float exponential_sum = 0.0f;
		for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			exponential_sum += std::exp(logit_sample[feature_index] - maximum_value);
		for (int feature_index = 0; feature_index < feature_count; ++feature_index)
			output_gradient_sample[feature_index] = std::exp(logit_sample[feature_index] - maximum_value) / exponential_sum;
		output_gradient_sample[correct_token] -= 1.0f;
	}
};
HookFunc EmbeddingForward = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	ParametricLayer *parametric_layer = std::get<ParametricLayer *>(layer);
	std::cout << "DEBUG: EmbeddingForward reached\n";
	int embedding_dimension = parametric_layer->output;
	int total_rows_count = 0;
	for (int length : sequence_lengths)
		total_rows_count += length;
	for (int row_index = 0; row_index < total_rows_count; ++row_index)
	{
		int token_index = static_cast<int>(original_inputs[row_index]);
		float *weight_row = parametric_layer->weights_begin + token_index * embedding_dimension;
		float *output_row = output_values + row_index * embedding_dimension;
		for (int dimension_index = 0; dimension_index < embedding_dimension; ++dimension_index)
			output_row[dimension_index] = weight_row[dimension_index];
	}
};
HookDerivative EmbeddingDerivative = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	ParametricLayer *parametric_layer = std::get<ParametricLayer *>(layer);
	int embedding_dimension = parametric_layer->output;
	float *weight_gradients_pointer = parametric_layer->weight_gradients;
	int total_rows_count = 0;
	for (int length : sequence_lengths)
		total_rows_count += length;
	for (int row_index = 0; row_index < total_rows_count; ++row_index)
	{
		int token_index = static_cast<int>(original_inputs[row_index]);
		float *gradient_row = weight_gradients_pointer + token_index * embedding_dimension;
		float *upstream_row = upstream_gradient + row_index * embedding_dimension;
		for (int dimension_index = 0; dimension_index < embedding_dimension; ++dimension_index)
			gradient_row[dimension_index] += upstream_row[dimension_index];
	}
	for (int element_index = 0; element_index < total_rows_count * parametric_layer->input; ++element_index)
		output_gradient[element_index] = 0.0f;
};
HookFunc AttentionForward = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	ParametricLayer *parametric_layer = std::get<ParametricLayer *>(layer);
	std::cout << "DEBUG: AttentionForward reached\n";
	int embedding_dimension = feature_count;
	int maximum_sequence_length = parametric_layer->extra_args[0];
	float *weight_matrix = parametric_layer->weights_begin;
	float *query_weights = weight_matrix;
	float *key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float *value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float *query_buffer = parametric_layer->scratch_pointer;
	float *key_buffer = query_buffer + maximum_sequence_length * embedding_dimension;
	float *value_buffer = key_buffer + maximum_sequence_length * embedding_dimension;
	float *attention_scores = value_buffer + maximum_sequence_length * embedding_dimension;
	float *stored_lengths = attention_scores + maximum_sequence_length * maximum_sequence_length;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	// The rest of the network stores sequences COMPACTED (packed back-to-back by
	// their ACTUAL lengths), so attention must process exactly actual_length rows
	// per sequence and advance offsets by actual_length * embedding_dimension.
	int current_offset = 0;
	for (int sequence_index = 0; sequence_index < batch_count; ++sequence_index)
	{
		int actual_length = sequence_lengths[sequence_index];
		stored_lengths[sequence_index] = static_cast<float>(actual_length);
		float *sequence_input = preactivation_values + current_offset;
		float *sequence_output = output_values + current_offset;
		matmult(sequence_input, query_weights, query_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(sequence_input, key_weights, key_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(sequence_input, value_weights, value_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(query_buffer, key_buffer, attention_scores, actual_length, embedding_dimension, actual_length, false, true, inverse_sqrt_dimension, 0.0f);
		// Causal mask: token i may attend only to columns <= i.
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			for (int col_index = row_index + 1; col_index < actual_length; ++col_index)
			{
				attention_scores[row_index * actual_length + col_index] = -1e30f;
			}
		}
		// Compacted storage has no padded positions, so no extra sequence-length
		// mask is needed beyond the causal mask: the padded rows/columns that the
		// previous implementation masked to -1e30 no longer exist in the buffer.
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			float row_maximum = attention_scores[row_index * actual_length];
			for (int col_index = 1; col_index < actual_length; ++col_index)
			{
				if (attention_scores[row_index * actual_length + col_index] > row_maximum)
					row_maximum = attention_scores[row_index * actual_length + col_index];
			}
			float row_exponential_sum = 0.0f;
			for (int col_index = 0; col_index < actual_length; ++col_index)
			{
				row_exponential_sum += std::exp(attention_scores[row_index * actual_length + col_index] - row_maximum);
			}
			for (int col_index = 0; col_index < actual_length; ++col_index)
			{
				attention_scores[row_index * actual_length + col_index] = std::exp(attention_scores[row_index * actual_length + col_index] - row_maximum) / row_exponential_sum;
			}
		}
		matmult(attention_scores, value_buffer, sequence_output, actual_length, actual_length, embedding_dimension, false, false, 1.0f, 0.0f);
		current_offset += actual_length * embedding_dimension;
		sequence_lengths[sequence_index] = actual_length;
	}
};
HookDerivative AttentionDerivative = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices)
{
	ParametricLayer *parametric_layer = std::get<ParametricLayer *>(layer);
	int embedding_dimension = feature_count;
	int maximum_sequence_length = parametric_layer->extra_args[0];
	float *input_embeddings = original_inputs;
	float *weight_matrix = parametric_layer->weights_begin;
	float *query_weights = weight_matrix;
	float *key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float *value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float *scratchpad_pointer = parametric_layer->scratch_pointer;
	float *query_buffer = scratchpad_pointer;
	float *key_buffer = query_buffer + maximum_sequence_length * embedding_dimension;
	float *temporary_buffer = key_buffer + maximum_sequence_length * embedding_dimension;
	float *attention_scores = temporary_buffer + maximum_sequence_length * embedding_dimension;
	// stored_lengths must live at the SAME offset AttentionForward writes it
	// (attention_scores + max_seq*max_seq), so the derivative can read back the
	// original sequence lengths that the forward pass stored.
	float *stored_lengths = attention_scores + maximum_sequence_length * maximum_sequence_length;
	float *score_gradients = stored_lengths + batch_count;
	float *full_upstream_sample = score_gradients + maximum_sequence_length * maximum_sequence_length;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	// Same compacted layout as AttentionForward (and the rest of the network):
	// sequences are packed by their ACTUAL lengths, so the input, the output
	// gradient and the upstream gradient all advance by actual_length * emb.
	int current_offset = 0;
	int upstream_offset = 0;
	for (int sequence_index = 0; sequence_index < batch_count; ++sequence_index)
	{
		int original_length = static_cast<int>(stored_lengths[sequence_index]);
		int actual_length = sequence_lengths[sequence_index];
		float *sequence_input = input_embeddings + current_offset;
		float *sequence_upstream_reduced = upstream_gradient + upstream_offset;
		float *sequence_output_gradient = output_gradient + current_offset;
		std::fill(full_upstream_sample, full_upstream_sample + maximum_sequence_length * embedding_dimension, 0.0f);
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			for (int col_index = 0; col_index < embedding_dimension; ++col_index)
			{
				full_upstream_sample[row_index * embedding_dimension + col_index] = sequence_upstream_reduced[row_index * embedding_dimension + col_index];
			}
		}
		matmult(sequence_input, query_weights, query_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(sequence_input, key_weights, key_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(sequence_input, value_weights, temporary_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(query_buffer, key_buffer, attention_scores, actual_length, embedding_dimension, actual_length, false, true, inverse_sqrt_dimension, 0.0f);
		// Causal mask: token i may attend only to columns <= i.
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			for (int col_index = row_index + 1; col_index < actual_length; ++col_index)
			{
				attention_scores[row_index * actual_length + col_index] = -1e30f;
			}
		}
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			float row_maximum = attention_scores[row_index * actual_length];
			for (int col_index = 1; col_index < actual_length; ++col_index)
			{
				if (attention_scores[row_index * actual_length + col_index] > row_maximum)
					row_maximum = attention_scores[row_index * actual_length + col_index];
			}
			float row_exponential_sum = 0.0f;
			for (int col_index = 0; col_index < actual_length; ++col_index)
			{
				row_exponential_sum += std::exp(attention_scores[row_index * actual_length + col_index] - row_maximum);
			}
			for (int col_index = 0; col_index < actual_length; ++col_index)
			{
				attention_scores[row_index * actual_length + col_index] = std::exp(attention_scores[row_index * actual_length + col_index] - row_maximum) / row_exponential_sum;
			}
		}
		matmult(full_upstream_sample, temporary_buffer, score_gradients, actual_length, embedding_dimension, actual_length, false, true, 1.0f, 0.0f);
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			float weighted_sum = 0.0f;
			for (int col_index = 0; col_index < actual_length; ++col_index)
				weighted_sum += score_gradients[row_index * actual_length + col_index] * attention_scores[row_index * actual_length + col_index];
			for (int col_index = 0; col_index < actual_length; ++col_index)
				score_gradients[row_index * actual_length + col_index] = attention_scores[row_index * actual_length + col_index] * (score_gradients[row_index * actual_length + col_index] - weighted_sum);
		}
		float *weight_gradients_pointer = parametric_layer->weight_gradients;
		// Q_grad, K_grad and V_grad must all survive until the input-gradient sum
		// dL/dx = Q_grad*Wq^T + K_grad*Wk^T + V_grad*Wv^T is formed. They share the
		// single temporary_buffer, so each gradient is computed and FULLY consumed
		// (input-gradient term plus weight gradient) before the buffer is reused for
		// the next one. That keeps the three gradients distinct with no extra scratch.
		// 1) V branch
		float *value_gradient = temporary_buffer;
		matmult(attention_scores, full_upstream_sample, value_gradient, actual_length, actual_length, embedding_dimension, true, false, 1.0f, 0.0f);
		matmult(value_gradient, value_weights, sequence_output_gradient, actual_length, embedding_dimension, embedding_dimension, false, true, 1.0f, 0.0f);
		matmult(sequence_input, value_gradient, weight_gradients_pointer + 2 * embedding_dimension * embedding_dimension, embedding_dimension, actual_length, embedding_dimension, true, false, 1.0f, 1.0f);
		// 2) Q branch
		float *query_gradient = temporary_buffer;
		matmult(score_gradients, key_buffer, query_gradient, actual_length, actual_length, embedding_dimension, false, false, inverse_sqrt_dimension, 0.0f);
		matmult(query_gradient, query_weights, sequence_output_gradient, actual_length, embedding_dimension, embedding_dimension, false, true, 1.0f, 1.0f);
		matmult(sequence_input, query_gradient, weight_gradients_pointer, embedding_dimension, actual_length, embedding_dimension, true, false, 1.0f, 1.0f);
		// 3) K branch
		float *key_gradient = temporary_buffer;
		matmult(score_gradients, query_buffer, key_gradient, actual_length, actual_length, embedding_dimension, true, false, inverse_sqrt_dimension, 0.0f);
		matmult(key_gradient, key_weights, sequence_output_gradient, actual_length, embedding_dimension, embedding_dimension, false, true, 1.0f, 1.0f);
		matmult(sequence_input, key_gradient, weight_gradients_pointer + embedding_dimension * embedding_dimension, embedding_dimension, actual_length, embedding_dimension, true, false, 1.0f, 1.0f);
		sequence_lengths[sequence_index] = original_length;
		current_offset += actual_length * embedding_dimension;
		upstream_offset += actual_length * embedding_dimension;
	}
};
#ifndef TRAINING_ON
HookFunc AttentionForwardWithCache = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *output_values, int feature_count)
{
	ParametricLayer *parametric_layer = std::get<ParametricLayer *>(layer);
	int embedding_dimension = feature_count;
	int maximum_sequence_length = parametric_layer->extra_args[0];
	int cached_pool_index = parametric_layer->extra_args[1];
	float *weight_matrix = parametric_layer->weights_begin;
	float *query_weights = weight_matrix;
	float *key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float *value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float *query_buffer = parametric_layer->scratch_pointer;
	float *key_buffer = query_buffer + maximum_sequence_length * embedding_dimension;
	float *value_buffer = key_buffer + maximum_sequence_length * embedding_dimension;
	float *attention_scores = value_buffer + maximum_sequence_length * embedding_dimension;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	std::vector<float> &cache = kv_cache_pool[cached_pool_index];
	int current_offset = 0;
	for (int sequence_index = 0; sequence_index < batch_count; ++sequence_index)
	{
		int actual_length = sequence_lengths[sequence_index];
		float *sequence_input = preactivation_values + current_offset;
		float *sequence_output = output_values + current_offset;
		matmult(sequence_input, query_weights, query_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(sequence_input, key_weights, key_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(sequence_input, value_weights, value_buffer, actual_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		int cached_length = cache.size() / (2 * embedding_dimension);
		int total_length = cached_length + actual_length;
		cache.resize(total_length * 2 * embedding_dimension);
		std::copy(key_buffer, key_buffer + actual_length * embedding_dimension, cache.begin() + cached_length * embedding_dimension);
		std::copy(value_buffer, value_buffer + actual_length * embedding_dimension, cache.begin() + total_length * embedding_dimension + cached_length * embedding_dimension);
		float *all_keys = cache.data();
		float *all_values = cache.data() + total_length * embedding_dimension;
		matmult(query_buffer, all_keys, attention_scores, actual_length, embedding_dimension, total_length, false, true, inverse_sqrt_dimension, 0.0f);
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			for (int col_index = cached_length + row_index + 1; col_index < total_length; ++col_index)
			{
				attention_scores[row_index * total_length + col_index] = -1e30f;
			}
		}
		for (int row_index = 0; row_index < actual_length; ++row_index)
		{
			float row_maximum = attention_scores[row_index * total_length];
			for (int col_index = 1; col_index < total_length; ++col_index)
			{
				if (attention_scores[row_index * total_length + col_index] > row_maximum)
					row_maximum = attention_scores[row_index * total_length + col_index];
			}
			float row_exponential_sum = 0.0f;
			for (int col_index = 0; col_index < total_length; ++col_index)
			{
				row_exponential_sum += std::exp(attention_scores[row_index * total_length + col_index] - row_maximum);
			}
			for (int col_index = 0; col_index < total_length; ++col_index)
			{
				attention_scores[row_index * total_length + col_index] = std::exp(attention_scores[row_index * total_length + col_index] - row_maximum) / row_exponential_sum;
			}
		}
		matmult(attention_scores, all_values, sequence_output, actual_length, total_length, embedding_dimension, false, false, 1.0f, 0.0f);
		current_offset += maximum_sequence_length * embedding_dimension;
	}
};
HookDerivative AttentionDerivativeWithCache = [](LayerRef layer, int batch_count, std::vector<int> &sequence_lengths, float *original_inputs, float *preactivation_values, float *upstream_gradient, float *output_gradient, int feature_count, const std::vector<int> &correct_indices) {};
#endif
inline HookFunc LearnableLayerNorm(LayerArgs &layer_arguments)
{
	layer_arguments.extra_weights += layer_arguments.layer_size * 2;
	return LearnableLayerNormImpl;
}
inline HookDerivative LearnableLayerNormDerivative(LayerArgs &layer_arguments)
{
	return LearnableLayerNormDerivativeImpl;
}
inline HookFunc RMSNorm(LayerArgs &layer_arguments)
{
	layer_arguments.extra_weights += layer_arguments.layer_size;
	return RMSNormImpl;
}
inline HookDerivative RMSNormDerivative(LayerArgs &layer_arguments)
{
	return RMSNormDerivativeImpl;
}
#endif

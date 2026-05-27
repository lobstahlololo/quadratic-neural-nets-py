#ifndef ACTIVATIONS_IMPORTED
#define ACTIVATIONS_IMPORTED
#include "../model/network.h"
#include <cmath>
#include <algorithm>
#include <variant>
#include <vector>
#ifndef TRAINING_ON
std::vector<std::vector<float>> kv_cache_pool;
#endif
HookFunc Residual = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = feature_count;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = preactivation_values[i] + original_inputs[i];
	}
};
HookDerivative ResidualGradHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = feature_count;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		output_gradient[i] = upstream_gradient[i];
	}
};
HookFunc NonLearnableLayerNorm = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int offset = 0;
	for (int b = 0; b < batch_count; ++b) {
		int rows = batch_sizes[b];
		for (int r = 0; r < rows; ++r) {
			float* input_sample = preactivation_values + offset;
			float* output_sample = output_values + offset;
			float mean = 0.0f;
			for (int i = 0; i < feature_count; ++i) mean += input_sample[i];
			mean /= feature_count;
			float variance = 0.0f;
			for (int i = 0; i < feature_count; ++i) variance += (input_sample[i] - mean) * (input_sample[i] - mean);
			variance /= feature_count;
			float inverse_std_dev = 1.0f / std::sqrt(variance + 1e-5f);
			for (int i = 0; i < feature_count; ++i) output_sample[i] = (input_sample[i] - mean) * inverse_std_dev;
			offset += feature_count;
		}
	}
};
HookDerivative NonLearnableLayerNormDerivative = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int offset = 0;
	for (int b = 0; b < batch_count; ++b) {
		int rows = batch_sizes[b];
		for (int r = 0; r < rows; ++r) {
			float* raw_input = original_inputs + offset;
			float* upstream = upstream_gradient + offset;
			float* out_grad = output_gradient + offset;
			float mean = 0.0f;
			for (int i = 0; i < feature_count; ++i) mean += raw_input[i];
			mean /= feature_count;
			float var = 0.0f;
			for (int i = 0; i < feature_count; ++i) { float diff = raw_input[i] - mean; var += diff * diff; }
			var /= feature_count;
			float inverse_std = 1.0f / std::sqrt(var + 1e-5f);
			float mean_up = 0.0f;
			for (int i = 0; i < feature_count; ++i) mean_up += upstream[i];
			mean_up /= feature_count;
			float sum_up_xhat = 0.0f;
			for (int i = 0; i < feature_count; ++i) { float xhat = (raw_input[i] - mean) * inverse_std; sum_up_xhat += upstream[i] * xhat; }
			sum_up_xhat /= feature_count;
			for (int i = 0; i < feature_count; ++i) {
				float xhat = (raw_input[i] - mean) * inverse_std;
				out_grad[i] = (upstream[i] - mean_up - xhat * sum_up_xhat) * inverse_std;
			}
			offset += feature_count;
		}
	}
};
HookFunc LearnableLayerNorm = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	float* gamma = parametric_layer->weights_begin;
	float* beta = gamma + parametric_layer->input;
	int offset = 0;
	for (int b = 0; b < batch_count; ++b) {
		int rows = batch_sizes[b];
		for (int r = 0; r < rows; ++r) {
			float* input_sample = preactivation_values + offset;
			float* output_sample = output_values + offset;
			float mean = 0.0f;
			for (int i = 0; i < feature_count; ++i) mean += input_sample[i];
			mean /= feature_count;
			float variance = 0.0f;
			for (int i = 0; i < feature_count; ++i) variance += (input_sample[i] - mean) * (input_sample[i] - mean);
			variance /= feature_count;
			float inverse_std_dev = 1.0f / std::sqrt(variance + 1e-5f);
			for (int i = 0; i < feature_count; ++i) output_sample[i] = (input_sample[i] - mean) * inverse_std_dev * gamma[i] + beta[i];
			offset += feature_count;
		}
	}
};
HookDerivative LearnableLayerNormDerivative = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	float* gamma = parametric_layer->weights_begin;
	float* beta = gamma + feature_count;
	float* weight_gradients_ptr = parametric_layer->weight_gradients;
	float* gamma_gradient = weight_gradients_ptr;
	float* beta_gradient = weight_gradients_ptr + feature_count;
	int offset = 0;
	for (int b = 0; b < batch_count; ++b) {
		int rows = batch_sizes[b];
		for (int r = 0; r < rows; ++r) {
			float* raw_input = original_inputs + offset;
			float* upstream = upstream_gradient + offset;
			float* out_grad = output_gradient + offset;
			float mean = 0.0f;
			for (int i = 0; i < feature_count; ++i) mean += raw_input[i];
			mean /= feature_count;
			float var = 0.0f;
			for (int i = 0; i < feature_count; ++i) { float diff = raw_input[i] - mean; var += diff * diff; }
			var /= feature_count;
			float inverse_std = 1.0f / std::sqrt(var + 1e-5f);
			float sum_gamma_up = 0.0f;
			float sum_gamma_up_xhat = 0.0f;
			for (int i = 0; i < feature_count; ++i) {
				float x_hat = (raw_input[i] - mean) * inverse_std;
				float g_up = gamma[i] * upstream[i];
				sum_gamma_up += g_up;
				sum_gamma_up_xhat += g_up * x_hat;
			}
			float inv_N = 1.0f / feature_count;
			for (int i = 0; i < feature_count; ++i) {
				float x_hat = (raw_input[i] - mean) * inverse_std;
				float dx = (gamma[i] * upstream[i] - sum_gamma_up * inv_N - x_hat * sum_gamma_up_xhat * inv_N) * inverse_std;
				out_grad[i] = dx;
				gamma_gradient[i] += upstream[i] * x_hat;
				beta_gradient[i] += upstream[i];
			}
			offset += feature_count;
		}
	}
};
HookFunc RMSNorm = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	float* gamma = parametric_layer->weights_begin;
	int offset = 0;
	for (int b = 0; b < batch_count; ++b) {
		int rows = batch_sizes[b];
		for (int r = 0; r < rows; ++r) {
			float* input_sample = preactivation_values + offset;
			float* output_sample = output_values + offset;
			float rms = 0.0f;
			for (int i = 0; i < feature_count; ++i) rms += input_sample[i] * input_sample[i];
			rms = std::sqrt(rms / feature_count + 1e-5f);
			float inverse_rms = 1.0f / rms;
			for (int i = 0; i < feature_count; ++i) output_sample[i] = input_sample[i] * inverse_rms * gamma[i];
			offset += feature_count;
		}
	}
};
HookDerivative RMSNormDerivative = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	float* gamma = parametric_layer->weights_begin;
	float* gamma_gradient = parametric_layer->weight_gradients;
	int offset = 0;
	for (int b = 0; b < batch_count; ++b) {
		int rows = batch_sizes[b];
		for (int r = 0; r < rows; ++r) {
			float* raw_input = original_inputs + offset;
			float* upstream = upstream_gradient + offset;
			float* out_grad = output_gradient + offset;
			float sum_sq = 0.0f;
			for (int i = 0; i < feature_count; ++i) sum_sq += raw_input[i] * raw_input[i];
			float rms = std::sqrt(sum_sq / feature_count + 1e-5f);
			float inverse_rms = 1.0f / rms;
			float inverse_rms_cubed = inverse_rms * inverse_rms * inverse_rms;
			float sum_gamma_up_x = 0.0f;
			for (int i = 0; i < feature_count; ++i) sum_gamma_up_x += gamma[i] * upstream[i] * raw_input[i];
			float inv_N = 1.0f / feature_count;
			for (int i = 0; i < feature_count; ++i) {
				float dx = gamma[i] * upstream[i] * inverse_rms - raw_input[i] * sum_gamma_up_x * inverse_rms_cubed * inv_N;
				out_grad[i] = dx;
				gamma_gradient[i] += upstream[i] * raw_input[i] * inverse_rms;
			}
			offset += feature_count;
		}
	}
};
HookFunc TemperatureHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	const std::vector<int>& extra_args = std::visit([](auto* l) -> const std::vector<int>& { return l->extra_args; }, layer);
	float temperature = extra_args.size() > 0 ? static_cast<float>(extra_args[0]) / 1000.0f : 0.0f;
	if (temperature <= 0.0f) {
		int total = 0;
		for (int s : batch_sizes) total += s * feature_count;
		for (int i = 0; i < total; ++i) output_values[i] = preactivation_values[i];
		return;
	}
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	float inv_temp = 1.0f / temperature;
	for (int i = 0; i < total; ++i) output_values[i] = preactivation_values[i] * inv_temp;
};
HookDerivative TemperatureGradHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	const std::vector<int>& extra_args = std::visit([](auto* l) -> const std::vector<int>& { return l->extra_args; }, layer);
	float temperature = extra_args.size() > 0 ? static_cast<float>(extra_args[0]) / 1000.0f : 0.0f;
	if (temperature <= 0.0f) {
		int total = 0;
		for (int s : batch_sizes) total += s * feature_count;
		for (int i = 0; i < total; ++i) output_gradient[i] = upstream_gradient[i];
		return;
	}
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	float inv_temp = 1.0f / temperature;
	for (int i = 0; i < total; ++i) output_gradient[i] = upstream_gradient[i] * inv_temp;
};
HookFunc ReLuHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = preactivation_values[i] > 0.0f ? preactivation_values[i] : 0.0f;
	}
};
HookDerivative ReLuGradHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		output_gradient[i] = upstream_gradient[i] * (preactivation_values[i] > 0.0f ? 1.0f : 0.0f);
	}
};
HookFunc SigmoidHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = 1.0f / (1.0f + std::exp(-preactivation_values[i]));
	}
};
HookDerivative SigmoidGradHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		float sig = 1.0f / (1.0f + std::exp(-preactivation_values[i]));
		output_gradient[i] = upstream_gradient[i] * sig * (1.0f - sig);
	}
};
HookFunc TanhHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = std::tanh(preactivation_values[i]);
	}
};
HookDerivative TanhGradHook = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = 0;
	for (int s : batch_sizes) total += s * feature_count;
	for (int i = 0; i < total; ++i) {
		float t = std::tanh(preactivation_values[i]);
		output_gradient[i] = upstream_gradient[i] * (1.0f - t * t);
	}
};
HookFunc Softmax = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int offset = 0;
	for (int b = 0; b < batch_count; ++b) {
		int rows = batch_sizes[b];
		for (int r = 0; r < rows; ++r) {
			float* input_sample = preactivation_values + offset;
			float* output_sample = output_values + offset;
			float maximum = input_sample[0];
			for (int i = 1; i < feature_count; ++i) if (input_sample[i] > maximum) maximum = input_sample[i];
			float sum = 0.0f;
			for (int i = 0; i < feature_count; ++i) sum += std::exp(input_sample[i] - maximum);
			for (int i = 0; i < feature_count; ++i) output_sample[i] = std::exp(input_sample[i] - maximum) / sum;
			offset += feature_count;
		}
	}
};
HookDerivative SoftmaxDerivative = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	std::vector<float> softmax_values(total_rows * feature_count);
	for (int i = 0; i < total_rows; ++i) {
		float* inp = preactivation_values + i * feature_count;
		float* out = softmax_values.data() + i * feature_count;
		float maximum = inp[0];
		for (int j = 1; j < feature_count; ++j) if (inp[j] > maximum) maximum = inp[j];
		float sum = 0.0f;
		for (int j = 0; j < feature_count; ++j) sum += std::exp(inp[j] - maximum);
		for (int j = 0; j < feature_count; ++j) out[j] = std::exp(inp[j] - maximum) / sum;
	}
	for (int i = 0; i < total_rows; ++i) {
		int correct_token = correct_indices[i];
		float* soft = softmax_values.data() + i * feature_count;
		float* out = output_gradient + i * feature_count;
		float* up = upstream_gradient + i * feature_count;
		float grad_at_correct = up[correct_token];
		for (int j = 0; j < feature_count; ++j) out[j] = up[j] * soft[j] - soft[j] * soft[correct_token] * grad_at_correct;
	}
};
HookDerivative SoftmaxForCrossEntropyLossDerivative = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	for (int i = 0; i < total_rows; ++i) {
		int correct_token = correct_indices[i];
		float* logit_sample = preactivation_values + i * feature_count;
		float* out_sample = output_gradient + i * feature_count;
		float maximum = logit_sample[0];
		for (int j = 1; j < feature_count; ++j) if (logit_sample[j] > maximum) maximum = logit_sample[j];
		float sum_exp = 0.0f;
		for (int j = 0; j < feature_count; ++j) sum_exp += std::exp(logit_sample[j] - maximum);
		for (int j = 0; j < feature_count; ++j) out_sample[j] = std::exp(logit_sample[j] - maximum) / sum_exp;
		out_sample[correct_token] -= 1.0f;
	}
};
HookFunc EmbeddingForward = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	for (int i = 0; i < total_rows; ++i) {
		int token_index = static_cast<int>(original_inputs[i]);
		float* weight_row = parametric_layer->weights_begin + token_index * embedding_dimension;
		float* output_row = output_values + i * embedding_dimension;
		for (int j = 0; j < embedding_dimension; ++j) output_row[j] = weight_row[j];
	}
};
HookDerivative EmbeddingDerivative = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	float* weight_gradients_ptr = parametric_layer->weight_gradients;
	int total_rows = 0;
	for (int s : batch_sizes) total_rows += s;
	for (int i = 0; i < total_rows; ++i) {
		int token_index = static_cast<int>(original_inputs[i]);
		float* gradient_row = weight_gradients_ptr + token_index * embedding_dimension;
		float* upstream_row = upstream_gradient + i * embedding_dimension;
		for (int j = 0; j < embedding_dimension; ++j) gradient_row[j] += upstream_row[j];
	}
	for (int i = 0; i < total_rows * embedding_dimension; ++i) output_gradient[i] = 0.0f;
};
HookFunc AttentionForward = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	int max_seq_len = parametric_layer->extra_args[0];
	float* weight_matrix = parametric_layer->weights_begin;
	float* query_weights = weight_matrix;
	float* key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float* value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float* query = parametric_layer->scratch_pointer;
	float* key = query + max_seq_len * embedding_dimension;
	float* value = key + max_seq_len * embedding_dimension;
	float* attention_scores = value + max_seq_len * embedding_dimension;
	float* stored_lengths = attention_scores + max_seq_len * max_seq_len;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	int offset = 0;
	for (int seq = 0; seq < batch_count; ++seq) {
		int actual_len = batch_sizes[seq];
		stored_lengths[seq] = static_cast<float>(actual_len);
		float* seq_input = preactivation_values + offset;
		float* seq_output = output_values + offset;
		matmult(seq_input, query_weights, query, max_seq_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, key_weights, key, max_seq_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, value_weights, value, max_seq_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(query, key, attention_scores, max_seq_len, embedding_dimension, max_seq_len, false, true, inverse_sqrt_dimension, 0.0f);
		for (int row = 0; row < max_seq_len; ++row) {
			for (int col = row + 1; col < max_seq_len; ++col) {
				attention_scores[row * max_seq_len + col] = -1e30f;
			}
		}
		if (actual_len < max_seq_len) {
			for (int row = 0; row < max_seq_len; ++row) {
				for (int col = actual_len; col < max_seq_len; ++col) {
					attention_scores[row * max_seq_len + col] = -1e30f;
				}
			}
			for (int row = actual_len; row < max_seq_len; ++row) {
				for (int col = 0; col < max_seq_len; ++col) {
					attention_scores[row * max_seq_len + col] = -1e30f;
				}
			}
		}
		for (int row = 0; row < max_seq_len; ++row) {
			float row_max = attention_scores[row * max_seq_len];
			for (int col = 1; col < max_seq_len; ++col) {
				if (attention_scores[row * max_seq_len + col] > row_max) row_max = attention_scores[row * max_seq_len + col];
			}
			float row_sum = 0.0f;
			for (int col = 0; col < max_seq_len; ++col) {
				row_sum += std::exp(attention_scores[row * max_seq_len + col] - row_max);
			}
			for (int col = 0; col < max_seq_len; ++col) {
				attention_scores[row * max_seq_len + col] = std::exp(attention_scores[row * max_seq_len + col] - row_max) / row_sum;
			}
		}
		matmult(attention_scores, value, seq_output, max_seq_len, max_seq_len, embedding_dimension, false, false, 1.0f, 0.0f);
		offset += max_seq_len * embedding_dimension;
		batch_sizes[seq] = actual_len;
	}
};
HookDerivative AttentionDerivative = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	int max_seq_len = parametric_layer->extra_args[0];
	float* input_embeddings = original_inputs;
	float* weight_matrix = parametric_layer->weights_begin;
	float* query_weights = weight_matrix;
	float* key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float* value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float* scratch = parametric_layer->scratch_pointer;
	float* query = scratch;
	float* key = query + max_seq_len * embedding_dimension;
	float* temporary_buffer = key + max_seq_len * embedding_dimension;
	float* attention_scores = temporary_buffer + max_seq_len * embedding_dimension;
	float* score_gradients = attention_scores + max_seq_len * max_seq_len;
	float* stored_lengths = score_gradients + max_seq_len * max_seq_len;
	float* full_upstream = stored_lengths + batch_count;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	int offset = 0;
	int upstream_offset = 0;
	for (int seq = 0; seq < batch_count; ++seq) {
		int original_len = static_cast<int>(stored_lengths[seq]);
		int actual_len = batch_sizes[seq];
		float* seq_input = input_embeddings + offset;
		float* seq_upstream_reduced = upstream_gradient + upstream_offset;
		float* seq_output_grad = output_gradient + offset;
		std::fill(full_upstream, full_upstream + max_seq_len * embedding_dimension, 0.0f);
		for (int row = 0; row < actual_len; ++row) {
			for (int col = 0; col < embedding_dimension; ++col) {
				full_upstream[row * embedding_dimension + col] = seq_upstream_reduced[row * embedding_dimension + col];
			}
		}
		matmult(seq_input, query_weights, query, max_seq_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, key_weights, key, max_seq_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, value_weights, temporary_buffer, max_seq_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(query, key, attention_scores, max_seq_len, embedding_dimension, max_seq_len, false, true, inverse_sqrt_dimension, 0.0f);
		for (int row = 0; row < max_seq_len; ++row) {
			for (int col = row + 1; col < max_seq_len; ++col) {
				attention_scores[row * max_seq_len + col] = -1e30f;
			}
		}
		if (actual_len < max_seq_len) {
			for (int row = 0; row < max_seq_len; ++row) {
				for (int col = actual_len; col < max_seq_len; ++col) {
					attention_scores[row * max_seq_len + col] = -1e30f;
				}
			}
			for (int row = actual_len; row < max_seq_len; ++row) {
				for (int col = 0; col < max_seq_len; ++col) {
					attention_scores[row * max_seq_len + col] = -1e30f;
				}
			}
		}
		for (int row = 0; row < max_seq_len; ++row) {
			float row_max = attention_scores[row * max_seq_len];
			for (int col = 1; col < max_seq_len; ++col) {
				if (attention_scores[row * max_seq_len + col] > row_max) row_max = attention_scores[row * max_seq_len + col];
			}
			float row_sum = 0.0f;
			for (int col = 0; col < max_seq_len; ++col) {
				row_sum += std::exp(attention_scores[row * max_seq_len + col] - row_max);
			}
			for (int col = 0; col < max_seq_len; ++col) {
				attention_scores[row * max_seq_len + col] = std::exp(attention_scores[row * max_seq_len + col] - row_max) / row_sum;
			}
		}
		matmult(full_upstream, temporary_buffer, score_gradients, max_seq_len, embedding_dimension, max_seq_len, false, true, 1.0f, 0.0f);
		for (int row = 0; row < max_seq_len; ++row) {
			float weighted_sum = 0.0f;
			for (int col = 0; col < max_seq_len; ++col) weighted_sum += score_gradients[row * max_seq_len + col] * attention_scores[row * max_seq_len + col];
			for (int col = 0; col < max_seq_len; ++col) score_gradients[row * max_seq_len + col] = attention_scores[row * max_seq_len + col] * (score_gradients[row * max_seq_len + col] - weighted_sum);
		}
		float* value_gradient = temporary_buffer;
		matmult(attention_scores, full_upstream, value_gradient, max_seq_len, max_seq_len, embedding_dimension, true, false, 1.0f, 0.0f);
		float* gradients = parametric_layer->weight_gradients;
		matmult(seq_input, value_gradient, gradients + 2 * embedding_dimension * embedding_dimension, embedding_dimension, max_seq_len, embedding_dimension, true, false, 1.0f, 1.0f);
		float* query_gradient = temporary_buffer;
		matmult(score_gradients, key, query_gradient, max_seq_len, max_seq_len, embedding_dimension, false, false, inverse_sqrt_dimension, 0.0f);
		matmult(seq_input, query_gradient, gradients, embedding_dimension, max_seq_len, embedding_dimension, true, false, 1.0f, 1.0f);
		float* key_gradient = temporary_buffer;
		matmult(score_gradients, query, key_gradient, max_seq_len, max_seq_len, embedding_dimension, true, false, inverse_sqrt_dimension, 0.0f);
		matmult(seq_input, key_gradient, gradients + embedding_dimension * embedding_dimension, embedding_dimension, max_seq_len, embedding_dimension, true, false, 1.0f, 1.0f);
		matmult(query_gradient, query_weights, seq_output_grad, max_seq_len, embedding_dimension, embedding_dimension, false, true, 1.0f, 0.0f);
		matmult(key_gradient, key_weights, seq_output_grad, max_seq_len, embedding_dimension, embedding_dimension, false, true, 1.0f, 1.0f);
		matmult(value_gradient, value_weights, seq_output_grad, max_seq_len, embedding_dimension, embedding_dimension, false, true, 1.0f, 1.0f);
		batch_sizes[seq] = original_len;
		offset += max_seq_len * embedding_dimension;
		upstream_offset += actual_len * embedding_dimension;
	}
};
#ifndef TRAINING_ON
HookFunc AttentionForwardWithCache = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	int max_seq_len = parametric_layer->extra_args[0];
	int kv_cache_idx = parametric_layer->extra_args[1];
	float* weight_matrix = parametric_layer->weights_begin;
	float* query_weights = weight_matrix;
	float* key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float* value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float* query = parametric_layer->scratch_pointer;
	float* key = query + max_seq_len * embedding_dimension;
	float* value = key + max_seq_len * embedding_dimension;
	float* attention_scores = value + max_seq_len * embedding_dimension;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	std::vector<float>& cache = kv_cache_pool[kv_cache_idx];
	int offset = 0;
	for (int seq = 0; seq < batch_count; ++seq) {
		int actual_len = batch_sizes[seq];
		float* seq_input = preactivation_values + offset;
		float* seq_output = output_values + offset;
		matmult(seq_input, query_weights, query, actual_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, key_weights, key, actual_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, value_weights, value, actual_len, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		int cached_len = cache.size() / (2 * embedding_dimension);
		int total_len = cached_len + actual_len;
		cache.resize(total_len * 2 * embedding_dimension);
		std::copy(key, key + actual_len * embedding_dimension, cache.begin() + cached_len * embedding_dimension);
		std::copy(value, value + actual_len * embedding_dimension, cache.begin() + total_len * embedding_dimension + cached_len * embedding_dimension);
		float* all_keys = cache.data();
		float* all_values = cache.data() + total_len * embedding_dimension;
		matmult(query, all_keys, attention_scores, actual_len, embedding_dimension, total_len, false, true, inverse_sqrt_dimension, 0.0f);
		for (int row = 0; row < actual_len; ++row) {
			for (int col = cached_len + row + 1; col < total_len; ++col) {
				attention_scores[row * total_len + col] = -1e30f;
			}
		}
		for (int row = 0; row < actual_len; ++row) {
			float row_max = attention_scores[row * total_len];
			for (int col = 1; col < total_len; ++col) {
				if (attention_scores[row * total_len + col] > row_max) row_max = attention_scores[row * total_len + col];
			}
			float row_sum = 0.0f;
			for (int col = 0; col < total_len; ++col) {
				row_sum += std::exp(attention_scores[row * total_len + col] - row_max);
			}
			for (int col = 0; col < total_len; ++col) {
				attention_scores[row * total_len + col] = std::exp(attention_scores[row * total_len + col] - row_max) / row_sum;
			}
		}
		matmult(attention_scores, all_values, seq_output, actual_len, total_len, embedding_dimension, false, false, 1.0f, 0.0f);
		offset += max_seq_len * embedding_dimension;
	}
};
HookDerivative AttentionDerivativeWithCache = [](LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
};
#endif
#endif
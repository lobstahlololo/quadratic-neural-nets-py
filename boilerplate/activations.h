
#ifndef ACTIVATIONS_IMPORTED
#define ACTIVATIONS_IMPORTED
#include "../model/network.h"
#include <cmath>
#include <algorithm>
#include <variant>
#include <vector>
// utilities 

HookFunc Residual = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = preactivation_values[i] + original_inputs[i];
	}
};


HookDerivative ResidualGradHook = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		output_gradient[i] = upstream_gradient[i];
	}
};

HookFunc NonLearnableLayerNorm = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	for (int b = 0; b < sample_count; ++b) {
		float* input_sample = preactivation_values + b * feature_count;
		float* output_sample = output_values + b * feature_count;
		float mean = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			mean += input_sample[i];
		}
		mean /= feature_count;
		float variance = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			variance += (input_sample[i] - mean) * (input_sample[i] - mean);
		}
		variance /= feature_count;
		float inverse_std_dev = 1.0f / std::sqrt(variance + 1e-5f);
		for (int i = 0; i < feature_count; ++i) {
			output_sample[i] = (input_sample[i] - mean) * inverse_std_dev;
		}
	}
};

HookDerivative NonLearnableLayerNormDerivative = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
		float* raw_input = original_inputs + sample_idx * feature_count;
		float* upstream = upstream_gradient + sample_idx * feature_count;
		float* out_grad = output_gradient + sample_idx * feature_count;
		float mean = 0.0f;
		for (int i = 0; i < feature_count; ++i) mean += raw_input[i];
		mean /= feature_count;
		float var = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			float diff = raw_input[i] - mean;
			var += diff * diff;
		}
		var /= feature_count;
		float inverse_std = 1.0f / std::sqrt(var + 1e-5f);
		float mean_up = 0.0f;
		for (int i = 0; i < feature_count; ++i) mean_up += upstream[i];
		mean_up /= feature_count;
		float sum_up_xhat = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			float xhat = (raw_input[i] - mean) * inverse_std;
			sum_up_xhat += upstream[i] * xhat;
		}
		sum_up_xhat /= feature_count;
		for (int i = 0; i < feature_count; ++i) {
			float xhat = (raw_input[i] - mean) * inverse_std;
			out_grad[i] = (upstream[i] - mean_up - xhat * sum_up_xhat) * inverse_std;
		}
	}
};

HookFunc LearnableLayerNorm = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	float* gamma = layer.weights_begin;
	float* beta = layer.weights_begin + layer.input;
	for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
		float* input_sample = preactivation_values + sample_idx * feature_count;
		float* output_sample = output_values + sample_idx * feature_count;
		float mean = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			mean += input_sample[i];
		}
		mean /= feature_count;
		float variance = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			variance += (input_sample[i] - mean) * (input_sample[i] - mean);
		}
		variance /= feature_count;
		float inverse_std_dev = 1.0f / std::sqrt(variance + 1e-5f);
		for (int i = 0; i < feature_count; ++i) {
			output_sample[i] = (input_sample[i] - mean) * inverse_std_dev * gamma[i] + beta[i];
		}
	}
};

HookDerivative LearnableLayerNormDerivative = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	float* gamma = parametric_layer->weights_begin;
	float* beta = gamma + feature_count;
	float* weight_gradients_ptr = parametric_layer->weight_gradients;
	float* gamma_gradient = weight_gradients_ptr;
	float* beta_gradient = weight_gradients_ptr + feature_count;
	for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
		float* raw_input = original_inputs + sample_idx * feature_count;
		float* upstream = upstream_gradient + sample_idx * feature_count;
		float* out_grad = output_gradient + sample_idx * feature_count;
		float mean = 0.0f;
		for (int i = 0; i < feature_count; ++i) mean += raw_input[i];
		mean /= feature_count;
		float var = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			float diff = raw_input[i] - mean;
			var += diff * diff;
		}
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
	}
};

HookFunc RMSNorm = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	float* gamma = layer.weights_begin;
	for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
		float* input_sample = preactivation_values + sample_idx * feature_count;
		float* output_sample = output_values + sample_idx * feature_count;
		float rms = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			rms += input_sample[i] * input_sample[i];
		}
		rms = std::sqrt(rms / feature_count + 1e-5f);
		float inverse_rms = 1.0f / rms;
		for (int i = 0; i < feature_count; ++i) {
			output_sample[i] = input_sample[i] * inverse_rms * gamma[i];
		}
	}
};

HookDerivative RMSNormDerivative = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	float* gamma = parametric_layer->weights_begin;
	float* gamma_gradient = parametric_layer->weight_gradients;
	for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
		float* raw_input = original_inputs + sample_idx * feature_count;
		float* upstream = upstream_gradient + sample_idx * feature_count;
		float* out_grad = output_gradient + sample_idx * feature_count;
		float sum_sq = 0.0f;
		for (int i = 0; i < feature_count; ++i) sum_sq += raw_input[i] * raw_input[i];
		float rms = std::sqrt(sum_sq / feature_count + 1e-5f);
		float inverse_rms = 1.0f / rms;
		float inverse_rms_cubed = inverse_rms * inverse_rms * inverse_rms;
		float sum_gamma_up_x = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			sum_gamma_up_x += gamma[i] * upstream[i] * raw_input[i];
		}
		float inv_N = 1.0f / feature_count;
		for (int i = 0; i < feature_count; ++i) {
			float dx = gamma[i] * upstream[i] * inverse_rms - raw_input[i] * sum_gamma_up_x * inverse_rms_cubed * inv_N;
			out_grad[i] = dx;
			gamma_gradient[i] += upstream[i] * raw_input[i] * inverse_rms;
		}
	}
};

// activations

HookFunc ReLuHook = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = preactivation_values[i] > 0.0f ? preactivation_values[i] : 0.0f;
	}
};
HookDerivative ReLuGradHook = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		output_gradient[i] = upstream_gradient[i] * (preactivation_values[i] > 0.0f ? 1.0f : 0.0f);
	}
};
HookFunc SigmoidHook = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = 1.0f / (1.0f + std::exp(-preactivation_values[i]));
	}
};
HookDerivative SigmoidGradHook = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		float sig = 1.0f / (1.0f + std::exp(-preactivation_values[i]));
		output_gradient[i] = upstream_gradient[i] * sig * (1.0f - sig);
	}
};
HookFunc TanhHook = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		output_values[i] = std::tanh(preactivation_values[i]);
	}
};
HookDerivative TanhGradHook = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int total = feature_count * sample_count;
	for (int i = 0; i < total; ++i) {
		float t = std::tanh(preactivation_values[i]);
		output_gradient[i] = upstream_gradient[i] * (1.0f - t * t);
	}
};
HookFunc Softmax = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
		float* input_sample = preactivation_values + sample_idx * feature_count;
		float* output_sample = output_values + sample_idx * feature_count;
		float maximum = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			maximum = std::max(maximum, input_sample[i]);
		}
		float sum = 0.0f;
		for (int i = 0; i < feature_count; ++i) {
			sum += std::exp(input_sample[i] - maximum);
		}
		for (int i = 0; i < feature_count; ++i) {
			output_sample[i] = std::exp(input_sample[i] - maximum) / sum;
		}
	}
};

HookDerivative SoftmaxDerivative = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int sample_count_per_target = sample_count / correct_indices.size();
	std::vector<float> softmax_values(sample_count * feature_count);
	for (int i = 0; i < sample_count; ++i) {
		float* inp = preactivation_values + i * feature_count;
		float* out = softmax_values.data() + i * feature_count;
		float maximum = 0.0f;
		for (int j = 0; j < feature_count; ++j) {
			maximum = std::max(maximum, inp[j]);
		}
		float sum = 0.0f;
		for (int j = 0; j < feature_count; ++j) {
			sum += std::exp(inp[j] - maximum);
		}
		for (int j = 0; j < feature_count; ++j) {
			out[j] = std::exp(inp[j] - maximum) / sum;
		}
	}

	for (int target_idx = 0; target_idx < correct_indices.size(); ++target_idx) {
		int correct_token = correct_indices[target_idx];
		for (int b = 0; b < sample_count_per_target; ++b) {
			float* soft = softmax_values.data() + (target_idx * sample_count_per_target + b) * feature_count;
			float* out = output_gradient + (target_idx * sample_count_per_target + b) * feature_count;
			float* up = upstream_gradient + (target_idx * sample_count_per_target + b) * feature_count;
			float grad_at_correct = up[correct_token];
			for (int j = 0; j < feature_count; ++j) {
				out[j] = up[j] * soft[j] - soft[j] * soft[correct_token] * grad_at_correct;
			}
		}
	}
};

HookDerivative SoftmaxForCrossEntropyLossDerivative = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	int sample_count_per_target = sample_count / correct_indices.size();
	for (int target_idx = 0; target_idx < correct_indices.size(); ++target_idx) {
		int correct_token = correct_indices[target_idx];
		for (int b = 0; b < sample_count_per_target; ++b) {
			float* logit_sample = preactivation_values + (target_idx * sample_count_per_target + b) * feature_count;
			float* out_sample = output_gradient + (target_idx * sample_count_per_target + b) * feature_count;
			float maximum = logit_sample[0];
			for (int j = 1; j < feature_count; ++j) {
				if (logit_sample[j] > maximum) maximum = logit_sample[j];
			}
			float sum_exp = 0.0f;
			for (int j = 0; j < feature_count; ++j) {
				sum_exp += std::exp(logit_sample[j] - maximum);
			}
			for (int j = 0; j < feature_count; ++j) {
				float probability = std::exp(logit_sample[j] - maximum) / sum_exp;
				out_sample[j] = probability;
			}
			out_sample[correct_token] -= 1.0f;
		}
	}
};

HookFunc EmbeddingForward = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	for (int i = 0; i < sample_count; ++i) {
		int token_index = static_cast<int>(original_inputs[i]);
		float* weight_row = parametric_layer->weights_begin + token_index * embedding_dimension;
		float* output_row = output_values + i * embedding_dimension;
		for (int j = 0; j < embedding_dimension; ++j) output_row[j] = weight_row[j];
	}
};

HookDerivative EmbeddingDerivative = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	float* weight_gradients_ptr = parametric_layer->weight_gradients;
	for (int i = 0; i < sample_count; ++i) {
		int token_index = static_cast<int>(original_inputs[i]);
		float* gradient_row = weight_gradients_ptr + token_index * embedding_dimension;
		float* upstream_row = upstream_gradient + i * embedding_dimension;
		for (int j = 0; j < embedding_dimension; ++j) gradient_row[j] += upstream_row[j];
	}
	for (int i = 0; i < sample_count * embedding_dimension; ++i) output_gradient[i] = 0.0f;
};

HookFunc AttentionForward = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* output_values, int feature_count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	int total_tokens = sample_count;
	int sequence_count = parametric_layer->batch;
	int sequence_length = total_tokens / sequence_count;
	float* weight_matrix = parametric_layer->weights_begin;
	float* query_weights = weight_matrix;
	float* key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float* value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float* query = parametric_layer->scratch_pointer;
	float* key = query + sequence_length * embedding_dimension;
	float* value = key + sequence_length * embedding_dimension;
	float* attention_scores = value + sequence_length * embedding_dimension;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	for (int seq = 0; seq < sequence_count; ++seq) {
		float* seq_input = preactivation_values + seq * sequence_length * embedding_dimension;
		float* seq_output = output_values + seq * sequence_length * embedding_dimension;
		matmult(seq_input, query_weights, query, sequence_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, key_weights, key, sequence_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, value_weights, value, sequence_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(query, key, attention_scores, sequence_length, embedding_dimension, sequence_length, false, true, inverse_sqrt_dimension, 0.0f);
		for (int row = 0; row < sequence_length; ++row) {
			for (int col = row + 1; col < sequence_length; ++col) {
				attention_scores[row * sequence_length + col] = -1e30f;
			}
		}
		for (int row = 0; row < sequence_length; ++row) {
			float row_max = attention_scores[row * sequence_length];
			for (int col = 1; col < sequence_length; ++col) {
				if (attention_scores[row * sequence_length + col] > row_max) row_max = attention_scores[row * sequence_length + col];
			}
			float row_sum = 0.0f;
			for (int col = 0; col < sequence_length; ++col) {
				row_sum += std::exp(attention_scores[row * sequence_length + col] - row_max);
			}
			for (int col = 0; col < sequence_length; ++col) {
				attention_scores[row * sequence_length + col] = std::exp(attention_scores[row * sequence_length + col] - row_max) / row_sum;
			}
		}
		matmult(attention_scores, value, seq_output, sequence_length, sequence_length, embedding_dimension, false, false, 1.0f, 0.0f);
	}
};

HookDerivative AttentionDerivative = [](LayerRef layer, int sample_count, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dimension = feature_count;
	int total_tokens = sample_count;
	int sequence_count = parametric_layer->batch;
	int sequence_length = total_tokens / sequence_count;
	float* input_embeddings = original_inputs;
	float* weight_matrix = parametric_layer->weights_begin;
	float* query_weights = weight_matrix;
	float* key_weights = weight_matrix + embedding_dimension * embedding_dimension;
	float* value_weights = weight_matrix + 2 * embedding_dimension * embedding_dimension;
	float* query = parametric_layer->scratch_pointer;
	float* key = query + sequence_length * embedding_dimension;
	float* temporary_buffer = key + sequence_length * embedding_dimension;
	float* attention_scores = temporary_buffer + sequence_length * embedding_dimension;
	float* score_gradients = attention_scores + sequence_length * sequence_length;
	float inverse_sqrt_dimension = 1.0f / std::sqrt(static_cast<float>(embedding_dimension));
	for (int seq = 0; seq < sequence_count; ++seq) {
		float* seq_input = input_embeddings + seq * sequence_length * embedding_dimension;
		float* seq_upstream = upstream_gradient + seq * sequence_length * embedding_dimension;
		float* seq_output_grad = output_gradient + seq * sequence_length * embedding_dimension;
		matmult(seq_input, query_weights, query, sequence_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, key_weights, key, sequence_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(seq_input, value_weights, temporary_buffer, sequence_length, embedding_dimension, embedding_dimension, false, false, 1.0f, 0.0f);
		matmult(query, key, attention_scores, sequence_length, embedding_dimension, sequence_length, false, true, inverse_sqrt_dimension, 0.0f);
		for (int row = 0; row < sequence_length; ++row) {
			for (int col = row + 1; col < sequence_length; ++col) {
				attention_scores[row * sequence_length + col] = -1e30f;
			}
		}
		for (int row = 0; row < sequence_length; ++row) {
			float row_max = attention_scores[row * sequence_length];
			for (int col = 1; col < sequence_length; ++col) {
				if (attention_scores[row * sequence_length + col] > row_max) row_max = attention_scores[row * sequence_length + col];
			}
			float row_sum = 0.0f;
			for (int col = 0; col < sequence_length; ++col) {
				row_sum += std::exp(attention_scores[row * sequence_length + col] - row_max);
			}
			for (int col = 0; col < sequence_length; ++col) {
				attention_scores[row * sequence_length + col] = std::exp(attention_scores[row * sequence_length + col] - row_max) / row_sum;
			}
		}
		matmult(seq_upstream, temporary_buffer, score_gradients, sequence_length, embedding_dimension, sequence_length, false, true, 1.0f, 0.0f);
		for (int row = 0; row < sequence_length; ++row) {
			float weighted_sum = 0.0f;
			for (int col = 0; col < sequence_length; ++col) weighted_sum += score_gradients[row * sequence_length + col] * attention_scores[row * sequence_length + col];
			for (int col = 0; col < sequence_length; ++col) score_gradients[row * sequence_length + col] = attention_scores[row * sequence_length + col] * (score_gradients[row * sequence_length + col] - weighted_sum);
		}
		float* value_gradient = temporary_buffer;
		matmult(attention_scores, seq_upstream, value_gradient, sequence_length, sequence_length, embedding_dimension, true, false, 1.0f, 0.0f);
		float* gradients = parametric_layer->weight_gradients;
		matmult(seq_input, value_gradient, gradients + 2 * embedding_dimension * embedding_dimension, embedding_dimension, sequence_length, embedding_dimension, true, false, 1.0f, 1.0f);
		matmult(value_gradient, value_weights, seq_output_grad, sequence_length, embedding_dimension, embedding_dimension, false, true, 1.0f, 0.0f);
		float* query_gradient = temporary_buffer;
		matmult(score_gradients, key, query_gradient, sequence_length, sequence_length, embedding_dimension, false, false, inverse_sqrt_dimension, 0.0f);
		matmult(seq_input, query_gradient, gradients, embedding_dimension, sequence_length, embedding_dimension, true, false, 1.0f, 1.0f);
		matmult(query_gradient, query_weights, seq_output_grad, sequence_length, embedding_dimension, embedding_dimension, false, true, 1.0f, 1.0f);
		float* key_gradient = temporary_buffer;
		matmult(score_gradients, query, key_gradient, sequence_length, sequence_length, embedding_dimension, true, false, inverse_sqrt_dimension, 0.0f);
		matmult(seq_input, key_gradient, gradients + embedding_dimension * embedding_dimension, embedding_dimension, sequence_length, embedding_dimension, true, false, 1.0f, 1.0f);
		matmult(key_gradient, key_weights, seq_output_grad, sequence_length, embedding_dimension, embedding_dimension, false, true, 1.0f, 1.0f);
	}
};


#endif
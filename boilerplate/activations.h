
#ifndef ACTIVATIONS_IMPORTED
#define ACTIVATIONS_IMPORTED
#include "../model/network.h"
#include <cmath>
// utilities 

HookFunc Residual = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = inputs[i] + layerInputs[i];
	}
};


HookDerivative ResidualGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = upstream_grad[i];
	}
};

HookFunc NonLearnableLayerNorm = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	for (int b = 0; b < batchSize; ++b) {
		float* inp = inputs + b * count;
		float* out = outputs + b * count;
		float mean = 0.0f;
		for (int i = 0; i < count; ++i) {
			mean += inp[i];
		}
		mean /= count;
		float variance = 0.0f;
		for (int i = 0; i < count; ++i) {
			variance += (inp[i] - mean) * (inp[i] - mean);
		}
		variance /= count;
		float invStdDev = 1.0f / std::sqrt(variance + 1e-5f);
		for (int i = 0; i < count; ++i) {
			out[i] = (inp[i] - mean) * invStdDev;
		}
	}
};

HookDerivative NonLearnableLayerNormDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	for (int b = 0; b < batchSize; ++b) {
		float* raw_in = layerInputs + b * count;
		float* upstream = upstream_grad + b * count;
		float* out_grad = outputs + b * count;
		float mean = 0.0f;
		for (int i = 0; i < count; ++i) mean += raw_in[i];
		mean /= count;
		float var = 0.0f;
		for (int i = 0; i < count; ++i) {
			float diff = raw_in[i] - mean;
			var += diff * diff;
		}
		var /= count;
		float invStd = 1.0f / std::sqrt(var + 1e-5f);
		float mean_up = 0.0f;
		for (int i = 0; i < count; ++i) mean_up += upstream[i];
		mean_up /= count;
		float sum_up_xhat = 0.0f;
		for (int i = 0; i < count; ++i) {
			float xhat = (raw_in[i] - mean) * invStd;
			sum_up_xhat += upstream[i] * xhat;
		}
		sum_up_xhat /= count;
		for (int i = 0; i < count; ++i) {
			float xhat = (raw_in[i] - mean) * invStd;
			out_grad[i] = (upstream[i] - mean_up - xhat * sum_up_xhat) * invStd;
		}
	}
};

HookFunc LearnableLayerNorm = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	float* gamma = layer.weightsBegin;
	float* beta = layer.weightsBegin + layer.input;
	for (int b = 0; b < batchSize; ++b) {
		float* inp = inputs + b * count;
		float* out = outputs + b * count;
		float mean = 0.0f;
		for (int i = 0; i < count; ++i) {
			mean += inp[i];
		}
		mean /= count;
		float variance = 0.0f;
		for (int i = 0; i < count; ++i) {
			variance += (inp[i] - mean) * (inp[i] - mean);
		}
		variance /= count;
		float invStdDev = 1.0f / std::sqrt(variance + 1e-5f);
		for (int i = 0; i < count; ++i) {
			out[i] = (inp[i] - mean) * invStdDev * gamma[i] + beta[i];
		}
	}
};

HookDerivative LearnableLayerNormDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	ParametricLayer* l = std::get<ParametricLayer*>(layer);
	float* gamma = l->weightsBegin;
	float* beta = gamma + count;
	float* weight_gradients = l->gradients;
	float* gamma_grad = weight_gradients;
	float* beta_grad = weight_gradients + count;
	for (int b = 0; b < batchSize; ++b) {
		float* raw_in = layerInputs + b * count;
		float* upstream = upstream_grad + b * count;
		float* out_grad = outputs + b * count;
		float mean = 0.0f;
		for (int i = 0; i < count; ++i) mean += raw_in[i];
		mean /= count;
		float var = 0.0f;
		for (int i = 0; i < count; ++i) {
			float diff = raw_in[i] - mean;
			var += diff * diff;
		}
		var /= count;
		float invStd = 1.0f / std::sqrt(var + 1e-5f);
		float sum_gamma_up = 0.0f;
		float sum_gamma_up_xhat = 0.0f;
		for (int i = 0; i < count; ++i) {
			float x_hat = (raw_in[i] - mean) * invStd;
			float g_up = gamma[i] * upstream[i];
			sum_gamma_up += g_up;
			sum_gamma_up_xhat += g_up * x_hat;
		}
		float inv_N = 1.0f / count;
		for (int i = 0; i < count; ++i) {
			float x_hat = (raw_in[i] - mean) * invStd;
			float dx = (gamma[i] * upstream[i] - sum_gamma_up * inv_N - x_hat * sum_gamma_up_xhat * inv_N) * invStd;
			out_grad[i] = dx;
			gamma_grad[i] += upstream[i] * x_hat;
			beta_grad[i] += upstream[i];
		}
	}
};

HookFunc RMSNorm = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	float* gamma = layer.weightsBegin;
	for (int b = 0; b < batchSize; ++b) {
		float* inp = inputs + b * count;
		float* out = outputs + b * count;
		float rms = 0.0f;
		for (int i = 0; i < count; ++i) {
			rms += inp[i] * inp[i];
		}
		rms = std::sqrt(rms / count + 1e-5f);
		float invRms = 1.0f / rms;
		for (int i = 0; i < count; ++i) {
			out[i] = inp[i] * invRms * gamma[i];
		}
	}
};

HookDerivative RMSNormDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	ParametricLayer* l = std::get<ParametricLayer*>(layer);
	float* gamma = l->weightsBegin;
	float* gamma_grad = l->gradients;
	for (int b = 0; b < batchSize; ++b) {
		float* raw_in = layerInputs + b * count;
		float* upstream = upstream_grad + b * count;
		float* out_grad = outputs + b * count;
		float sum_sq = 0.0f;
		for (int i = 0; i < count; ++i) sum_sq += raw_in[i] * raw_in[i];
		float rms = std::sqrt(sum_sq / count + 1e-5f);
		float invRms = 1.0f / rms;
		float invRms3 = invRms * invRms * invRms;
		float sum_gamma_up_x = 0.0f;
		for (int i = 0; i < count; ++i) {
			sum_gamma_up_x += gamma[i] * upstream[i] * raw_in[i];
		}
		float inv_N = 1.0f / count;
		for (int i = 0; i < count; ++i) {
			float dx = gamma[i] * upstream[i] * invRms - raw_in[i] * sum_gamma_up_x * invRms3 * inv_N;
			out_grad[i] = dx;
			gamma_grad[i] += upstream[i] * raw_in[i] * invRms;
		}
	}
};

// activations

HookFunc ReLuHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = inputs[i] > 0.0f ? inputs[i] : 0.0f;
	}
};
HookDerivative ReLuGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = upstream_grad[i] * (preactivations[i] > 0.0f ? 1.0f : 0.0f);
	}
};
HookFunc SigmoidHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = 1.0f / (1.0f + std::exp(-inputs[i]));
	}
};
HookDerivative SigmoidGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		float sig = 1.0f / (1.0f + std::exp(-preactivations[i]));
		outputs[i] = upstream_grad[i] * sig * (1.0f - sig);
	}
};
HookFunc TanhHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = std::tanh(inputs[i]);
	}
};
HookDerivative TanhGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		float t = std::tanh(preactivations[i]);
		outputs[i] = upstream_grad[i] * (1.0f - t * t);
	}
};
HookFunc Softmax = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	for (int b = 0; b < batchSize; ++b) {
		float* inp = inputs + b * count;
		float* out = outputs + b * count;
		float maximum = 0.0f;
		for (int i = 0; i < count; ++i) {
			maximum = std::max(maximum, inp[i]);
		}
		float sum = 0.0f;
		for (int i = 0; i < count; ++i) {
			sum += std::exp(inp[i] - maximum);
		}
		for (int i = 0; i < count; ++i) {
			out[i] = std::exp(inp[i] - maximum) / sum;

		}
	}
};

HookDerivative SoftmaxDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	int originalBatchSize = batchSize / correctIndices.size();
	std::vector<float> softMaxes(batchSize * count);
	for (int i = 0; i < batchSize; ++i) {
		float* inp = preactivations + i * count;
		float* out = softMaxes.data() + i * count;
		float maximum = 0.0f;
		for (int j = 0; j < count; ++j) {
			maximum = std::max(maximum, inp[j]);
		}
		float sum = 0.0f;
		for (int j = 0; j < count; ++j) {
			sum += std::exp(inp[j] - maximum);
		}
		for (int j = 0; j < count; ++j) {
			out[j] = std::exp(inp[j] - maximum) / sum;
		}
	}

	for (int i = 0; i < correctIndices.size(); ++i) {
		int idx = correctIndices[i];
		for (int b = 0; b <  originalBatchSize; ++b) {
			float* soft = softMaxes.data() + (i * originalBatchSize + b) * count;
			float* out = outputs + (i * originalBatchSize + b) * count;
			float* up = upstream_grad + (i * originalBatchSize + b) * count;
			float grad_at_idx = up[idx];
			for (int j = 0; j < count; ++j) {
				out[j] = up[j] * soft[j] - soft[j] * soft[idx] * grad_at_idx;
			}
		}
	}
};

HookFunc EmbeddingForward = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	ParametricLayer* l = std::get<ParametricLayer*>(layer);
	int embeddingDim = count;
	for (int i = 0; i < batchSize; ++i) {
		int idx = static_cast<int>(inputs[i]);
		float* weightRow = l->weightsBegin + idx * embeddingDim;
		float* outRow = outputs + i * embeddingDim;
		for (int j = 0; j < embeddingDim; ++j) outRow[j] = weightRow[j];
	}
};

HookDerivative EmbeddingDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	ParametricLayer* l = std::get<ParametricLayer*>(layer);
	int embeddingDim = count;
	float* gradients = l->gradients;
	for (int i = 0; i < batchSize; ++i) {
		int idx = static_cast<int>(layerInputs[i]);
		float* gradRow = gradients + idx * embeddingDim;
		float* upstreamRow = upstream_grad + i * embeddingDim;
		for (int j = 0; j < embeddingDim; ++j) gradRow[j] += upstreamRow[j];
	}
	for (int i = 0; i < batchSize * embeddingDim; ++i) outputs[i] = 0.0f;
};

HookFunc AttentionForward = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dim = count;
	int sequence_length = batchSize;
	float* weight_matrix = parametric_layer->weightsBegin;
	float* query_weights = weight_matrix;
	float* key_weights = weight_matrix + embedding_dim * embedding_dim;
	float* value_weights = weight_matrix + 2 * embedding_dim * embedding_dim;
	float* query = parametric_layer->scratchPad;
	float* key = query + sequence_length * embedding_dim;
	float* value = key + sequence_length * embedding_dim;
	float* attention_scores = value + sequence_length * embedding_dim;
	matmult(inputs, query_weights, query, sequence_length, embedding_dim, embedding_dim, false, false, 1.0f, 0.0f);
	matmult(inputs, key_weights, key, sequence_length, embedding_dim, embedding_dim, false, false, 1.0f, 0.0f);
	matmult(inputs, value_weights, value, sequence_length, embedding_dim, embedding_dim, false, false, 1.0f, 0.0f);
	float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
	matmult(query, key, attention_scores, sequence_length, embedding_dim, sequence_length, false, true, inv_sqrt_dim, 0.0f);
	for (int row = 0; row < sequence_length; ++row) {
		float row_max = attention_scores[row * sequence_length];
		for (int col = 1; col < sequence_length; ++col) if (attention_scores[row * sequence_length + col] > row_max) row_max = attention_scores[row * sequence_length + col];
		float row_sum = 0.0f;
		for (int col = 0; col < sequence_length; ++col) row_sum += std::exp(attention_scores[row * sequence_length + col] - row_max);
		for (int col = 0; col < sequence_length; ++col) attention_scores[row * sequence_length + col] = std::exp(attention_scores[row * sequence_length + col] - row_max) / row_sum;
	}
	matmult(attention_scores, value, outputs, sequence_length, sequence_length, embedding_dim, false, false, 1.0f, 0.0f);
};

HookDerivative AttentionDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* preactivations, float* upstream_grad, float* outputs, int count, const std::vector<int>& correctIndices) {
	ParametricLayer* parametric_layer = std::get<ParametricLayer*>(layer);
	int embedding_dim = count;
	int sequence_length = batchSize;
	float* input_embeddings = layerInputs;
	float* weight_matrix = parametric_layer->weightsBegin;
	float* query_weights = weight_matrix;
	float* key_weights = weight_matrix + embedding_dim * embedding_dim;
	float* value_weights = weight_matrix + 2 * embedding_dim * embedding_dim;
	float* query = parametric_layer->scratchPad;
	float* key = query + sequence_length * embedding_dim;
	float* reusable_buffer = key + sequence_length * embedding_dim;
	float* attention_scores = reusable_buffer + sequence_length * embedding_dim;
	float* score_gradients = attention_scores + sequence_length * sequence_length;
	matmult(input_embeddings, query_weights, query, sequence_length, embedding_dim, embedding_dim, false, false, 1.0f, 0.0f);
	matmult(input_embeddings, key_weights, key, sequence_length, embedding_dim, embedding_dim, false, false, 1.0f, 0.0f);
	matmult(input_embeddings, value_weights, reusable_buffer, sequence_length, embedding_dim, embedding_dim, false, false, 1.0f, 0.0f);
	float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(embedding_dim));
	matmult(query, key, attention_scores, sequence_length, embedding_dim, sequence_length, false, true, inv_sqrt_dim, 0.0f);
	matmult(upstream_grad, reusable_buffer, score_gradients, sequence_length, embedding_dim, sequence_length, false, true, 1.0f, 0.0f);
	for (int row = 0; row < sequence_length; ++row) {
		float weighted_sum = 0.0f;
		for (int col = 0; col < sequence_length; ++col) weighted_sum += score_gradients[row * sequence_length + col] * attention_scores[row * sequence_length + col];
		for (int col = 0; col < sequence_length; ++col) score_gradients[row * sequence_length + col] = attention_scores[row * sequence_length + col] * (score_gradients[row * sequence_length + col] - weighted_sum);
	}
	float* value_gradient = reusable_buffer;
	matmult(attention_scores, upstream_grad, value_gradient, sequence_length, sequence_length, embedding_dim, true, false, 1.0f, 0.0f);
	float* gradients = parametric_layer->gradients;
	matmult(input_embeddings, value_gradient, gradients + 2 * embedding_dim * embedding_dim, embedding_dim, sequence_length, embedding_dim, true, false, 1.0f, 1.0f);
	matmult(value_gradient, value_weights, outputs, sequence_length, embedding_dim, embedding_dim, false, true, 1.0f, 0.0f);
	float* query_gradient = reusable_buffer;
	matmult(score_gradients, key, query_gradient, sequence_length, sequence_length, embedding_dim, false, false, inv_sqrt_dim, 0.0f);
	matmult(input_embeddings, query_gradient, gradients, embedding_dim, sequence_length, embedding_dim, true, false, 1.0f, 1.0f);
	matmult(query_gradient, query_weights, outputs, sequence_length, embedding_dim, embedding_dim, false, true, 1.0f, 1.0f);
	float* key_gradient = reusable_buffer;
	matmult(score_gradients, query, key_gradient, sequence_length, sequence_length, embedding_dim, true, false, inv_sqrt_dim, 0.0f);
	matmult(input_embeddings, key_gradient, gradients + embedding_dim * embedding_dim, embedding_dim, sequence_length, embedding_dim, true, false, 1.0f, 1.0f);
	matmult(key_gradient, key_weights, outputs, sequence_length, embedding_dim, embedding_dim, false, true, 1.0f, 1.0f);
};


#endif
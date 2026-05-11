
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


HookDerivative ResidualGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = 1.0f;
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

HookDerivative NonLearnableLayerNormDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
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
			outputs[b * count + i] = invStdDev * (1.0f - 1.0f / count);
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

HookDerivative LearnableLayerNormDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	float* gamma = layer.weightsBegin;
	int total = count * batchSize;
	std::vector<float> current(inputs, inputs + total);
	for (auto& hook : layer.forwardHooks) {
		if (hook == LearnableLayerNorm) break;
		std::vector<float> temp(total);
		hook(layer, batchSize, layerInputs, current.data(), temp.data(), count);
		current.swap(temp);
	}
	for (int b = 0; b < batchSize; ++b) {
		float* curr = current.data() + b * count;
		float* out = outputs + b * count;
		float mean = 0.0f;
		for (int i = 0; i < count; ++i) mean += curr[i];
		mean /= count;
		float var = 0.0f;
		for (int i = 0; i < count; ++i) var += (curr[i] - mean) * (curr[i] - mean);
		var /= count;
		float invStd = 1.0f / std::sqrt(var + 1e-5f);
		for (int i = 0; i < count; ++i) {
			out[i] = invStd * gamma[i] * (1.0f - 1.0f / count);
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

HookDerivative RMSNormDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	float* gamma = layer.weightsBegin;
	int total = count * batchSize;
	std::vector<float> current(inputs, inputs + total);
	for (auto& hook : layer.forwardHooks) {
		if (hook == RMSNorm) break;
		std::vector<float> temp(total);
		hook(layer, batchSize, layerInputs, current.data(), temp.data(), count);
		current.swap(temp);
	}
	for (int b = 0; b < batchSize; ++b) {
		float* curr = current.data() + b * count;
		float* out = outputs + b * count;
		float rms = 0.0f;
		for (int i = 0; i < count; ++i) {
			rms += curr[i] * curr[i];
		}
		rms = std::sqrt(rms / count + 1e-5f);
		float invRms = 1.0f / rms;
		float invRmsCubedDivCount = invRms * invRms * invRms / count;
		for (int i = 0; i < count; ++i) {
			float sumTerm = 0.0f;
			for (int j = 0; j < count; ++j) {
				sumTerm += curr[j] * curr[j] * gamma[j];
			}
			out[i] = gamma[i] * invRms - curr[i] * invRmsCubedDivCount * sumTerm;
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
HookDerivative ReLuGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = inputs[i] > 0.0f ? 1.0f : 0.0f;
	}
};
HookFunc SigmoidHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = 1.0f / (1.0f + std::exp(-inputs[i]));
	}
};
HookDerivative SigmoidGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		float sig = 1.0f / (1.0f + std::exp(-inputs[i]));
		outputs[i] = sig * (1.0f - sig);
	}
};
HookFunc TanhHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = std::tanh(inputs[i]);
	}
};
HookDerivative TanhGradHook = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		float t = std::tanh(inputs[i]);
		outputs[i] = 1.0f - t * t;
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

HookDerivative SoftmaxDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int originalBatchSize = batchSize / correctIndices.size();
	std::vector<float> softMaxes(batchSize * count);
	for (int i = 0; i < batchSize; ++i) {
		float* inp = inputs + i * count;
		float* out = softMaxes.data() + i * count; float maximum = 0.0f;
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
			float* inp  = softMaxes.data() + (i * originalBatchSize + b) * count;
			for (int j = 0; j < count; ++j) {
				if (j == idx) {
					outputs[(i * originalBatchSize + b) * count + j] = inp[j] * (1.0f - inp[j]);
				} else {
					outputs[(i * originalBatchSize + b) * count + j] = -inp[j] * inp[idx];
				}
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

HookDerivative EmbeddingDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	ParametricLayer* l = std::get<ParametricLayer*>(layer);
	int embeddingDim = count;
	float* gradients = l->gradients;
	for (int i = 0; i < batchSize; ++i) {
		int idx = static_cast<int>(layerInputs[i]);
		float* gradRow = gradients + idx * embeddingDim;
		float* upstreamRow = inputs + i * embeddingDim;
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

HookDerivative AttentionDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
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
	float* upstream_gradient = inputs;
	matmult(upstream_gradient, reusable_buffer, score_gradients, sequence_length, embedding_dim, sequence_length, false, true, 1.0f, 0.0f);
	for (int row = 0; row < sequence_length; ++row) {
		float weighted_sum = 0.0f;
		for (int col = 0; col < sequence_length; ++col) weighted_sum += score_gradients[row * sequence_length + col] * attention_scores[row * sequence_length + col];
		for (int col = 0; col < sequence_length; ++col) score_gradients[row * sequence_length + col] = attention_scores[row * sequence_length + col] * (score_gradients[row * sequence_length + col] - weighted_sum);
	}
	float* value_gradient = reusable_buffer;
	matmult(attention_scores, upstream_gradient, value_gradient, sequence_length, sequence_length, embedding_dim, true, false, 1.0f, 0.0f);
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
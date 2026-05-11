
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
	ParametricLayer* l = std::get<ParametricLayer*>(layer);
	int d_model = count;
	int N = batchSize;
	float* W = l->weightsBegin;
	float* QW = W;
	float* KW = W + d_model * d_model;
	float* VW = W + 2 * d_model * d_model;
	float* Q = l->scratchPad;
	float* K = Q + N * d_model;
	float* V = K + N * d_model;
	float* scores = V + N * d_model;
	matmult(inputs, QW, Q, N, d_model, d_model, false, false, 1.0f, 0.0f);
	matmult(inputs, KW, K, N, d_model, d_model, false, false, 1.0f, 0.0f);
	matmult(inputs, VW, V, N, d_model, d_model, false, false, 1.0f, 0.0f);
	float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
	matmult(Q, K, scores, N, d_model, N, false, true, scale, 0.0f);
	for (int i = 0; i < N; ++i) {
		float maxVal = scores[i * N];
		for (int j = 1; j < N; ++j) if (scores[i * N + j] > maxVal) maxVal = scores[i * N + j];
		float sumExp = 0.0f;
		for (int j = 0; j < N; ++j) sumExp += std::exp(scores[i * N + j] - maxVal);
		for (int j = 0; j < N; ++j) scores[i * N + j] = std::exp(scores[i * N + j] - maxVal) / sumExp;
	}
	matmult(scores, V, outputs, N, N, d_model, false, false, 1.0f, 0.0f);
};

HookDerivative AttentionDerivative = [](LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	ParametricLayer* l = std::get<ParametricLayer*>(layer);
	int d_model = count;
	int N = batchSize;
	float* X = layerInputs;
	float* W = l->weightsBegin;
	float* QW = W;
	float* KW = W + d_model * d_model;
	float* VW = W + 2 * d_model * d_model;
	float* Q = l->scratchPad;
	float* K = Q + N * d_model;
	float* V_reuse = K + N * d_model;
	float* scores = V_reuse + N * d_model;
	float* dScores_dZ = scores + N * N;
	matmult(X, QW, Q, N, d_model, d_model, false, false, 1.0f, 0.0f);
	matmult(X, KW, K, N, d_model, d_model, false, false, 1.0f, 0.0f);
	matmult(X, VW, V_reuse, N, d_model, d_model, false, false, 1.0f, 0.0f);
	float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
	matmult(Q, K, scores, N, d_model, N, false, true, scale, 0.0f);
	float* dY = inputs;
	matmult(dY, V_reuse, dScores_dZ, N, d_model, N, false, true, 1.0f, 0.0f);
	for (int i = 0; i < N; ++i) {
		float sum_ds_s = 0.0f;
		for (int j = 0; j < N; ++j) sum_ds_s += dScores_dZ[i * N + j] * scores[i * N + j];
		for (int j = 0; j < N; ++j) dScores_dZ[i * N + j] = scores[i * N + j] * (dScores_dZ[i * N + j] - sum_ds_s);
	}
	float* dV = V_reuse;
	matmult(scores, dY, dV, N, N, d_model, true, false, 1.0f, 0.0f);
	float* grads = l->gradients;
	matmult(X, dV, grads + 2 * d_model * d_model, d_model, N, d_model, true, false, 1.0f, 1.0f);
	matmult(dV, VW, outputs, N, d_model, d_model, false, true, 1.0f, 0.0f);
	float* dQ = V_reuse;
	matmult(dScores_dZ, K, dQ, N, N, d_model, false, false, scale, 0.0f);
	matmult(X, dQ, grads, d_model, N, d_model, true, false, 1.0f, 1.0f);
	matmult(dQ, QW, outputs, N, d_model, d_model, false, true, 1.0f, 1.0f);
	float* dK = V_reuse;
	matmult(dScores_dZ, Q, dK, N, N, d_model, true, false, scale, 0.0f);
	matmult(X, dK, grads + d_model * d_model, d_model, N, d_model, true, false, 1.0f, 1.0f);
	matmult(dK, KW, outputs, N, d_model, d_model, false, true, 1.0f, 1.0f);
};


#endif

#ifndef ACTIVATIONS_IMPORTED
#define ACTIVATIONS_IMPORTED
#include "../model/network.h"
#include <cmath>
HookFunc ReLuHook = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = inputs[i] > 0.0f ? inputs[i] : 0.0f;
	}
};
HookDerivative ReLuGradHook = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = inputs[i] > 0.0f ? 1.0f : 0.0f;
	}
};
HookFunc SigmoidHook = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = 1.0f / (1.0f + std::exp(-inputs[i]));
	}
};
HookDerivative SigmoidGradHook = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		float sig = 1.0f / (1.0f + std::exp(-inputs[i]));
		outputs[i] = sig * (1.0f - sig);
	}
};
HookFunc TanhHook = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		outputs[i] = std::tanh(inputs[i]);
	}
};
HookDerivative TanhGradHook = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices) {
	int total = count * batchSize;
	for (int i = 0; i < total; ++i) {
		float t = std::tanh(inputs[i]);
		outputs[i] = 1.0f - t * t;
	}
};
HookFunc Softmax = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count) {
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

HookDerivative SoftmaxDerivative = [](LayerRef layer, int batchSize, float* inputs, float* outputs, int count, const std::vector<int> correctIndices) {
	// calculate Jacobian for correct indices only.
	// note: in this case we must split inputs into batchSize / correctIndices.size() groups
	int originalBatchSize = batchSize / correctIndices.size();
	// previous softMaxes
	vector<float> softMaxes(batchSize * count);
	for (int i = 0; i < batchSize; ++i) {
		float* inp = inputs + i * count;
		float* out = softMaxes.data() + i * count;		float maximum = 0.0f;
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


#endif


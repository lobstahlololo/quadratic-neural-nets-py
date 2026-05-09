#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include "network.h"
#include "../math/math.h"


bool setuped = false;
int batchSize = 1;
int networkSize = 0;
std::pair<std::vector<float>, std::vector<float>> allocatedOutputs;
std::vector<float> allocatedSquaredInputs;
std::vector<float> weights;
std::vector<float> globalScratch;
std::vector<Layer> layers;
std::vector<int> perLayerSize;
float activationBuffer[16834];
#ifdef TRAINING_ON
std::vector<float> globalInputs;
std::vector<float> globalPreacts;
std::vector<float> globalGrads;

#endif
float* Layer::forward(float* inputs, int tempBatchSize) {
	int effectiveBatch = batch * tempBatchSize;
	if (size == 0) {
		#ifdef TRAINING_ON
		previous_inputs = inputs;
		previous_preacts = inputs;
		#endif
		if (forwardHooks.empty()) {
			return inputs;
		}
		LayerRef self(this);
		std::copy(inputs, inputs + output * effectiveBatch, allocatedOutputs.second.data());
		for (auto& hook : forwardHooks) {
			hook(self, effectiveBatch, inputs, allocatedOutputs.second.data(), allocatedOutputs.second.data(), output);
		}
		std::copy(allocatedOutputs.second.data(), allocatedOutputs.second.data() + output * effectiveBatch, allocatedOutputs.first.data());
		if (output_ptr) {
			std::copy(allocatedOutputs.second.data(), allocatedOutputs.second.data() + output * effectiveBatch, output_ptr);
		}
		std::fill(allocatedOutputs.second.data(), allocatedOutputs.second.data() + output * effectiveBatch, 0.0f);
		return allocatedOutputs.first.data();
	}

	float* in_ptr = inputs;
	for (std::size_t i = 0; i < input * effectiveBatch; ++i) {
		allocatedSquaredInputs[i] = in_ptr[i] * in_ptr[i];
	}

	matmult(linear(), inputs, allocatedOutputs.second.data(), output, input, effectiveBatch);
	matmult(quadratic(), allocatedSquaredInputs.data(), allocatedOutputs.second.data(), output, input, effectiveBatch);
	for (std::size_t i = 0; i < output * effectiveBatch; ++i) {
		allocatedOutputs.second[i] += biases()[i % output];
	}
	if (!forwardHooks.empty()) {
		LayerRef self(this);
		std::copy(allocatedOutputs.second.data(), allocatedOutputs.second.data() + output * effectiveBatch, previous_preacts);
		for (auto& hook : forwardHooks) {
			hook(self, effectiveBatch, inputs, allocatedOutputs.second.data(), allocatedOutputs.second.data(), output);
		}
	}
	#ifdef TRAINING_ON
	std::copy(allocatedOutputs.second.data(), allocatedOutputs.second.data() + output * effectiveBatch, output_ptr);
	#endif
	std::copy(allocatedOutputs.second.data(), allocatedOutputs.second.data() + output * effectiveBatch, allocatedOutputs.first.data());
	std::fill(allocatedOutputs.second.data(), allocatedOutputs.second.data() + output * effectiveBatch, 0.0f);
	return allocatedOutputs.first.data();
}

float* ParametricLayer::forward(float* inputs, int tempBatchSize) {
	int effectiveBatch = batch * tempBatchSize;
	if (forwardHooks.empty()) {
		return inputs;
	}
	LayerRef self(this);
		std::copy(inputs, inputs + input * effectiveBatch, allocatedOutputs.first.data());
		for (auto& hook : forwardHooks) {
			hook(self, effectiveBatch, inputs, allocatedOutputs.first.data(), allocatedOutputs.first.data(), input);
		}
		if (output_ptr) {
			std::copy(allocatedOutputs.first.data(), allocatedOutputs.first.data() + input * effectiveBatch, output_ptr);
		}
		return allocatedOutputs.first.data();
}
#ifdef TRAINING_ON
float* Layer::backward(float* upstream_grad, const std::vector<int>& correctIndices, int tempBatchSize) {
	int effectiveBatch = tempBatchSize;
	if (!forwardHookDerivatives.empty()) {
		LayerRef self(this);
		for (auto& derivHook : forwardHookDerivatives) {
			derivHook(self, effectiveBatch, previous_inputs, previous_preacts, activationBuffer, output, correctIndices);
			for (size_t i = 0; i < output * effectiveBatch; ++i) {
				upstream_grad[i] *= activationBuffer[i];
			}
		}
	}

	size_t totalWeights = weightCount();
	size_t bias_offset = totalWeights;
	std::fill(gradients + bias_offset, gradients + size, 0.0f);
	for (size_t b = 0; b < effectiveBatch; ++b) {
		for (size_t n = 0; n < neurons; ++n) {
			gradients[bias_offset + n] += upstream_grad[b * neurons + n];
		}
	}

	float* outputData = allocatedOutputs.first.data();
	float* scratchData = allocatedOutputs.second.data();

	matmult(upstream_grad, previous_inputs, gradients, output, effectiveBatch, input);
	matmult(upstream_grad, allocatedSquaredInputs.data(), gradients + input * output, output, effectiveBatch, input);

	matmult(upstream_grad, linear(), scratchData, effectiveBatch, output, input);
	for (size_t i = 0; i < input * effectiveBatch; ++i) {
		outputData[i] = scratchData[i];
		scratchData[i] = 0.0f;
	}

	matmult(upstream_grad, quadratic(), scratchData, effectiveBatch, output, input);
	for (size_t i = 0; i < input * effectiveBatch; ++i) {
		outputData[i] += 2.0f * previous_inputs[i] * scratchData[i];
	}

	return outputData;
}
float* ParametricLayer::backward(float* upstream_grad, const std::vector<int>& correctIndices, int tempBatchSize) {
	int effectiveBatch = tempBatchSize;
	size_t weight_count = input * weightsPerInput;
	std::fill(gradients, gradients + weight_count, 0.0f);
	if (!forwardHookDerivatives.empty()) {
		LayerRef self(this);
		for (auto& derivHook : forwardHookDerivatives) {
			derivHook(self, effectiveBatch, previous_inputs, previous_inputs, activationBuffer, input, correctIndices);
			for (size_t i = 0; i < input * effectiveBatch; ++i) {
				upstream_grad[i] *= activationBuffer[i];
			}
		}
	}
	if (scratchSize == 0) {
		for (size_t i = 0; i < input * effectiveBatch; ++i) {
			size_t f = i % input;
			for (size_t j = 0; j < weightsPerInput; ++j) {
				gradients[f * weightsPerInput + j] += upstream_grad[i * weightsPerInput + j];
			}
		}
	}
	for (size_t i = 0; i < input * effectiveBatch * weightsPerInput; ++i) {
		allocatedOutputs.first[i] = upstream_grad[i];
	}
	return allocatedOutputs.first.data();
}
#endif
void setupNeuralNetwork(std::vector<LayerArgs> layersAdd, std::string weightsPath) {
	// rely on callrr to chrck setuped
	layers.clear();
	weights.clear();
	perLayerSize.clear();
	int maxNeurons = 0;
	networkSize = 0;
	int currentBatch = batchSize;
	int maxBatch = batchSize;
	int maxBufferSize = 0;
	LayerArgs* prev = nullptr;
	for (auto& args : layersAdd) {
		maxNeurons = std::max(maxNeurons, args.layerSize);
		int layerBatch = currentBatch;
		maxBufferSize = std::max(maxBufferSize, args.layerSize * layerBatch);
		if (prev == nullptr) {
			perLayerSize.push_back(args.layerSize);
			prev = &args;
			currentBatch *= args.outputsPerNeuron;
			maxBatch = std::max(maxBatch, currentBatch);
			continue;
		}
		currentBatch *= args.outputsPerNeuron;
		maxBatch = std::max(maxBatch, currentBatch);
		if (args.kind == Parametric) {
			networkSize += prev->layerSize * args.weightsPerInput;
		} else {
			networkSize += args.layerSize + args.layerSize * prev->layerSize * 2 * args.outputsPerNeuron;
		}
		perLayerSize.push_back(args.layerSize);
		prev = &args;
	}
	auto resizeBuffers = [&](size_t size) {
		allocatedOutputs.first.resize(size);
		allocatedOutputs.second.resize(size);
		allocatedSquaredInputs.resize(size);
	};
	resizeBuffers(maxBufferSize);
	size_t totalScratch = 0;
	for (auto& args : layersAdd) totalScratch += args.scratchSize;
	globalScratch.resize(totalScratch);
	#ifdef TRAINING_ON
	globalInputs.resize(maxNeurons * maxNeurons * maxBatch);
	globalPreacts.resize(maxNeurons * maxNeurons * maxBatch);
	globalGrads.resize(networkSize);
	#endif
	if (weightsPath == "UNDEFINED276lineosersyoujelly?") {
		weights.resize(networkSize);
	} else {
		std::ifstream file(weightsPath, std::ios::binary);
		if (!file) {
			throw std::runtime_error("wrong filepath");
		}
		file.seekg(0, std::ios::end);
		std::streampos fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		if (static_cast<long>(networkSize * sizeof(float)) != fileSize) {
			throw std::runtime_error("wrong file size relation to weights");
		}
		weights.resize(networkSize);
		file.read(reinterpret_cast<char*>(weights.data()), networkSize * sizeof(float));
	}
	float* weightStart = weights.data();
	std::size_t prevOutputOffset = 0;
	std::size_t accumulatedWeights = 0;
	std::size_t gradOffset = 0;
	std::size_t scratchOffset = 0;
	int accumulatedBatch = batchSize;
	for (std::size_t i = 0; i < layersAdd.size(); ++i) {
		Layer layer;
		for (auto& h : layersAdd[i].hooks) {
			layer.forwardHooks.push_back(h);
		}
		for (auto& hd : layersAdd[i].hookGrads) {
			layer.forwardHookDerivatives.push_back(hd);
		}
		layer.neurons = layersAdd[i].layerSize;
		if (i == 0) {
			layer.input = layersAdd[i].layerSize;
			layer.output = layersAdd[i].layerSize;
			layer.size = 0;
			layer.weightsBegin = weightStart;
		} else {
			layer.input = layersAdd[i-1].layerSize;
			layer.output = layersAdd[i].layerSize;
			if (layersAdd[i].kind == Parametric) {
				layer.size = layer.input * layersAdd[i].weightsPerInput;
				layer.weightsBegin = weightStart + accumulatedWeights;
			} else {
				layer.size = layersAdd[i].layerSize + layersAdd[i].layerSize * layersAdd[i-1].layerSize * 2 * layersAdd[i].outputsPerNeuron;
				layer.weightsBegin = weightStart + accumulatedWeights;
			}
		}
		layer.batch = accumulatedBatch;
		accumulatedBatch *= layersAdd[i].outputsPerNeuron;
		size_t outputDataSize = layer.output * layer.batch * layersAdd[i].outputsPerNeuron;
		layer.scratchSize = layersAdd[i].scratchSize;
		layer.scratchPad = layer.scratchSize > 0 ? globalScratch.data() + scratchOffset : nullptr;
		scratchOffset += layer.scratchSize;
		#ifdef TRAINING_ON
		layer.previous_preacts = globalPreacts.data() + prevOutputOffset;
		layer.gradients = globalGrads.data() + gradOffset;
		layer.output_ptr = globalInputs.data() + prevOutputOffset;
		if (i > 0) {
			layer.previous_inputs = layers.back().output_ptr;
		} else {
			layer.previous_inputs = nullptr;
		}
		#endif
		layers.push_back(layer);
		prevOutputOffset += outputDataSize;
		if (i > 0) {
			gradOffset += layer.size;
			accumulatedWeights += layer.size;
		}
	}
	setuped = true;
}
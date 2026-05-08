#include "train.h"
#include "../model/network.h"
#include <cmath>

using std::vector;

vector<float> firstMoment;
vector<float> secondMoment;




void train(std::vector<Layer>& layers, const std::vector<float>& trainingData, const std::vector<std::vector<int>>& requiredIndices, const std::vector<float>& requiredOutput, float learningRate, const LossFunc& lossFunction, const LossDerivative& lossDerivative, int step) {
	float* allocatedInputPtr = allocatedOutputs.first.data(); 
	for (int i = 0; i < layers.size(); ++i) {
		allocatedInputPtr = layers[i].forward(allocatedInputPtr);
	}
	vector<float> losses(batchSize);
	for (int i = 0; i < batchSize; ++i) {
		losses[i] = lossFunction(allocatedOutputs.first.data() + layers.back().output * i, requiredOutput.data() + layers.back().output * i, requiredIndices[i], layers.back().output);
	}
	
	
	vector<float> downstream(layers.back().output * batchSize);
	// rebatch
	
	for (int i = 0; i < batchSize; ++i) {
		lossDerivative(losses[i], allocatedOutputs.first.data() + layers.back().output * i, requiredOutput.data() + layers.back().output * i, requiredIndices[i], downstream.data() + layers.back().output * i, layers.back().output);
	}
	
	for (int i = 0; i < layers.size(); ++i) {
		std::fill(layers[i].gradients, layers[i].gradients + layers[i].size, 0.0f);
	}
	float* downstreamPtr = downstream.data();
	for (int b = 0; b < batchSize; ++b) {
		int tempBatchSize = requiredIndices[b].size();
		float* downstreamSamplePtr = downstreamPtr + b * layers.back().output;
		const std::vector<int>& sampleCorrectIndices = requiredIndices[b];
		float* samplePtr = downstreamSamplePtr;
		for (int i = layers.size() - 1; i >= 0; --i) {
			samplePtr = layers[i].backward(samplePtr, sampleCorrectIndices, tempBatchSize);
		}
	}

	// now everything is stored in layers.gradients 
	float scaledLR = learningRate / batchSize;
	float mPow = pow(0.9f, step + 1);
	float vPow = pow(0.999f, step + 1);
	size_t momentOffset = 0;
	for (int i = 0; i < layers.size(); ++i) {
		const Layer& layer = layers[i];
		for (int j = 0; j < layer.size; ++j) {
			float grad = layer.gradients[j];
			firstMoment[momentOffset + j] = 0.9f * firstMoment[momentOffset + j] + 0.1f * grad;
			secondMoment[momentOffset + j] = 0.999f * secondMoment[momentOffset + j] + 0.001f * grad * grad;
			float firstBiasCorr = firstMoment[momentOffset + j] / (1 - mPow);
			float secondBiasCorr = secondMoment[momentOffset + j] / (1 - vPow);
			layer.weightsBegin[j] -= scaledLR * firstBiasCorr / (sqrt(secondBiasCorr) + 1e-8f);
		}
		momentOffset += layer.size;
	}

}



void trainScheduler(std::vector<Layer>& layers, const std::vector<float>& trainingData, const std::vector<std::vector<int>>& requiredIndices, std::vector<float> requiredOutput, float learningRate, float minLearnRate, const LossFunc& lossFunction, const LossDerivative& lossDerivative, int epochs, int batch) {
	firstMoment.resize(networkSize);
	secondMoment.resize(networkSize);
	batchSize = batch;
	int inputSize = layers[0].input;
	int outputSize = layers.back().output;
	int stepsPerEpoch = trainingData.size() / (batch * inputSize);
	for (int i = 0; i < epochs; ++i) {
		for (int j = 0; j < stepsPerEpoch; ++j) {
			float currentLR = minLearnRate + (learningRate - minLearnRate) * (1 + cos(3.14159265f * j / stepsPerEpoch)) / 2;
			std::vector<float> batchData(trainingData.begin() + j * batchSize * inputSize, trainingData.begin() + (j + 1) * batchSize * inputSize);
			std::vector<float> batchRequired(requiredOutput.begin() + j * batchSize * outputSize, requiredOutput.begin() + (j + 1) * batchSize * outputSize);
			int step = i * stepsPerEpoch + j;
			train(layers, batchData, requiredIndices, batchRequired, currentLR, lossFunction, lossDerivative, step);
		}
	}
}
	
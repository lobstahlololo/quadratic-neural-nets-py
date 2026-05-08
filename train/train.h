#ifndef TRAINING_ON
#include "../model/network.h"
#include <vector>
#define TRAINING_ON

typedef float (*LossFunc)(const float* predicted,
		const float* required,
		const std::vector<int>& requiredIndices,
		int size);

typedef void (*LossDerivative)(
		float loss,
		const float* predicted,
		const float* required,
		const std::vector<int>& requiredIndices,
		float* output,
		int size);

void train(std::vector<Layer>& layers, const std::vector<float>& trainingData, const std::vector<std::vector<int>>& requiredIndices, const std::vector<float>& requiredOutput, float learningRate, const LossFunc& lossFunction, const LossDerivative& lossDerivative, int step);

void trainScheduler(std::vector<Layer>& layers, const std::vector<float>& trainingData, const std::vector<std::vector<int>>& requiredIndices, std::vector<float> requiredOutput, float learningRate, float minLearnRate, const LossFunc& lossFunction, const LossDerivative& lossDerivative, int epochs=10, int batch=4); 

#endif
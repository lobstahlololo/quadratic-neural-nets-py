#ifndef TRAINING_ON
#include "../model/network.h"
#include <vector>
#define TRAINING_ON
typedef float (*LossFunc)(const float* predicted,
		const float* required,
		const std::vector<int>& required_indices,
		int size);
typedef void (*LossDerivative)(
		float loss,
		const float* predicted,
		const float* required,
		const std::vector<int>& required_indices,
		float* output,
		int size);
void train(std::vector<Layer>& layers, const std::vector<float>& training_data, const std::vector<std::vector<int>>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step);
void trainScheduler(std::vector<Layer>& layers, const std::vector<float>& training_data, const std::vector<std::vector<int>>& correct_indices, std::vector<float> required_output, float learning_rate, float minimum_learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int total_epochs=10, int batch_size=4); 
#endif

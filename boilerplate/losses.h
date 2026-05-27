#ifndef BOILERPLATE_LOSSES_H
#define BOILERPLATE_LOSSES_H
#include <vector>
#include <cmath>
inline float MeanSquaredErrorLoss(const float* predicted, const float* required, const std::vector<int>& required_indices, int size) {
	float sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		float diff = predicted[i] - required[i];
		sum += diff * diff;
	}
	return sum / size;
}
inline void MeanSquaredErrorLossDerivative(float loss, const float* predicted, const float* required, const std::vector<int>& required_indices, float* output, int size) {
	float scale = 2.0f / size;
	for (int i = 0; i < size; ++i) {
		output[i] = scale * (predicted[i] - required[i]);
	}
}
inline float CrossEntropyLoss(const float* predicted, const float* required, const std::vector<int>& required_indices, int size) {
	float total_loss = 0.0f;
	int batch_size = required_indices.size();
	for (int i = 0; i < batch_size; ++i) {
		int target = required_indices[i];
		total_loss += -std::log(predicted[i * size + target] + 1e-7f);
	}
	return total_loss / batch_size;
}
inline void CrossEntropyLossDerivative(float loss, const float* predicted, const float* required, const std::vector<int>& required_indices, float* output, int size) {
	int batch_size = required_indices.size();
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < size; ++j) {
			output[i * size + j] = predicted[i * size + j];
		}
		output[i * size + required_indices[i]] -= 1.0f;
	}
	float inv_batch = 1.0f / batch_size;
	for (int i = 0; i < batch_size * size; ++i) {
		output[i] *= inv_batch;
	}
}
inline float CrossEntropyLossForSoftmax(const float* predicted, const float* required, const std::vector<int>& required_indices, int size) {
	float total_loss = 0.0f;
	int batch_size = required_indices.size();
	for (int i = 0; i < batch_size; ++i) {
		int target = required_indices[i];
		total_loss += -std::log(predicted[i * size + target] + 1e-7f);
	}
	return total_loss / batch_size;
}
inline void CrossEntropyLossForSoftmaxDerivative(float loss, const float* predicted, const float* required, const std::vector<int>& required_indices, float* output, int size) {
	int total = size * required_indices.size();
	for (int i = 0; i < total; ++i) {
		output[i] = predicted[i];
	}
}
#endif
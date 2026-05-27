#ifdef TRAINING_ON
#include "../model/network.h"
#include "../boilerplate/losses.h"
#include <vector>
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
float train(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& batch_sizes);
void trainScheduler(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, std::vector<float> required_output, float learning_rate, float minimum_learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int total_epochs=10, int batch_size=4, std::vector<int> initial_batch_sizes={}); 
#endif
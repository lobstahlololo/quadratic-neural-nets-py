#ifndef QQ_TRAIN_LOADED
#define QQ_TRAIN_LOADED
typedef void (*TrainHook)();
extern TrainHook trainHook;
#ifdef TRAINING_ON
#include "../model/network.h"
#include "../boilerplate/losses.h"
#include <vector>

extern std::vector<float> first_moment_buffer;
extern std::vector<float> second_moment_buffer;

// Per-slot LR/weight-decay scales applied to the quadratic weight block of
// Quadratic layers in train_adams (defined in boilerplate/train_functions.h).
// Default 1.0 = no scaling.
extern float quad_lr_scale;
extern float quad_wd_scale;

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
typedef float (*TrainFunction)(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& sequence_lengths);
float train(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& sequence_lengths);
void trainScheduler(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, std::vector<float> required_output, float learning_rate, float minimum_learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int total_epochs, int batch_size, std::vector<int> initial_sequence_lengths, TrainFunction train_func = train); 
#endif
#endif
#ifndef Q2_NN
#define Q2_NN
#include <vector>
#include <functional>
#include <string>
#define TRAINING_ON
#include <variant>
typedef std::variant<struct Layer*, struct ParametricLayer*> LayerRef;
typedef void (*HookFunc)(LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* output_values, int feature_count);
typedef void (*HookDerivative)(LayerRef layer, int batch_count, std::vector<int>& batch_sizes, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices);
typedef void (*WeightInitFunc)(float* weights, int total_size, const std::vector<LayerArgs>& layers);
extern bool is_setup;
extern int network_size;
extern int batch_size;
extern std::pair<std::vector<float>, std::vector<float>> output_buffers;
struct Layer {
	float* weights_begin;
	size_t input;
	size_t output;
	size_t size;
	size_t neurons;
	float* scratch_pointer;
	int scratch_size;
	std::vector<int> extra_args;
	std::vector<HookFunc> forward_hooks;
	std::vector<HookDerivative> forward_hook_derivatives;
	float* quadratic() { return weights_begin; }
	float* linear() { return weights_begin + (size - neurons) / 2; }
	float* biases() { return weights_begin + (size - neurons); }
	size_t weight_count() const { return input * output * 2; }
	float* forward(float* inputs, int batch_count, std::vector<int>& batch_sizes);
	#ifdef TRAINING_ON
	float* previous_inputs;
	float* previous_preactivations;
	float* weight_gradients;
	float* output_pointer;
	float* backward(float* upstream_gradient, int batch_count, std::vector<int>& batch_sizes, const std::vector<int>& correct_indices);
	#endif
};
struct ParametricLayer {
	float* weights_begin;
	size_t input;
	size_t output;
	int weights_per_input;
	float* scratch_pointer;
	int scratch_size;
	std::vector<int> extra_args;
	std::vector<HookFunc> forward_hooks;
	std::vector<HookDerivative> forward_hook_derivatives;
	float* forward(float* inputs, int batch_count, std::vector<int>& batch_sizes);
	#ifdef TRAINING_ON
	float* previous_inputs;
	float* weight_gradients;
	float* output_pointer;
	float* backward(float* upstream_gradient, int batch_count, std::vector<int>& batch_sizes, const std::vector<int>& correct_indices);
	#endif
};
enum LayerKind { Quadratic, Parametric };
struct LayerArgs {
	int layer_size;
	std::vector<HookFunc> hooks;
	std::vector<HookDerivative> hook_gradients;
	LayerKind kind;
	int weights_per_input = 1;
	int scratch_size = 0;
	std::vector<int> extra_args;
};
void xavier_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void he_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void uniform_random_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void zero_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void setupNeuralNetwork(std::vector<LayerArgs> layers, std::string weights_path = "", WeightInitFunc initialiser = xavier_initialisation, int max_sequence_length = 1);
#endif
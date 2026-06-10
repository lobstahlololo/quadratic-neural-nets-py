#ifndef Q2_NN
#define Q2_NN
#include <vector>
#include <functional>
#include <string>
#include <variant>
struct LayerArgs;
typedef std::variant<struct Layer*, struct ParametricLayer*> LayerRef;
typedef void (*HookFunc)(LayerRef layer, int batch_count, std::vector<int>& sequence_lengths, float* original_inputs, float* preactivation_values, float* output_values, int feature_count);
typedef void (*HookDerivative)(LayerRef layer, int batch_count, std::vector<int>& sequence_lengths, float* original_inputs, float* preactivation_values, float* upstream_gradient, float* output_gradient, int feature_count, const std::vector<int>& correct_indices);
typedef void (*WeightInitFunc)(float* weights, int total_size, const std::vector<LayerArgs>& layers);
typedef void (*setupNNHookFunction)();
extern bool is_setup;
extern int network_size;
extern int batch_size;
extern std::pair<std::vector<float>, std::vector<float>> output_buffers;
extern std::vector<std::variant<Layer, ParametricLayer>> layers;
struct Layer {
	float* weights_begin = nullptr;
	size_t input = 0;
	size_t output = 0;
	size_t size = 0;
	size_t neurons = 0;
	float* scratch_pointer = nullptr;
	int scratch_size = 0;
	float* extra_weights_begin = nullptr;
	int extra_weights_size = 0;
	std::vector<int> extra_args;
	std::vector<HookFunc> forward_hooks;
	std::vector<HookDerivative> forward_hook_derivatives;
	float* quadratic() { return weights_begin; }
	float* linear() { return weights_begin + (size - neurons) / 2; }
	float* biases() { return weights_begin + (size - neurons); }
	size_t weight_count() const { return input * output * 2; }
	float* forward(float* inputs, int batch_count, std::vector<int>& sequence_lengths);
	#ifdef TRAINING_ON
	float* previous_inputs = nullptr;
	float* previous_preactivations = nullptr;
	float* weight_gradients = nullptr;
	float* extra_weight_gradients = nullptr;
	float* output_pointer = nullptr;
	float* backward(float* upstream_gradient, int batch_count, std::vector<int>& sequence_lengths, const std::vector<int>& correct_indices);
	#endif
};
struct ParametricLayer {
	float* weights_begin = nullptr;
	size_t input = 0;
	size_t output = 0;
	int weights_per_input = 0;
	float* scratch_pointer = nullptr;
	int scratch_size = 0;
	float* extra_weights_begin = nullptr;
	int extra_weights_size = 0;
	std::vector<int> extra_args;
	std::vector<HookFunc> forward_hooks;
	std::vector<HookDerivative> forward_hook_derivatives;
	float* forward(float* inputs, int batch_count, std::vector<int>& sequence_lengths);
	#ifdef TRAINING_ON
	float* previous_inputs = nullptr;
	float* previous_preactivations = nullptr;
	float* weight_gradients = nullptr;
	float* extra_weight_gradients = nullptr;
	float* output_pointer = nullptr;
	float* backward(float* upstream_gradient, int batch_count, std::vector<int>& sequence_lengths, const std::vector<int>& correct_indices);
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
	int extra_weights = 0;
	std::vector<int> extra_args;
};
void xavier_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void he_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void uniform_random_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void zero_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers);
void setupNeuralNetwork(std::vector<LayerArgs> layers, std::string weights_path = "", WeightInitFunc initialiser = xavier_initialisation, int buffer_multiplier = 1, setupNNHookFunction setup_hook = nullptr);
void save_weights(std::string path, bool compress = false);
#endif
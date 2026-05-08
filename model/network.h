#ifndef Q2_NN
#define Q2_NN
#include <vector>
#include <functional>
#include <string>
#define TRAINING_ON
#include <variant>
typedef std::variant<struct Layer*, struct ParametricLayer*> LayerRef;
typedef void (*HookFunc)(LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count);
typedef void (*HookDerivative)(LayerRef layer, int batchSize, float* layerInputs, float* inputs, float* outputs, int count, const std::vector<int>& correctIndices);
extern bool setuped;
extern int networkSize;
extern int batchSize;
extern std::pair<std::vector<float>, std::vector<float>> allocatedOutputs;

struct Layer {
	float* weightsBegin;
	size_t input;
	size_t output;
	size_t size;
	size_t neurons;
	int outputsPerNeuron;
	int batch;
	std::vector<HookFunc> forwardHooks;
	std::vector<HookDerivative> forwardHookDerivatives;
	float* quadratic() { return weightsBegin; }
	float* linear() { return weightsBegin + (size - neurons) / 2; }
	float* biases() { return weightsBegin + (size - neurons); }
	size_t weightCount() const { return input * output * 2; }
	float* forward(float* inputs, int tempBatchSize = 1);
	#ifdef TRAINING_ON
	float* previous_inputs;
	float* previous_preacts;
	float* gradients;
	float* output_ptr;
	float* backward(float* upstream_grad, const std::vector<int>& correctIndices, int tempBatchSize = 1);
	#endif
};

struct ParametricLayer {
	float* weightsBegin;                                    size_t input;                                           size_t output;
        int batch;
        int outputsPerNeuron;
	std::vector<HookFunc> forwardHooks;
	std::vector<HookDerivative> forwardHookDerivatives;
	float* forward(float* inputs, int tempBatchSize = 1);
	#ifdef TRAINING_ON
	float* previous_inputs;
	float* gradients;
	float* output_ptr;
	float* backward(float* upstream_grad, const std::vector<int>& correctIndices, int tempBatchSize = 1);
	#endif
};
enum LayerKind { Quadratic, Parametric };

struct LayerArgs {
	int layerSize;
	std::vector<HookFunc> hooks;
	std::vector<HookDerivative> hookGrads;
	LayerKind kind;
	int outputsPerNeuron;
	int weightsPerInput = 1;
};
void setupNeuralNetwork(std::vector<LayerArgs> layers, std::string weightsPath);
#endif
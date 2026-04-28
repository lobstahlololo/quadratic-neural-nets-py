#ifndef Q2_NN

#define Q2_NN

#define TRAINING_ON

typedef float (*HookFunc)(float);
struct Layer {
	float* weightsBegin;
	float* current;
	size_t input;
	size_t output;
	size_t size;
	// Hooks allow customizability easily. deriv is for backward pass
	HookFunc forwardHook;
	HookFunc forwardHookDerivative;
	float* quadratic() {return weightsBegin;};
	float* linear() {return weightsBegin+(size/3);};
	float* biases() {return weightsBegin+(size/3*2);};

	float* forward(float* inputs);
	#ifdef TRAINING_ON
	float* previous_inputs;
	float* gradients;

	float* backward(float* upstream_grad) {
	
	}
	#endif
}

struct LayerArgs {
        int layerSize;
        HookFunc hook;
        hookFunc hookGrad;

}
void setupNeuralNetwork(std::vector<LayerArgs> layers, std::string weightsPath) (

}

#endif


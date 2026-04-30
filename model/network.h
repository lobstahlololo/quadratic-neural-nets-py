#ifndef Q2_NN

#define Q2_NN

#define TRAINING_ON

typedef float (*HookFunc)(float);
strucit Layer {
	float* weightsBegin;
	float* current;
	size_t input;
	size_t output;
	size_t size;
	size_t neurons;
	// Hooks allow customizability easily. deriv is for backward pass
	HookFunc forwardHook;
	HookFunc forwardHookDerivative;
	float* quadratic() {return weightsBegin;};
	float* linear() {return weightsBegin+(size-neurons) /2;};
	float* biases() {return weightsBegin+(size-neurons);};

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


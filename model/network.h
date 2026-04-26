#ifndef Q2_NN

#define Q2_NN

#define TRAINING_ON

typedef float (*HookFunc)(float);
struct Layer {
	float* weightsBegin;
	float* current;
	size_t input;
	size_t output;
	// Hooks allow customizability easily. deriv is for backward pass
	HookFunc forwardHook;
	HookFunc forwardHookDerivative;
	float* quadratic() {return weightsBegin;};
	float* inear() {return weightsBegin+(input*output);};
	float* biases() {return weightsBegin+(input*output*2);};
	float* size() {return input*output;};
	void forward(float* inputs);
	#ifdef TRAINING_ON
	float* previous_inputs;
	float* gradients;

	void backward(float* upstream_grad) {
	
	}
	#endif
}

#endif


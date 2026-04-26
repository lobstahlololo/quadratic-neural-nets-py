#ifndef Q2_NN

#define Q2_NN

#define TRAININF

struct Layer {
	float* weightsBegin;
	float* current;
	size_t input;
	size_t output;
	float* quadratic() {return weightsBegin;};
	float* inear() {return weightsBegin+(input*output);};
	float* biases() {return weightsBegin+(input*output*2);};
	float* size() {return input*output;};
	void forward(float* inputs);
}

#endif


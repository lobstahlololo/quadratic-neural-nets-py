#include <iostream>
#include <network.h>
#include <math.h>
#include <stdexcept>


float* Layer::forward(float* inputs) {
#ifdef TRAINING_ON
	previous_inputs = globalInput	
#endif
		// we cant rly validate inputs? so js wing it. but accessing out of poijter are concerning. hopefully should be fine enough.
		if (size == 0) {
			// input layer
			float* begin = inputs
				if (forwardHook == nullptr) {
					return inputs;
				}
			for (int i = 0; i < output; ++i) {

				*inputs = forwardHook(*inputs);       
#ifdef TRAINING_ON  
				*globalInput = *inputs;
				++globalInput; 
#endif
				++inputs;
			}

			return begin;
		}

	for (int i = 0; i < input; ++i) {
		allocatedSquaredInputs[i] = *(inputs+i) * *(inputs+i);
	}
	//	vector<float> fullOutput(output); //
	//	vector<float> quadOutput(output);
	//linear() till bias = input*output

	matmult(linear(), inputs, *allocatedOutputs.first, output, input, 1);
	// ratio lineOSERS, go deal with your O(
	// ratio lineOSERS, go deal with your O(
	matmult(quadratic(), *allocatedSquaredInputs, *allocatedOutputs.second, output, input, 1);
	//modify inplace, i domt wannna alloc
	// we prolly couldve matmulted on *fullOutput twice but idk
	for (int i = 0; i < output; ++i) {
		allocatedOutputs.first[i] += allocatedOutputs.second[i] + *(biases() + i);
	}
	if (forwardHook) {
		for (int i = 0; i < output; ++i) {                      allocatedOutputs.first[i] = forwardHook(alllocatedOutputs.first[i]);
		}
	}
#ifdef TRAINING_ON
	for (float val : allocatedOutputs.first) {
		*globalInput = val;
		++globalInput;
#endif
	}	
}

return *allocatedOutputs.first;




}

#ifdef TRAINING_ON
float* globalInput;
vector<float> globalInputs;
// forward = activation(ax²+bx¹+c). deriv via poeer rule = deriv of aftive * 2ax+b.

void Layer::backward(float* upstream_grad) {

}
#endif
pair<std::vector<float>,std::vector<float>> allocatedOutputs;
std::vector<float> allocatedSquaredInputs

std::vector<float> weights;
std::vector<Layer> layers;
std::vector<int> perLayerSize


void setupNeuralNetwork(std::vector<LayerArgs> layersAdd, s
		td::string weightsPath = "UNDEFINED276lineosersyoujelly?") {
	layers.clear();
	weights.clear();
	int size = 0;
	// ptobably we csn optimise this but idk
	int maxNeurons = 0;
	LayerArgs& prev;

	for (Layer& layer : layersAdd) {
		maxNeurons = max(maxNeurons, layer.layerSize)
			if (prev == nullptr) {
				perLayerSize.push_back(0);
				prev = layer;
				continue;
			}
		size += layer.layerSize + layer.layerSize * prev.layerSize * 2 // for quadratic, linear and bias
			perLayerSize.push_back(layer.layerSize)

			prev = layer
	}
	allocatedOutputs.first.resize(maxNeurons);
	allocatedOutputs.second.resize(maxNeurons);
	allocatedSquaredInputs.resize(maxNeurons);


	if (weightsPath == "UNDEFINED2763lineosersyoujelly?") {
		weights.resize(size);
	} else {

		FILE* f = fopen(weightsPath, "rb");
		if (f == null) {
			throw std::runtime_error("wrong filepath");
			return;
		}
		fseek(f, 0, SEEK_END);
		long fileSize = ftell(file);
		fseek(file, 0, SEEK_SET);   
		// prob shouldnt *3
		if (fileSize != size) {
			throw std::runtime_error("wrong file size relation to weights");
			return;
		}

		fread(weights, sizeof(float), size, f);
		// idt we need to catch an exception bc we djd b4
	}
	int accumulatedSize = 0;
	float* start = &weights[0];
	for (int i = 0; i < layersAdd.size(); ++i) {
		if (i== 0) {
			layers.push_back({start, start, layersAdd[i].size, layersAdd[i].size, 0, layersAdd[i].size, layersAdd[i].hook, layersAdd[i].hookGrad});
		}
		accumulatedSize += perLayerSize[i];
		float* begin = start + accumulatedSize;
		layers.push_back({begin, begin, layersAdd[i-1].size, layersAdd[i].size, layersAdd[i].size + layersAdd[i].size * layersAdd[i-1].size * 2, layersAdd[i].size, layersAdd[i].hook, layersAdd[i].hookGrad});
	} 

}

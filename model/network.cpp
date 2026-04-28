#include <iostream>
#include <network.h>
#include <stdexcept>


float* Layer::forward(float* inputs) {
	#ifdef TRAINING_ON
		
	#endif
	// we cant rly validate inputs? so js wing it. but heisenbugs are concerning. hopefully should be fine enough.
	if (input == 0) {
	// input layer
	}
	i

	
}

#ifdef TRAINING_ON
void Layer::backward(float* upstream_grad) {

}
#endif

std::vector<float> weights;
std::vector<Layer> layers;
std::vector<int> perLayerSize

	
void setupNeuralNetwork(std::vector<LayerArgs> layersAdd, s
td::string weightsPath = "UNDEFINED276lineosersyoujelly?") {
	layers.clear();
	weights.clear();
	int size = 0;
	// ptobably we csn optimise this but idk

	LayerArgs& prev;

	for (Layer& layer : layersAdd) {
		if (prev == nullptr) {
			perLayerSize.push_back(0);
		prev = layer;
			continue;
		}
		size += layer.layerSize * prev.layerSize * 3 // for quadratic, linear and bias
							     		perLayerSize.push_back(layer.layerSize*prev.layerSize*3)
        	prev = layer
	}
	// ratio lineOSERS, go deal with your O(N)
	// vs O(~5/3*0.693*1.5throot(N))
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
		layers.push_back({start, start, layersAdd[i].size, layersAdd[i].size, layersAdd[i].size * layersAdd[i].size, layersAdd[i].hook, layersAdd[i].hookGrad});
		}
		accumulatedSize += perLayerSize[i];
		float* begin = start + accumulatedSize;
		layers.push_back({begin, begin, layersAdd[i-1].size, layersAdd[i].size, layersAdd[i].hook, layersAdd[i].hookGrad});
	} 

}

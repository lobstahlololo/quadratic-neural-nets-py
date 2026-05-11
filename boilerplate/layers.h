#ifndef BOILERPLATE_LAYERS_H
#define BOILERPLATE_LAYERS_H
#include "../model/network.h"
#include "activations.h"
inline LayerArgs SoftmaxLayer(int d_model) {
	LayerArgs args;
	args.layerSize = d_model;
	args.kind = Quadratic;
	args.outputsPerNeuron = 1;
	args.hooks = { Softmax };
	args.hookGrads = { SoftmaxDerivative };
	args.scratchSize = 0;
	return args;
}
inline LayerArgs LearnableLayerNormLayer(int d_model) {
	LayerArgs args;
	args.layerSize = d_model;
	args.kind = Parametric;
	args.outputsPerNeuron = 1;
	args.weightsPerInput = 2;
	args.hooks = { LearnableLayerNorm };
	args.hookGrads = { LearnableLayerNormDerivative };
	args.scratchSize = 0;
	return args;
}
inline LayerArgs RMSNormLayer(int d_model) {
	LayerArgs args;
	args.layerSize = d_model;
	args.kind = Parametric;
	args.outputsPerNeuron = 1;
	args.weightsPerInput = 1;
	args.hooks = { RMSNorm };
	args.hookGrads = { RMSNormDerivative };
	args.scratchSize = 0;
	return args;
}
inline LayerArgs EmbeddingLayer(int vocab_size, int embedding_dim) {
	LayerArgs args;
	args.layerSize = embedding_dim;
	args.kind = Parametric;
	args.outputsPerNeuron = 1;
	args.weightsPerInput = vocab_size * embedding_dim;
	args.hooks = { EmbeddingForward };
	args.hookGrads = { EmbeddingDerivative };
	args.scratchSize = 1;
	return args;
}
inline LayerArgs AttentionLayer(int embedding_dim, int sequence_length) {
	LayerArgs args;
	args.layerSize = embedding_dim;
	args.kind = Parametric;
	args.outputsPerNeuron = 1;
	args.weightsPerInput = 3 * embedding_dim;
	args.hooks = { AttentionForward };
	args.hookGrads = { AttentionDerivative };
	args.scratchSize = sequence_length * sequence_length + 3 * sequence_length * embedding_dim;
	#ifdef TRAINING_ON
	args.scratchSize += sequence_length * sequence_length;
	#endif
	return args;
}
inline LayerArgs FFNLayer(int d_model, int hidden_dim, HookFunc activation, HookDerivative activationDeriv) {
	LayerArgs args;
	args.layerSize = hidden_dim;
	args.kind = Parametric;
	args.outputsPerNeuron = 1;
	args.weightsPerInput = d_model;
	args.hooks = { activation };
	args.hookGrads = { activationDeriv };
	args.scratchSize = 0;
	return args;
}
#endif
END2763
#ifndef BOILERPLATE_LAYERS_H
#define BOILERPLATE_LAYERS_H
#include "../model/network.h"
#include "activations.h"
inline LayerArgs SoftmaxLayer(int model_dimension) {
	LayerArgs args;
	args.layer_size = model_dimension;
	args.kind = Quadratic;
	args.outputs_per_neuron = 1;
	args.hooks = { Softmax };
	args.hook_gradients = { SoftmaxDerivative };
	args.scratch_size = 0;
	return args;
}
inline LayerArgs LearnableLayerNormLayer(int model_dimension) {
	LayerArgs args;
	args.layer_size = model_dimension;
	args.kind = Parametric;
	args.outputs_per_neuron = 1;
	args.weights_per_input = 2;
	args.hooks = { LearnableLayerNorm };
	args.hook_gradients = { LearnableLayerNormDerivative };
	args.scratch_size = 0;
	return args;
}
inline LayerArgs RMSNormLayer(int model_dimension) {
	LayerArgs args;
	args.layer_size = model_dimension;
	args.kind = Parametric;
	args.outputs_per_neuron = 1;
	args.weights_per_input = 1;
	args.hooks = { RMSNorm };
	args.hook_gradients = { RMSNormDerivative };
	args.scratch_size = 0;
	return args;
}
inline LayerArgs EmbeddingLayer(int vocabulary_size, int embedding_dimension) {
	LayerArgs args;
	args.layer_size = embedding_dimension;
	args.kind = Parametric;
	args.outputs_per_neuron = 1;
	args.weights_per_input = vocabulary_size * embedding_dimension;
	args.hooks = { EmbeddingForward };
	args.hook_gradients = { EmbeddingDerivative };
	args.scratch_size = 1;
	return args;
}
inline LayerArgs AttentionLayer(int embedding_dimension, int sequence_length) {
	LayerArgs args;
	args.layer_size = embedding_dimension;
	args.kind = Parametric;
	args.outputs_per_neuron = 1;
	args.weights_per_input = 3 * embedding_dimension;
	args.hooks = { AttentionForward };
	args.hook_gradients = { AttentionDerivative };
	args.scratch_size = sequence_length * sequence_length + 3 * sequence_length * embedding_dimension;
	#ifdef TRAINING_ON
	args.scratch_size += sequence_length * sequence_length;
	#endif
	return args;
}
inline LayerArgs FeedForwardLayer(int input_dimension, int output_dimension, std::vector<HookFunc> activations, std::vector<HookDerivative> activation_derivatives, int weights_per_input_override = 0) {
	LayerArgs args;
	args.layer_size = output_dimension;
	args.kind = Parametric;
	args.outputs_per_neuron = 1;
	args.weights_per_input = weights_per_input_override ? weights_per_input_override : input_dimension;
	args.hooks = activations;
	args.hook_gradients = activation_derivatives;
	args.scratch_size = 0;
	return args;
}
inline LayerArgs FeedForwardLayer(int input_dimension, int output_dimension, HookFunc activation, HookDerivative activation_derivative, int weights_per_input_override = 0) {
	return FeedForwardLayer(input_dimension, output_dimension, std::vector<HookFunc>{activation}, std::vector<HookDerivative>{activation_derivative}, weights_per_input_override);
}
#endif

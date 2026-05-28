#ifndef BOILERPLATE_LAYERS_H
#define BOILERPLATE_LAYERS_H
#include "../model/network.h"
#include "activations.h"
#include <vector>
#include <iostream>
inline LayerArgs InputLayer(int input_dimension) {
	LayerArgs args;
	args.layer_size = input_dimension;
	args.kind = Quadratic;
	args.scratch_size = 0;
	return args;
}
inline LayerArgs SoftmaxLayer(int model_dimension) {
	LayerArgs args;
	args.layer_size = model_dimension;
	args.kind = Quadratic;
	args.hooks = { Softmax };
	args.hook_gradients = { SoftmaxDerivative };
	args.scratch_size = 0;
	return args;
}
inline LayerArgs LearnableLayerNormLayer(int model_dimension) {
	LayerArgs args;
	args.layer_size = model_dimension;
	args.kind = Parametric;
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
	args.weights_per_input = vocabulary_size * embedding_dimension;
	args.hooks = { EmbeddingForward };
	args.hook_gradients = { EmbeddingDerivative };
	args.scratch_size = 1;
	return args;
}
inline LayerArgs AttentionLayer(int embedding_dimension, int max_sequence_length, bool use_kv_cache = false) {
	LayerArgs args;
	args.layer_size = embedding_dimension;
	args.kind = Parametric;
	args.weights_per_input = 3 * embedding_dimension;
	args.scratch_size = 2 * max_sequence_length * max_sequence_length + 4 * max_sequence_length * embedding_dimension + batch_size + 1;
	args.extra_args = { max_sequence_length };
#ifndef TRAINING_ON
	if (use_kv_cache) {
		args.extra_args.push_back(kv_cache_pool.size());
		args.hooks = { AttentionForwardWithCache };
		args.hook_gradients = { AttentionDerivativeWithCache };
		kv_cache_pool.emplace_back();
	} else {
		args.hooks = { AttentionForward };
		args.hook_gradients = { AttentionDerivative };
	}
#else
	args.hooks = { AttentionForward };
	args.hook_gradients = { AttentionDerivative };
#endif
	return args;
}
inline LayerArgs FeedForwardLayer(int input_dimension, int output_dimension, std::vector<HookFunc> activations, std::vector<HookDerivative> activation_derivatives, int weights_per_input_override = 0) {
	LayerArgs args;
	args.layer_size = output_dimension;
	args.kind = Quadratic;
	args.hooks = activations;
	args.hook_gradients = activation_derivatives;
	args.scratch_size = 0;
	return args;
}
inline LayerArgs FeedForwardLayer(int input_dimension, int output_dimension, HookFunc activation, HookDerivative activation_derivative, int weights_per_input_override = 0) {
	return FeedForwardLayer(input_dimension, output_dimension, std::vector<HookFunc>{activation}, std::vector<HookDerivative>{activation_derivative}, weights_per_input_override);
}
#endif
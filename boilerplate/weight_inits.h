#ifndef WEIGHT_INITS_IMPORTED
#define WEIGHT_INITS_IMPORTED
#include "../model/network.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
inline void xavier_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	int weight_index = 0;
	for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
		const LayerArgs& current = layers[layer_idx];
		const LayerArgs& previous = layers[layer_idx - 1];
		int input_dimension = previous.layer_size;
		int output_dimension = current.layer_size;
		float base_scale = std::sqrt(6.0f / (input_dimension + output_dimension));
		float quad_scale = base_scale / std::sqrt(3.0f);
		if (current.kind == Parametric) {
			int weights_count = input_dimension * current.weights_per_input;
			for (int i = 0; i < weights_count; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * base_scale;
			}
			weight_index += weights_count;
		} else {
			int quad_weights = input_dimension * output_dimension;
			int lin_weights = quad_weights;
			int bias_weights = output_dimension;
			for (int i = 0; i < quad_weights; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * quad_scale;
			}
			weight_index += quad_weights;
			for (int i = 0; i < lin_weights; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * base_scale;
			}
			weight_index += lin_weights;
			for (int i = 0; i < bias_weights; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * base_scale;
			}
			weight_index += bias_weights;
		}
	}
}
inline void he_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	int weight_index = 0;
	for (size_t layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
		const LayerArgs& current = layers[layer_idx];
		const LayerArgs& previous = layers[layer_idx - 1];
		int input_dimension = previous.layer_size;
		int output_dimension = current.layer_size;
		float base_scale = std::sqrt(6.0f / input_dimension);
		float quad_scale = base_scale / std::sqrt(3.0f);
		if (current.kind == Parametric) {
			int weights_count = input_dimension * current.weights_per_input;
			for (int i = 0; i < weights_count; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * base_scale;
			}
			weight_index += weights_count;
		} else {
			int quad_weights = input_dimension * output_dimension;
			int lin_weights = quad_weights;
			int bias_weights = output_dimension;
			for (int i = 0; i < quad_weights; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * quad_scale;
			}
			weight_index += quad_weights;
			for (int i = 0; i < lin_weights; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * base_scale;
			}
			weight_index += lin_weights;
			for (int i = 0; i < bias_weights; ++i) {
				float random_value = static_cast<float>(rand()) / RAND_MAX;
				weights[weight_index + i] = (random_value * 2.0f - 1.0f) * base_scale;
			}
			weight_index += bias_weights;
		}
	}
}
inline void uniform_random_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	for (int i = 0; i < total_size; ++i) {
		float random_value = static_cast<float>(rand()) / RAND_MAX;
		weights[i] = random_value - 0.5f;
	}
}
inline void zero_initialisation(float* weights, int total_size, const std::vector<LayerArgs>& layers) {
	std::fill(weights, weights + total_size, 0.0f);
	
}
#endif

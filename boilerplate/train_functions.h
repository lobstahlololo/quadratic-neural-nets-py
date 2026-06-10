#ifndef BOILERPLATE_TRAIN_FUNCTIONS_H
#define BOILERPLATE_TRAIN_FUNCTIONS_H
#include "../model/network.h"
#include "../train/train.h"
#include <vector>
#include <variant>
#include <cmath>
#include <algorithm>

inline float global_weight_decay = 0.01f;
inline float global_max_trust_ratio = 10.0f;

inline float train_adams(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& sequence_lengths) {
	int total_rows = 0;
	for (int s : sequence_lengths) total_rows += s;
	float* allocated_input_pointer = const_cast<float*>(training_data.data()); 
	for (int i = 0; i < layers.size(); ++i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			allocated_input_pointer = layer_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			allocated_input_pointer = param_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
	}
	int last_layer_output = 0;
	std::visit([&](auto& l) { last_layer_output = l.output; }, layers.back());
	std::vector<float> sample_losses(total_rows);
	for (int i = 0; i < total_rows; ++i) {
		std::vector<int> single_target = { correct_indices[i] };
		sample_losses[i] = loss_function(output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, last_layer_output);
	}
	float batch_loss = 0.0f;
	for (int i = 0; i < total_rows; ++i) batch_loss += sample_losses[i];
	batch_loss /= total_rows;
	std::vector<float> downstream_gradient(last_layer_output * total_rows);
	for (int i = 0; i < total_rows; ++i) {
		std::vector<int> single_target = { correct_indices[i] };
		loss_derivative(sample_losses[i], output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, downstream_gradient.data() + last_layer_output * i, last_layer_output);
	}
	std::vector<int> mutable_sequence_lengths = sequence_lengths;
	float* grad_ptr = downstream_gradient.data();
	for (int i = layers.size() - 1; i >= 0; --i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			grad_ptr = layer_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			grad_ptr = param_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
		}
	}
	float scaled_learning_rate = learning_rate / batch_size;
	float first_moment_decay_power = std::pow(0.9f, step + 1);
	float second_moment_decay_power = std::pow(0.999f, step + 1);
	size_t moment_offset = 0;
	float max = 0.0f;
	for (int i = 0; i < layers.size(); ++i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			for (int j = 0; j < layer_ptr->size; ++j) {
				float grad = layer_ptr->weight_gradients[j];
				max = std::max(max, std::abs(grad));
				first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
				second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
				float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
				float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float w_val = layer_ptr->weights_begin[j];
				layer_ptr->weights_begin[j] -= scaled_learning_rate * (adam_step + global_weight_decay * w_val);
			}
			moment_offset += layer_ptr->size;
			for (int j = 0; j < layer_ptr->extra_weights_size; ++j) {
				float grad = layer_ptr->extra_weight_gradients[j];
				max = std::max(max, std::abs(grad));
				first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
				second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
				float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
				float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float w_val = layer_ptr->extra_weights_begin[j];
				layer_ptr->extra_weights_begin[j] -= scaled_learning_rate * (adam_step + global_weight_decay * w_val);
			}
			moment_offset += layer_ptr->extra_weights_size;
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			int weight_count = param_ptr->input * param_ptr->weights_per_input;
			for (int j = 0; j < weight_count; ++j) {
				float grad = param_ptr->weight_gradients[j];
				max = std::max(max, std::abs(grad));
				first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
				second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
				float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
				float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float w_val = param_ptr->weights_begin[j];
				param_ptr->weights_begin[j] -= scaled_learning_rate * (adam_step + global_weight_decay * w_val);
			}
			moment_offset += weight_count;
			for (int j = 0; j < param_ptr->extra_weights_size; ++j) {
				float grad = param_ptr->extra_weight_gradients[j];
				max = std::max(max, std::abs(grad));
				first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
				second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
				float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
				float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
				float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
				float w_val = param_ptr->extra_weights_begin[j];
				param_ptr->extra_weights_begin[j] -= scaled_learning_rate * (adam_step + global_weight_decay * w_val);
			}
			moment_offset += param_ptr->extra_weights_size;
		}
	}
	if (trainHook) trainHook();
	return batch_loss;
}

inline float train_lamb(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& sequence_lengths) {
	int total_rows = 0;
	for (int s : sequence_lengths) total_rows += s;
	float* allocated_input_pointer = const_cast<float*>(training_data.data()); 
	for (int i = 0; i < layers.size(); ++i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			allocated_input_pointer = layer_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			allocated_input_pointer = param_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
		}
	}
	int last_layer_output = 0;
	std::visit([&](auto& l) { last_layer_output = l.output; }, layers.back());
	std::vector<float> sample_losses(total_rows);
	for (int i = 0; i < total_rows; ++i) {
		std::vector<int> single_target = { correct_indices[i] };
		sample_losses[i] = loss_function(output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, last_layer_output);
	}
	float batch_loss = 0.0f;
	for (int i = 0; i < total_rows; ++i) batch_loss += sample_losses[i];
	batch_loss /= total_rows;
	std::vector<float> downstream_gradient(last_layer_output * total_rows);
	for (int i = 0; i < total_rows; ++i) {
		std::vector<int> single_target = { correct_indices[i] };
		loss_derivative(sample_losses[i], output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, downstream_gradient.data() + last_layer_output * i, last_layer_output);
	}
	std::vector<int> mutable_sequence_lengths = sequence_lengths;
	float* grad_ptr = downstream_gradient.data();
	for (int i = layers.size() - 1; i >= 0; --i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			grad_ptr = layer_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			grad_ptr = param_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
		}
	}
	float scaled_learning_rate = learning_rate / batch_size;
	float first_moment_decay_power = std::pow(0.9f, step + 1);
	float second_moment_decay_power = std::pow(0.999f, step + 1);
	size_t moment_offset = 0;
	float max = 0.0f;
	for (int i = 0; i < layers.size(); ++i) {
		if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
			if (layer_ptr->size > 0) {
				std::vector<float> u_temp(layer_ptr->size);
				float sum_w2 = 0.0f;
				float sum_u2 = 0.0f;
				for (int j = 0; j < layer_ptr->size; ++j) {
					float grad = layer_ptr->weight_gradients[j];
					max = std::max(max, std::abs(grad));
					first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
					second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
					float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
					float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float w_val = layer_ptr->weights_begin[j];
					float u_val = adam_step + global_weight_decay * w_val;
					u_temp[j] = u_val;
					sum_w2 += w_val * w_val;
					sum_u2 += u_val * u_val;
				}
				float r_w = std::sqrt(sum_w2);
				float r_u = std::sqrt(sum_u2);
				float trust_ratio = 1.0f;
				if (r_w > 0.0f && r_u > 0.0f) {
					trust_ratio = r_w / r_u;
					if (trust_ratio > global_max_trust_ratio) {
						trust_ratio = global_max_trust_ratio;
					}
				}
				for (int j = 0; j < layer_ptr->size; ++j) {
					layer_ptr->weights_begin[j] -= scaled_learning_rate * trust_ratio * u_temp[j];
				}
			}
			moment_offset += layer_ptr->size;
			if (layer_ptr->extra_weights_size > 0) {
				std::vector<float> u_temp(layer_ptr->extra_weights_size);
				float sum_w2 = 0.0f;
				float sum_u2 = 0.0f;
				for (int j = 0; j < layer_ptr->extra_weights_size; ++j) {
					float grad = layer_ptr->extra_weight_gradients[j];
					max = std::max(max, std::abs(grad));
					first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
					second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
					float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
					float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float w_val = layer_ptr->extra_weights_begin[j];
					float u_val = adam_step + global_weight_decay * w_val;
					u_temp[j] = u_val;
					sum_w2 += w_val * w_val;
					sum_u2 += u_val * u_val;
				}
				float r_w = std::sqrt(sum_w2);
				float r_u = std::sqrt(sum_u2);
				float trust_ratio = 1.0f;
				if (r_w > 0.0f && r_u > 0.0f) {
					trust_ratio = r_w / r_u;
					if (trust_ratio > global_max_trust_ratio) trust_ratio = global_max_trust_ratio;
				}
				for (int j = 0; j < layer_ptr->extra_weights_size; ++j) {
					layer_ptr->extra_weights_begin[j] -= scaled_learning_rate * trust_ratio * u_temp[j];
				}
			}
			moment_offset += layer_ptr->extra_weights_size;
		} else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
			int weight_count = param_ptr->input * param_ptr->weights_per_input;
			if (weight_count > 0) {
				std::vector<float> u_temp(weight_count);
				float sum_w2 = 0.0f;
				float sum_u2 = 0.0f;
				for (int j = 0; j < weight_count; ++j) {
					float grad = param_ptr->weight_gradients[j];
					max = std::max(max, std::abs(grad));
					first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
					second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
					float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
					float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float w_val = param_ptr->weights_begin[j];
					float u_val = adam_step + global_weight_decay * w_val;
					u_temp[j] = u_val;
					sum_w2 += w_val * w_val;
					sum_u2 += u_val * u_val;
				}
				float r_w = std::sqrt(sum_w2);
				float r_u = std::sqrt(sum_u2);
				float trust_ratio = 1.0f;
				if (r_w > 0.0f && r_u > 0.0f) {
					trust_ratio = r_w / r_u;
					if (trust_ratio > global_max_trust_ratio) {
						trust_ratio = global_max_trust_ratio;
					}
				}
				for (int j = 0; j < weight_count; ++j) {
					param_ptr->weights_begin[j] -= scaled_learning_rate * trust_ratio * u_temp[j];
				}
			}
			moment_offset += weight_count;
			if (param_ptr->extra_weights_size > 0) {
				std::vector<float> u_temp(param_ptr->extra_weights_size);
				float sum_w2 = 0.0f;
				float sum_u2 = 0.0f;
				for (int j = 0; j < param_ptr->extra_weights_size; ++j) {
					float grad = param_ptr->extra_weight_gradients[j];
					max = std::max(max, std::abs(grad));
					first_moment_buffer[moment_offset + j] = 0.9f * first_moment_buffer[moment_offset + j] + 0.1f * grad;
					second_moment_buffer[moment_offset + j] = 0.999f * second_moment_buffer[moment_offset + j] + 0.001f * grad * grad;
					float first_bias_correction = first_moment_buffer[moment_offset + j] / (1.0f - first_moment_decay_power);
					float second_bias_correction = second_moment_buffer[moment_offset + j] / (1.0f - second_moment_decay_power);
					float adam_step = first_bias_correction / (std::sqrt(second_bias_correction) + 1e-8f);
					float w_val = param_ptr->extra_weights_begin[j];
					float u_val = adam_step + global_weight_decay * w_val;
					u_temp[j] = u_val;
					sum_w2 += w_val * w_val;
					sum_u2 += u_val * u_val;
				}
				float r_w = std::sqrt(sum_w2);
				float r_u = std::sqrt(sum_u2);
				float trust_ratio = 1.0f;
				if (r_w > 0.0f && r_u > 0.0f) {
					trust_ratio = r_w / r_u;
					if (trust_ratio > global_max_trust_ratio) trust_ratio = global_max_trust_ratio;
				}
				for (int j = 0; j < param_ptr->extra_weights_size; ++j) {
					param_ptr->extra_weights_begin[j] -= scaled_learning_rate * trust_ratio * u_temp[j];
				}
			}
			moment_offset += param_ptr->extra_weights_size;
		}
	}
	if (trainHook) trainHook();
	return batch_loss;
}

inline float regular_gd(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& sequence_lengths) {
    int total_rows = 0;
    for (int s : sequence_lengths) total_rows += s;
    float* allocated_input_pointer = const_cast<float*>(training_data.data()); 
    for (int i = 0; i < layers.size(); ++i) {
        if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
            allocated_input_pointer = layer_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
        } else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
            allocated_input_pointer = param_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
        }
    }
    int last_layer_output = 0;
    std::visit([&](auto& l) { last_layer_output = l.output; }, layers.back());
    std::vector<float> sample_losses(total_rows);
    for (int i = 0; i < total_rows; ++i) {
        std::vector<int> single_target = { correct_indices[i] };
        sample_losses[i] = loss_function(output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, last_layer_output);
    }
    float batch_loss = 0.0f;
    for (int i = 0; i < total_rows; ++i) batch_loss += sample_losses[i];
    batch_loss /= total_rows;
    std::vector<float> downstream_gradient(last_layer_output * total_rows);
    for (int i = 0; i < total_rows; ++i) {
        std::vector<int> single_target = { correct_indices[i] };
        loss_derivative(sample_losses[i], output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, downstream_gradient.data() + last_layer_output * i, last_layer_output);
    }
    std::vector<int> mutable_sequence_lengths = sequence_lengths;
    float* grad_ptr = downstream_gradient.data();
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
            grad_ptr = layer_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
        } else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
            grad_ptr = param_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
        }
    }
    for (int i = 0; i < layers.size(); ++i) {
        if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
            for (int j = 0; j < layer_ptr->size; ++j) {
                float grad = layer_ptr->weight_gradients[j];
                layer_ptr->weights_begin[j] -= learning_rate * grad;
            }
            for (int j = 0; j < layer_ptr->extra_weights_size; ++j) {
                float grad = layer_ptr->extra_weight_gradients[j];
                layer_ptr->extra_weights_begin[j] -= learning_rate * grad;
            }
        } else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
            int weight_count = param_ptr->input * param_ptr->weights_per_input;
            for (int j = 0; j < weight_count; ++j) {
                float grad = param_ptr->weight_gradients[j];
                param_ptr->weights_begin[j] -= learning_rate * grad;
            }
            for (int j = 0; j < param_ptr->extra_weights_size; ++j) {
                float grad = param_ptr->extra_weight_gradients[j];
                param_ptr->extra_weights_begin[j] -= learning_rate * grad;
            }
        }
    }
    if (trainHook) trainHook();
    return batch_loss;
}
inline float stochastic_gd(std::vector<std::variant<Layer, ParametricLayer>>& layers, const std::vector<float>& training_data, const std::vector<int>& correct_indices, const std::vector<float>& required_output, float learning_rate, const LossFunc& loss_function, const LossDerivative& loss_derivative, int step, int batch_count, std::vector<int>& sequence_lengths) {
    int total_rows = 0;
    for (int s : sequence_lengths) total_rows += s;
    float* allocated_input_pointer = const_cast<float*>(training_data.data()); 
    for (int i = 0; i < layers.size(); ++i) {
        if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
            allocated_input_pointer = layer_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
        } else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
            allocated_input_pointer = param_ptr->forward(allocated_input_pointer, batch_count, sequence_lengths);
        }
    }
    int last_layer_output = 0;
    std::visit([&](auto& l) { last_layer_output = l.output; }, layers.back());
    std::vector<float> sample_losses(total_rows);
    for (int i = 0; i < total_rows; ++i) {
        std::vector<int> single_target = { correct_indices[i] };
        sample_losses[i] = loss_function(output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, last_layer_output);
    }
    float batch_loss = 0.0f;
    for (int i = 0; i < total_rows; ++i) batch_loss += sample_losses[i];
    batch_loss /= total_rows;
    std::vector<float> downstream_gradient(last_layer_output * total_rows);
    for (int i = 0; i < total_rows; ++i) {
        std::vector<int> single_target = { correct_indices[i] };
        loss_derivative(sample_losses[i], output_buffers.first.data() + last_layer_output * i, required_output.data() + last_layer_output * i, single_target, downstream_gradient.data() + last_layer_output * i, last_layer_output);
    }
    std::vector<int> mutable_sequence_lengths = sequence_lengths;
    float* grad_ptr = downstream_gradient.data();
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
            grad_ptr = layer_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
        } else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
            grad_ptr = param_ptr->backward(grad_ptr, batch_count, mutable_sequence_lengths, correct_indices);
        }
    }
    float scaled_learning_rate = learning_rate / batch_size;
    for (int i = 0; i < layers.size(); ++i) {
        if (auto* layer_ptr = std::get_if<Layer>(&layers[i])) {
            for (int j = 0; j < layer_ptr->size; ++j) {
                float grad = layer_ptr->weight_gradients[j];
                layer_ptr->weights_begin[j] -= scaled_learning_rate * grad;
            }
            for (int j = 0; j < layer_ptr->extra_weights_size; ++j) {
                float grad = layer_ptr->extra_weight_gradients[j];
                layer_ptr->extra_weights_begin[j] -= scaled_learning_rate * grad;
            }
        } else if (auto* param_ptr = std::get_if<ParametricLayer>(&layers[i])) {
            int weight_count = param_ptr->input * param_ptr->weights_per_input;
            for (int j = 0; j < weight_count; ++j) {
                float grad = param_ptr->weight_gradients[j];
                param_ptr->weights_begin[j] -= scaled_learning_rate * grad;
            }
            for (int j = 0; j < param_ptr->extra_weights_size; ++j) {
                float grad = param_ptr->extra_weight_gradients[j];
                param_ptr->extra_weights_begin[j] -= scaled_learning_rate * grad;
            }
        }
    }
    if (trainHook) trainHook();
    return batch_loss;
}
#endif
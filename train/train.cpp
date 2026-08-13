#include "train.h"
#include "../model/network.h"
#include "../boilerplate/train_functions.h"
#include <iostream>
#include <cmath>
#include <variant>
#include <iostream>
using std::vector;

TrainHook trainHook = nullptr;

vector<float> first_moment_buffer;
vector<float> second_moment_buffer;

float train(std::vector<std::variant<Layer, ParametricLayer>> &layers, const std::vector<float> &training_data, const std::vector<int> &correct_indices, const std::vector<float> &required_output, float learning_rate, const LossFunc &loss_function, const LossDerivative &loss_derivative, int step, int batch_count, std::vector<int> &sequence_lengths)
{
	std::cerr << "!!! TRAIN FUNCTION REACHED !!!\n";
	return train_adams(layers, training_data, correct_indices, required_output, learning_rate, loss_function, loss_derivative, step, batch_count, sequence_lengths);
}
void trainScheduler(std::vector<std::variant<Layer, ParametricLayer>> &layers, const std::vector<float> &training_data, const std::vector<int> &correct_indices, std::vector<float> required_output, float learning_rate, float minimum_learning_rate, const LossFunc &loss_function, const LossDerivative &loss_derivative, int total_epochs, int batch_size_arg, std::vector<int> initial_sequence_lengths, TrainFunction train_func)
{
	// std::cout << "TRAIN BEGIN." << "\n";
	std::cout << "DEBUG: SCHEDULER CALLED, train_func=" << (void *)train_func << "\n";
	first_moment_buffer.resize(network_size);
	second_moment_buffer.resize(network_size);
	batch_size = batch_size_arg;
	if (initial_sequence_lengths.empty())
	{
		initial_sequence_lengths.resize(batch_size, 1);
	}
	// initial_sequence_lengths describes the batch_size sequences of ONE batch.
	// Clamp to batch_size entries so total_input_rows (and therefore each step's
	// total_rows) can never exceed the buffer capacity of batch_size rows, even
	// if a caller passes one entry per window instead.
	if ((int)initial_sequence_lengths.size() > batch_size)
	{
		initial_sequence_lengths.resize(batch_size);
	}

	int input_size = 0;
	std::visit([&](auto &l)
			   { input_size = l.input; }, layers[0]);
	int output_size = 0;
	std::visit([&](auto &l)
			   { output_size = l.output; }, layers.back());
	int total_input_rows = 0;
	for (int s : initial_sequence_lengths)
	{
		total_input_rows += s;
	}

	// Each training row is ONE token-id float (the embedding hook reads original_inputs[row_index]).
	// The data layout has input_size=embedding_dim floats per row only AFTER the embedding layer,
	// so dividing by (total_input_rows * input_size) makes steps_per_epoch == 0 here, and the
	// epoch loss then becomes 0.0f / 0 = NaN before any training function is ever called.
	int steps_per_epoch = training_data.size() / total_input_rows;
	for (int epoch = 0; epoch < total_epochs; ++epoch)
	{
		float epoch_loss = 0.0f;
		for (int j = 0; j < steps_per_epoch; ++j)
		{
			float current_learning_rate = minimum_learning_rate + (learning_rate - minimum_learning_rate) * (1 + cos(3.14159265f * j / steps_per_epoch)) / 2;
			std::vector<float> batch_inputs(training_data.begin() + j * total_input_rows, training_data.begin() + (j + 1) * total_input_rows);
			std::vector<float> batch_targets(required_output.begin() + j * total_input_rows * output_size, required_output.begin() + (j + 1) * total_input_rows * output_size);
			std::vector<int> batch_correct_indices(correct_indices.begin() + j * total_input_rows, correct_indices.begin() + (j + 1) * total_input_rows);
			int step = epoch * steps_per_epoch + j;
			std::vector<int> mutable_sequence_lengths = initial_sequence_lengths;
			float batch_loss = train_func(layers, batch_inputs, batch_correct_indices, batch_targets, current_learning_rate, loss_function, loss_derivative, step, batch_size, mutable_sequence_lengths);
			epoch_loss += batch_loss;
		}
		epoch_loss /= steps_per_epoch;
		std::cout << "Epoch " << epoch + 1 << "/" << total_epochs << " - Loss: " << epoch_loss << "\n";
	}
}
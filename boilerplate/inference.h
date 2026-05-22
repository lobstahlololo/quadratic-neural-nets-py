#ifndef BOILERPLATE_INFERENCE_H
#define BOILERPLATE_INFERENCE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "../model/network.h"
inline void bpe_tokenize(const std::string& text, const std::unordered_map<std::string, int>& vocabulary, std::vector<int>& token_ids) {
	token_ids.clear();
	size_t pos = 0;
	while (pos < text.size()) {
		int best_len = 0;
		int best_id = -1;
		for (int len = 1; pos + len <= text.size(); ++len) {
			std::string sub = text.substr(pos, len);
			auto it = vocabulary.find(sub);
			if (it != vocabulary.end()) {
				best_len = len;
				best_id = it->second;
			}
		}
		if (best_len == 0) {
			std::string single(1, text[pos]);
			auto it = vocabulary.find(single);
			if (it != vocabulary.end()) {
				best_id = it->second;
				best_len = 1;
			} else {
				best_id = 0;
				best_len = 1;
			}
		}
		token_ids.push_back(best_id);
		pos += best_len;
	}
}
inline void save_vocabulary(const std::unordered_map<std::string, int>& vocabulary, const std::string& filepath) {
	std::ofstream out(filepath);
	for (const auto& pair : vocabulary) {
		out << pair.first << " " << pair.second << "\n";
	}
}
inline void load_vocabulary(std::unordered_map<std::string, int>& vocabulary, const std::string& filepath) {
	vocabulary.clear();
	std::ifstream in(filepath);
	std::string token;
	int id;
	while (in >> token >> id) {
		vocabulary[token] = id;
	}
}
inline int sample_from_probabilities(const float* probabilities, int vocab_size) {
	float r = static_cast<float>(rand()) / RAND_MAX;
	float cumulative = 0.0f;
	for (int i = 0; i < vocab_size; ++i) {
		cumulative += probabilities[i];
		if (r <= cumulative) return i;
	}
	return vocab_size - 1;
}
inline std::vector<int> generate_tokens(const std::vector<Layer>& layers, const std::unordered_map<std::string, int>& vocabulary, const std::vector<int>& initial_tokens, int tokens_to_generate, int buffer_multiplier) {
	std::vector<int> generated = initial_tokens;
	std::vector<int> current_sequence = initial_tokens;
	int vocab_size = vocabulary.size();
	int max_seq_len = buffer_multiplier;
	for (int step = 0; step < tokens_to_generate; ++step) {
		while (current_sequence.size() < max_seq_len) current_sequence.push_back(0);
		if (current_sequence.size() > max_seq_len) {
			current_sequence.erase(current_sequence.begin(), current_sequence.begin() + (current_sequence.size() - max_seq_len));
		}
		std::vector<float> input_floats(current_sequence.size());
		for (size_t i = 0; i < current_sequence.size(); ++i) input_floats[i] = static_cast<float>(current_sequence[i]);
		std::vector<int> batch_sizes = { static_cast<int>(current_sequence.size()) };
		int batch_count = 1;
		float* output_ptr = input_floats.data();
		for (size_t i = 0; i < layers.size(); ++i) {
			output_ptr = layers[i].forward(output_ptr, batch_count, batch_sizes);
		}
		int total_outputs = layers.back().output * current_sequence.size();
		int next_token = sample_from_probabilities(output_ptr + (current_sequence.size() - 1) * layers.back().output, vocab_size);
		generated.push_back(next_token);
		current_sequence.push_back(next_token);
	}
	return generated;
}
#endif
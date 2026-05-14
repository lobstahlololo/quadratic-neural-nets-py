#ifndef TOKENIZERS_IMPORTED
#define TOKENIZERS_IMPORTED
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <map>
#include <algorithm>
void tokenize_letter(const std::vector<char>& input, std::vector<int>& output) {
	for (int i = 0; i < input.size(); ++i) {
		output[i] = static_cast<int>(input[i]);
	}
}
void tokenize_words(const std::vector<std::string>& input, std::vector<int>& output, std::unordered_map<std::string, int>& wordToIndex) {
	int currentIndex = wordToIndex.size();
	for (const std::string& word : input) {
		if (wordToIndex.find(word) == wordToIndex.end()) {
			wordToIndex[word] = currentIndex++;
		}
		output.push_back(wordToIndex[word]);
	}
}
struct Token {
	std::string representation;
	int left;
	int right;
	bool duplicate;
};
void tokenize_bpe(const std::vector<char>& input, std::vector<int>& output, std::unordered_map<std::string, int>& vocabulary, int vocabSize) {
	std::vector<Token> tokens;
	tokens.reserve(input.size());
	std::map<std::pair<std::string, std::string>, std::vector<int>> pairPositions;
	std::map<std::pair<std::string, std::string>, int> pairFrequencies;
	std::priority_queue<std::pair<int, std::pair<std::string, std::string>>> mergeQueue;
	for (int i = 0; i < input.size(); ++i) {
		std::string str(1, input[i]);
		vocabulary[str] = 0;
		if (i > 0) {
			tokens.push_back({str, i - 1, -1, false});
			continue;
		}
		tokens.push_back({str, -1, -1, false});
	}
	for (int i = 0; i < tokens.size() - 1; ++i) {
		std::string string1(1, input[i]);
		std::string string2(1, input[i + 1]);
		std::pair<std::string, std::string> tokenPair = {string1, string2};
		pairPositions[tokenPair].push_back(i);
		pairFrequencies[tokenPair]++;
		tokens[i].right = i + 1;
	}
	for (auto& p : pairFrequencies) {
		mergeQueue.push({p.second, p.first});
	}
	while (mergeQueue.size() > 0 && vocabulary.size() < vocabSize) {
		auto bestToken = mergeQueue.top();
		mergeQueue.pop();
		std::pair<std::string, std::string> bestPair = bestToken.second;
		if (pairFrequencies[bestPair] != bestToken.first) {
			if (pairFrequencies[bestPair] > 0) {
				mergeQueue.push({pairFrequencies[bestPair], bestPair});
			}
			continue;
		}
		std::string string1 = bestPair.first;
		std::string string2 = bestPair.second;
		std::string mergedStr = string1 + string2;
		std::map<std::pair<std::string, std::string>, int> newPairFrequencies;
		std::vector<int> current_positions = pairPositions[bestPair];
		for (int i = 0; i < current_positions.size(); ++i) {
			int pos = current_positions[i];
			if (pos < 0 || pos >= tokens.size() || tokens[pos].duplicate) continue;
			Token& merger = tokens[pos];
			int right_pos = merger.right;
			if (right_pos == -1 || tokens[right_pos].duplicate || 
			    tokens[right_pos].representation != string2 || 
			    merger.representation != string1) {
				continue;
			}
			Token& mergee = tokens[right_pos];
			int prev_pos = merger.left;
			int next_pos = mergee.right;
			if (prev_pos != -1) {
				Token& prevToken = tokens[prev_pos];
				std::pair<std::string, std::string> old_pair = {prevToken.representation, merger.representation};
				--pairFrequencies[old_pair];
				pairPositions[old_pair].erase(std::remove(pairPositions[old_pair].begin(), pairPositions[old_pair].end(), prev_pos), pairPositions[old_pair].end());
				std::pair<std::string, std::string> new_pair = {prevToken.representation, mergedStr};
				++pairFrequencies[new_pair];
				pairPositions[new_pair].push_back(prev_pos);
				newPairFrequencies[new_pair]++;
			}
			if (next_pos != -1) {
				Token& nextToken = tokens[next_pos];
				std::pair<std::string, std::string> old_pair = {mergee.representation, nextToken.representation};
				--pairFrequencies[old_pair];
				pairPositions[old_pair].erase(std::remove(pairPositions[old_pair].begin(), pairPositions[old_pair].end(), right_pos), pairPositions[old_pair].end());
				std::pair<std::string, std::string> new_pair = {mergedStr, nextToken.representation};
				++pairFrequencies[new_pair];
				pairPositions[new_pair].push_back(pos);
				newPairFrequencies[new_pair]++;
				nextToken.left = pos;
			}
			if (prev_pos != -1) {
				tokens[prev_pos].right = pos;
			}
			merger.right = next_pos;
			merger.representation = mergedStr;
			mergee.duplicate = true;
			vocabulary[mergedStr] = 0;
		}
		for (auto& p : newPairFrequencies) {
			mergeQueue.push({pairFrequencies[p.first], p.first});
		}
	}
	int idx = 0;
	for (auto& kv : vocabulary) {
		kv.second = idx++;
	}
	for (int i = 0; i < tokens.size(); ++i) {
		if (tokens[i].duplicate) continue;
		output.push_back(vocabulary[tokens[i].representation]);
	}
}
#endif
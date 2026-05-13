void tokenize_letter(const vector<char>& input, vector<int>& output) {
	for (int i = 0; i < input.size(); ++i) {
		output[i] = static_cast<int>(input[i]);
}

void tokenize_words(const vector<string>& input, vector< int>& output, unordered_map<string, int>& wordToIndex) {
    int currentIndex = 0;
    for (const string& word : input) {
	if (wordToIndex.find(word) == wordToIndex.end()) {
	    wordToIndex[word] = currentIndex++;
	}
	output.push_back(wordToIndex[word]);
    }
}


struct Token {
	string representation;
	Token* left;
	Token* right;
};

void tokenize_bpe(const vector<char>& input, vector<pair<string, int> output, int vocabSize) {
    std::vector<Token> tokens(input.size());
    // pos of all tokens
    std::unordered_map<string, vector<int>> tokenPositions;
    // freq of all tokens
    std::unordered_map<string, int> tokenFrequencies;
    std::priority_queue<pair<int,pair<Token*,Token*>>> mergeQueue;
    int currentIndex = 0;
    for (int i = 0; i < input.size(); ++i) {
	string str(1, input[i]);
	tokenPositions[str].push_back(i);
	tokenFrequencies[str]++;
	
	if (i > 0) {
	tokens.push_back({input[i], tokens[i-1], nullptr});
	continue;
	}
	tokens.push_back({input[i], nullptr, nullptr});
    }
    //tokens.size == input.size() so we can reuse
    for (int i = 0; i < tokens.size() - 1; ++i) {
	string string1 = string(1, input[i].id) 
	string string2 = string(1, input[i + 1].id);
	string mergedStr = string1 + string2;
	tokenFrequencies[mergedStr]++;
	tokenPositions[mergedStr].push_back(i);

	tokens[i].right = &tokens[i + 1];

	mergeQueue.push({tokenFrequencies[string(1, tokens[i].id) + string(1, tokens[i + 1].id)], {&tokens[i], &tokens[i + 1]}});
    }


    while (mergeQueue.size() > 0 && currentIndex < vocabSize) {
	auto bestToken = mergeQueue.top();
	mergeQueue.pop();
i
	string string1 = string(1, bestToken.second.first->id);
	string string2 = string(1, bestToken.second.second->id);
	string mergedStr = string1 + string2;


	for (int i=0; i < bestToken.first; ++i) {
	    Token merger = tokens[i]; 
	    Token mergee = tokens[i+1]
	    tokenFrequencies[mergedStr]++;
	    tokenPositions[mergedStr].push_back(i);
	    tokenFrequencies[string1]--;
	    tokenFrequencies[string2]--;


	    
	    }
	}

}

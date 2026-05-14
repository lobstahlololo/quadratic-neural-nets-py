# Tokenizers
Tokenizers convert text into numbers that the network can process. `boilerplate/tokenizers.h` provides three tokenizers.
## Character-Level Tokenizer
Converts each character to its ASCII value.
```
void tokenize_letter(const vector<char>& input, vector<int>& output);
```
- `input` — the text as a vector of characters
- `output` — filled with integer IDs (one per character)
Example:
```
std::vector<char> text = {'h', 'e', 'l', 'l', 'o'};
std::vector<int> token_ids(text.size());
tokenize_letter(text, token_ids);
```
**Pros:** Simple, vocabulary size is small (256 for ASCII)
**Cons:** Long sequences, no understanding of words or subwords
Use for: simple experiments, very small datasets
## Word-Level Tokenizer
Splits text into words and assigns each unique word an index.
```
void tokenize_words(
    const vector<string>& input,
    vector<int>& output,
    unordered_map<string, int>& wordToIndex
);
```
- `input` — list of words (already split)
- `output` — filled with word indices
- `wordToIndex` — map from word string to index, updated with new words
Example:
```
std::vector<std::string> words = {"the", "hero", "leaves", "the", "cave"};
std::vector<int> token_ids;
std::unordered_map<std::string, int> word_to_index;
tokenize_words(words, token_ids, word_to_index);
```
**Pros:** Intuitive, shorter sequences than character-level
**Cons:** Large vocabulary, unknown words are a problem, loses sub-word information
Use for: small to medium datasets with limited vocabulary
## Splitting Text into Words
The tokenizer expects pre-split words. Here's how to split text:
```
std::string text = "The hero leaves home.";
std::vector<std::string> words;
std::string current_word;
for (char character : text) {
    if (character == ' ' || character == '.' || character == '?' || character == '!') {
        if (!current_word.empty()) {
            words.push_back(current_word);
            current_word.clear();
        }
    } else {
        current_word += character;
    }
}
if (!current_word.empty()) {
    words.push_back(current_word);
## BPE Tokenizer
Byte-Pair Encoding merges frequently co-occurring character pairs into new tokens. The implementation in `tokenizers.h` is a fully functional BPE trainer and tokenizer.
```
void tokenize_bpe(
    const vector<char>& input,
    vector<int>& output,
    unordered_map<string, int>& vocabulary,
    int vocabSize
);
```
- `input` — the raw text as a vector of characters
- `output` — filled with BPE token IDs
- `vocabulary` — filled with the learned BPE vocabulary mapping string -> ID
- `vocabSize` — the maximum number of tokens to learn

Example:
```
std::vector<char> text = {'t', 'h', 'e', ' ', 't', 'h', 'e', 'y'};
std::vector<int> token_ids;
std::unordered_map<std::string, int> vocab;
tokenize_bpe(text, token_ids, vocab, 100);
```
This will progressively merge pairs (e.g., 't'+'h' -> 'th', 'th'+'e' -> 'the') until the vocabulary reaches the target size.

## Choosing a Tokenizer
- **Character-Level (`tokenize_letter`)**: Easiest to set up, tiny vocabulary (256). Network needs a much larger sequence length to understand full sentences. Good for absolute basics.
- **Word-Level (`tokenize_words`)**: Fast training since sequence length is small (1 word = 1 token). However, vocabulary grows uncontrollably on large datasets, and unseen words crash the system or get mapped to UNK.
- **Byte-Pair Encoding (`tokenize_bpe`)**: The industry standard. Combines the best of both: small vocabulary size (defined by you), handles unseen words (splits them into known sub-words or characters), and compresses text efficiently.

## Using Token IDs as Network Input
The network expects float inputs. Convert token IDs:
```
std::vector<float> float_tokens(token_ids.size());
for (size_t i = 0; i < token_ids.size(); ++i) {
    float_tokens[i] = static_cast<float>(token_ids[i]);
}
```
When using `EmbeddingLayer`, these float token IDs are cast back to integers inside the embedding hook to look up the correct row in the embedding table.
## Preparing Sequences for Language Models
For next-token prediction, create input-target pairs:
```
int sequence_length = 10;
std::vector<float> training_data;
std::vector<float> targets;
std::vector<std::vector<int>> correct_indices;
for (size_t i = 0; i + sequence_length < token_ids.size(); ++i) {
    for (int j = 0; j < sequence_length; ++j) {
        training_data.push_back(static_cast<float>(token_ids[i + j]));
    }
    int next_token = token_ids[i + sequence_length];
    std::vector<float> one_hot(vocabulary_size, 0.0f);
    one_hot[next_token] = 1.0f;
    targets.insert(targets.end(), one_hot.begin(), one_hot.end());
    correct_indices.push_back({next_token});
}
```
## Adding Special Tokens
Reserve indices for special tokens like start-of-sequence, end-of-sequence, padding, and unknown:
```
enum SpecialTokens {
    PAD_TOKEN = 0,
    START_TOKEN = 1,
    END_TOKEN = 2,
    UNKNOWN_TOKEN = 3
};
int first_word_index = 4;
```
Update `vocabulary_size` to include these special tokens:
```
int vocabulary_size = word_to_index.size() + 4; // 4 special tokens
```
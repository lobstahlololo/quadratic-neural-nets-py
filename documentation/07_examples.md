# Examples
The `documentation/` folder contains two complete working examples.
## MNIST Digit Classifier
**File:** `documentation/example_mnist.cpp`
**What it does:** Trains a quadratic neural network to recognise handwritten digits (0-9) from the MNIST dataset.
**Architecture:**
- Input: 784 pixels (28×28 image flattened)
- Hidden layer 1: 256 quadratic neurons with ReLU
- Hidden layer 2: 128 quadratic neurons with ReLU
- Output: 10 quadratic neurons with Softmax
**Loss:** Combined O(n) system — `CrossEntropyLossForSoftmax` + `CrossEntropyLossForSoftmaxDerivative`
**To run:**
1. Download MNIST files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`)
2. Compile:
```
g++ -std=c++17 -DTRAINING_ON documentation/example_mnist.cpp model/network.cpp train/train.cpp math/math.cpp -o mnist_classifier
```
3. Run:
```
./mnist_classifier
```
**What you'll see:** The program trains for 20 epochs and prints "Training finished." Add your own accuracy printing to monitor progress.
---
## Language Model (Hero's Journey)
**File:** `documentation/example_language_model.cpp`
**What it does:** Trains a tiny transformer-like language model on 10 paragraphs of a Hero's Journey story. Predicts the next word given 10 previous words.
**Architecture:**
- Embedding: ~30 vocabulary → 32 dimensions
- Learnable Layer Norm
- Single-head Self-Attention (sequence length 10)
- Feed-forward: 32 → 64 with ReLU
- Learnable Layer Norm
- Output projection: 64 → vocabulary size with Softmax
**Loss:** Combined O(n) system — `CrossEntropyLossForSoftmax` + `CrossEntropyLossForSoftmaxDerivative`
**Training data:** Built-in Hero's Journey text (10 paragraphs, ~50 words total)
**To run:**
```
g++ -std=c++17 -DTRAINING_ON documentation/example_language_model.cpp model/network.cpp train/train.cpp math/math.cpp -o language_model
./language_model
```
**What you'll see:** The program trains for 30 epochs and prints "Language model trained on Hero's Journey."
**To generate text after training:** You would need to add inference code that:
1. Takes a seed sequence of 10 words
2. Runs the forward pass to get probabilities
3. Samples the next word
4. Slides the window and repeats
---
## Building Your Own Example
Start from one of these templates:
1. Copy `example_mnist.cpp` for classification tasks
2. Copy `example_language_model.cpp` for sequence/language tasks
3. Replace the data loading with your own
4. Adjust the architecture to match your data dimensions
5. Choose the loss system that fits your needs
Key things to change:
- Input layer size (match your data)
- Hidden layer sizes
- Output layer size (match your number of classes/vocabulary)
- `batch_size`
- Number of epochs
- Learning rate
---
## Compilation Checklist
Make sure you include all needed source files:
```
g++ -std=c++17 -DTRAINING_ON \
    your_file.cpp \
    model/network.cpp \
    train/train.cpp \
    math/math.cpp \
    -o your_program
```
Optional: add GPU support
```
g++ -std=c++17 -DTRAINING_ON -DQQ_BLAS_CUBLAS \
    your_file.cpp model/network.cpp train/train.cpp math/math.cpp \
    -lcublas -lcudart -o your_program
```
---
## Common Issues
**"wrong filepath" error:** You tried to load weights from a file that doesn't exist. Use `""` (empty string) for random initialisation with the default Xavier method.
**"wrong file size relation to weights" error:** The weight file size doesn't match the network size. Check that the file was saved from a network with the same architecture.
**Network not learning:** Try lowering the learning rate, increasing epochs, or checking that your data is normalised to [0, 1] range.
**Gradient explosion:** Add gradient clipping in the training loop, or reduce the learning rate.

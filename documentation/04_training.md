# Training
## Training Functions
`train/train.h` provides two functions:
### `train()` — Single Training Step
```
void train(
    std::vector<Layer>& layers,
    const std::vector<float>& training_data,
    const std::vector<std::vector<int>>& correct_indices,
    const std::vector<float>& required_output,
    float learning_rate,
    const LossFunc& loss_function,
    const LossDerivative& loss_derivative,
    int step
);
```
Performs one forward pass, backward pass, and weight update. The `step` parameter tracks how many updates have happened (used for Adam bias correction).
### `trainScheduler()` — Full Training Loop
```
void trainScheduler(
    std::vector<Layer>& layers,
    const std::vector<float>& training_data,
    const std::vector<std::vector<int>>& correct_indices,
    std::vector<float> required_output,
    float learning_rate,
    float minimum_learning_rate,
    const LossFunc& loss_function,
    const LossDerivative& loss_derivative,
    int total_epochs = 10,
    int batch_size_arg = 4
);
```
Runs multiple epochs with:
- Cosine learning rate decay from `learning_rate` down to `minimum_learning_rate`
- Adam-style momentum (first and second moment estimates)
- Automatic batching
## Preparing Training Data
### Data format
`training_data` is a flat `std::vector<float>` containing all input samples concatenated:
```
training_data = [sample0_feat0, sample0_feat1, ..., sample0_featN,
                 sample1_feat0, sample1_feat1, ..., sample1_featN,
                 ...]
```
### Targets
`required_output` has the same layout as the network output. For classification with 10 classes:
```
targets = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,   // sample 0 (class 2)
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0,   // sample 1 (class 0)
           ...]
```
### Correct indices
`correct_indices` stores the correct class index for each sample:
```
correct_indices = [{2}, {0}, {5}, ...]
correct_indices = [{2}, {0}, {5}, ...]
```
## Setting Batch Size
Set the global `batch_size` before calling `setupNeuralNetwork()`:
```
batch_size = 64;
setupNeuralNetwork(architecture);
```
The `batch_size_arg` parameter in `trainScheduler` overrides the batch size for training only (doesn't change network setup).
## What Happens in One Training Step
1. **Forward pass:** Each layer's forward function runs in order
2. **Loss computation:** The loss function compares outputs to targets
3. **Loss gradient:** The loss derivative computes the initial gradient
4. **Backward pass:** Each layer's backward function runs in reverse order, computing weight gradients
5. **Weight update:** Adam optimiser updates weights using first and second moments
## Adam Optimiser Details
The optimiser uses:
- First moment decay: 0.9
- Second moment decay: 0.999
- Epsilon: 1e-8 (for numerical stability)
- Bias correction: applied using the `step` counter
Weights are updated as:
```
first_moment = 0.9 * first_moment + 0.1 * gradient
second_moment = 0.999 * second_moment + 0.001 * gradient^2
corrected_first = first_moment / (1 - 0.9^step)
corrected_second = second_moment / (1 - 0.999^step)
weight -= learning_rate * corrected_first / (sqrt(corrected_second) + 1e-8)
```
## Learning Rate Schedule
`trainScheduler` uses cosine decay:
```
current_lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * step / steps_per_epoch)) / 2
```
This smoothly decreases the learning rate from `learning_rate` to `minimum_learning_rate` over each epoch, then resets for the next epoch.
## Example Training Loop
```
batch_size = 32;
std::vector<LayerArgs> architecture;
setupNeuralNetwork(architecture, "", he_initialisation);
int total_epochs = 50;
float initial_learning_rate = 0.01f;
float minimum_learning_rate = 0.0001f;
trainScheduler(layers, training_data, correct_indices, targets,
               initial_learning_rate, minimum_learning_rate,
               CrossEntropyLossForSoftmax,
               CrossEntropyLossForSoftmaxDerivative,
               total_epochs, batch_size);
```
## Manual Training (Without Scheduler)
If you want full control, use `train()` directly in your own loop:
```
for (int step = 0; step < total_steps; ++step) {
    float current_lr = compute_my_learning_rate(step);
    train(layers, batch_data, batch_indices, batch_targets,
          current_lr, my_loss, my_loss_derivative, step);
}
```

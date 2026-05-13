# Loss Functions
`boilerplate/losses.h` provides loss functions for training. There are two complete systems for classification. Pick one.
## System 1: Generic Softmax + CrossEntropyLoss
Use this when you want a standard softmax that works with any loss function.
### What you need
- Output layer: `Softmax` as forward hook, `SoftmaxDerivative` as backward hook
- Loss: `CrossEntropyLoss` + `CrossEntropyLossDerivative`
### How it flows
```
Forward:  logits → Softmax → probabilities → CrossEntropyLoss → loss value
Backward: CrossEntropyLossDerivative → (prob - one_hot)/N
          SoftmaxDerivative → full Jacobian * (prob - one_hot)/N
          Result: gradient w.r.t. logits
```
### Code
```
LayerArgs output_layer;
output_layer.layer_size = num_classes;
output_layer.kind = Quadratic;
output_layer.outputs_per_neuron = 1;
output_layer.hooks = { Softmax };
output_layer.hook_gradients = { SoftmaxDerivative };
architecture.push_back(output_layer);
trainScheduler(layers, data, correct_indices, targets,
               learning_rate, min_learning_rate,
               CrossEntropyLoss,
               CrossEntropyLossDerivative,
               epochs, batch_size);
```
### When to use
- You might swap the loss for something else later
- You need to inspect softmax probabilities separately
- You want the standard textbook implementation
### Performance
O(n²) per sample because `SoftmaxDerivative` computes the full softmax Jacobian.
---
## System 2: Combined O(n) — SoftmaxForCrossEntropyLoss + CrossEntropyLossForSoftmax
Use this for the standard fast softmax-cross-entropy combination.
### What you need
- Output layer: `Softmax` as forward hook, `SoftmaxForCrossEntropyLossDerivative` as backward hook
- Loss: `CrossEntropyLossForSoftmax` + `CrossEntropyLossForSoftmaxDerivative`
### How it flows
```
Forward:  logits → Softmax → probabilities → CrossEntropyLossForSoftmax → loss value
Backward: CrossEntropyLossForSoftmaxDerivative → just copies probabilities (no-op)
          SoftmaxForCrossEntropyLossDerivative → computes softmax(logits) - one_hot
          Result: gradient w.r.t. logits
```
The key insight: when softmax is followed by cross-entropy, the combined gradient simplifies to `softmax(logits) - one_hot`. The softmax backward hook computes this directly in O(n). The loss derivative does nothing (no-op).
### Code
```
LayerArgs output_layer;
output_layer.layer_size = num_classes;
output_layer.kind = Quadratic;
output_layer.outputs_per_neuron = 1;
output_layer.hooks = { Softmax };
output_layer.hook_gradients = { SoftmaxForCrossEntropyLossDerivative };
architecture.push_back(output_layer);
trainScheduler(layers, data, correct_indices, targets,
               learning_rate, min_learning_rate,
               CrossEntropyLossForSoftmax,
               CrossEntropyLossForSoftmaxDerivative,
               epochs, batch_size);
```
### When to use
- Standard choice for classification
- You want the fastest option
- You know you'll always use cross-entropy with softmax
### Performance
O(n) per sample. Faster than System 1.
---
## Quick Reference Table
| System | Forward Hook | Backward Hook | Loss Function | Loss Derivative | Speed |
|---|---|---|---|---|---|
| Generic | `Softmax` | `SoftmaxDerivative` | `CrossEntropyLoss` | `CrossEntropyLossDerivative` | O(n²) |
| Combined | `Softmax` | `SoftmaxForCrossEntropyLossDerivative` | `CrossEntropyLossForSoftmax` | `CrossEntropyLossForSoftmaxDerivative` | O(n) |
---
## Regression Loss
For regression (predicting numbers, not classes), use Mean Squared Error:
```
float MeanSquaredErrorLoss(const float* predicted, const float* required,
                           const std::vector<int>& required_indices, int size);
void MeanSquaredErrorLossDerivative(float loss, const float* predicted,
                                     const float* required,
                                     const std::vector<int>& required_indices,
                                     float* output, int size);
```
No softmax needed for regression.
```
trainScheduler(layers, data, correct_indices, targets,
               learning_rate, min_learning_rate,
               MeanSquaredErrorLoss,
               MeanSquaredErrorLossDerivative,
               epochs, batch_size);
```
---
## Writing a Custom Loss
A loss function receives:
- `predicted` — the network output
- `required` — the target values
- `required_indices` — for classification, the correct class index per sample
- `size` — number of output features
It returns a single float (the loss value, typically averaged over the batch).
A loss derivative receives the same plus:
- `loss` — the loss value (rarely used, included for completeness)
- `output` — buffer to fill with the gradient w.r.t. `predicted`
Example — a custom absolute error loss:
```
float MeanAbsoluteErrorLoss(const float* predicted, const float* required,
                             const std::vector<int>& required_indices, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += std::abs(predicted[i] - required[i]);
    }
    return sum / size;
}
void MeanAbsoluteErrorLossDerivative(float loss, const float* predicted,
                                      const float* required,
                                      const std::vector<int>& required_indices,
                                      float* output, int size) {
    float scale = 1.0f / size;
    for (int i = 0; i < size; ++i) {
        float diff = predicted[i] - required[i];
        output[i] = scale * (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
    }
}
```

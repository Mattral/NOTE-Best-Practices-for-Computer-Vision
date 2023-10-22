
# Training Neural Networks

Training neural networks is the process of teaching a model to make accurate predictions by learning from data. This involves iterative steps of forward and backward propagation, optimization, and the use of loss functions to quantify the model's performance. In this note, we'll delve into the training process, introducing mathematical concepts gradually.

## Forward Propagation

Forward propagation is the initial phase of training in which data is passed through the network to compute predictions. It involves several key steps:

1. **Input Layer:** The raw input data, often represented as a feature vector, is provided to the input layer of the neural network.

2. **Weighted Sum:** In each neuron (or node), a weighted sum of the input data is computed using the following equation:
   
   Weighted Sum=∑ 
i=1 
​
 (w 
i
​
 ⋅x 
i
​
 )+b

   Where:
   - Weighted Sum is the result of the weighted sum.
   - \(w_i\) are the weights.
   - \(x_i\) are the input values.
   - \(b\) is the bias term.

3. **Activation Function:** After computing the weighted sum, an activation function is applied to introduce non-linearity. Common activation functions include the sigmoid, ReLU, and tanh functions.

4. **Output:** The result of the activation function is the output of the neuron and serves as the input to the subsequent layer.

This process continues through each layer until the final output is computed.

## Backward Propagation

Backward propagation, often referred to as backpropagation, is the phase of training where the model learns from its errors. It involves calculating gradients with respect to model parameters (weights and biases) and using them to adjust these parameters. Key steps include:

1. **Loss Function:** A loss function is used to measure the difference between the model's predictions and the actual target values. Common loss functions include mean squared error for regression and cross-entropy for classification.

2. **Gradient Computation:** The gradients of the loss function with respect to the model parameters are computed using the chain rule. This is done by calculating how changes in each parameter affect the loss.

3. **Parameter Updates:** Optimization algorithms, such as gradient descent, use the computed gradients to adjust the model's parameters to minimize the loss. The parameter update is typically done using the following equation:

   \[ \text{New Parameter Value} = \text{Old Parameter Value} - \text{Learning Rate} \times \text{Gradient} \]

   Where:
   - \(\text{New Parameter Value}\) is the updated parameter value.
   - \(\text{Old Parameter Value}\) is the current parameter value.
   - \(\text{Learning Rate}\) controls the size of the update (a hyperparameter).
   - \(\text{Gradient}\) is the computed gradient.

This process of forward and backward propagation continues for a set number of iterations or until a convergence criterion is met.

## Training Objective

The objective of training is to minimize the loss function, which quantifies the model's error. As the model iteratively updates its parameters using gradients, it gradually improves its ability to make accurate predictions on the training data.


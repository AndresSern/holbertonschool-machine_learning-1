# binary classification
This is ‘Classification’ tutorial which is a part of the Machine Learning course offered by Simplilearn. We will learn Classification algorithms, types of classification algorithms, support vector machines(SVM), Naive Bayes, Decision Tree and Random Forest Classifier in this tutorial.

0. Neuron 
Write a class Neuron that defines a single neuron performing binary classification

1. Privatize Neuron
with Private instance attributes

2. Neuron Forward Propagation
with sigmoid activation function
![alt text](https://www.gstatic.com/education/formulas2/-1/en/sigmoid_function.svg)

3. Neuron loss and  Cost
using logistic regression
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each
```sh
loss = -(Y * log(A) + (1 - Y) * log(1 - A))
cost = 1/m (sum(loss))
```
4. Evaluate Neuron 
values should be 1 if the output of the network is >= 0.5 and 0 otherwise
5. Neuron Gradient Descent
Updates the private attributes __W and __b
self.__W -= (dw * alpha)
self.__b -= (db * alpha)

6. Train Neuron

Returns the evaluation of the training data after
iterations of training have occurred
used:
-forward_prop
-gradient_descent
-evaluate

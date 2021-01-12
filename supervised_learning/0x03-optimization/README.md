 # 0x03. Optimization
 
Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

## 1. Normalize
In machine learning, we can handle various types of data. audio signals and pixel values for image data, and this data can include multiple dimensions. Feature standardization makes the values of each feature in the data have zero-mean (when subtracting the mean in the numerator) and unit-variance. This method is widely used for normalization in many machine learning algorithms

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/b0aa2e7d203db1526c577192f2d9102b718eafd5)

Where x is the original feature vector, x bar  is the mean of that feature vector, and sigma  is its standard deviation.

## 3. Mini-Batch
Batch gradient descent is a variation of the gradient descent algorithm that calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated
What is Mini-Batch Gradient Descent?
Mini-batch gradient descent is a variation of the gradient descent algorithm that splits the training dataset into small batches that are used to calculate model error and update model coefficients.

Implementations may choose to sum the gradient over the mini-batch which further reduces the variance of the gradient.
## 5. Momentum optimization algorithm
that updates a variable using the gradient descent with momentum optimization algorithm:

* alpha is the learning rate
* beta1 is the momentum weight
* var is a numpy.ndarray containing the variable to be updated
* grad is a numpy.ndarray containing the gradient of var
* v is the previous first moment of var
* Returns: the updated variable and the new moment, respectively

```sh
    v = (beta1*dw_prev) + ((1- beta1)*dw)
    w = w - (alpha *v)
```
## 7. RMSProp optimization algorithm
* alpha is the learning rate
* beta2 is the RMSProp weight
* epsilon is a small number to avoid division by zero
* var is a numpy.ndarray containing the variable to be updated
* grad is a numpy.ndarray containing the gradient of var
* s is the previous second moment of var
* Returns: the updated variable and the new moment, respectively
```sh
    vdw = (beta2 * s) + ((1 - beta2)*grad**2)
    new = var - alpha * (grad / (np.sqrt(vdw)+epsilon))
```
## 9.Adam optimization algorithm
>Adam use  Momentum optimization algorithm and  RMSProp optimization algorithm

* alpha is the learning rate
* beta1 is the weight used for the first moment
* beta2 is the weight used for the second moment
* epsilon is a small number to avoid division by zero
* var is a numpy.ndarray containing the variable to be updated
* grad is a numpy.ndarray containing the gradient of var
* v is the previous first moment of var
* s is the previous second moment of var
* t is the time step used for bias correction
* Returns: the updated variable, the new first moment, and the new second moment, respectively

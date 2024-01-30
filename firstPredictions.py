import numpy as np

#Will start introducing non-linear equations(sigmoid function) and create an AI that will predict a 1 or 0 for true or false

"""
Task:
    Create a neural network to predict a result as true or false (1 or 0).
Summary:
    This will be a neural netowork with only 2 layers, this can take in an input and produce eitehr a true or false answer.
    In order to do this I will be using the dot product like I did last time, but there is one issue with the dot product.
    The dot product is a linear equation; this means that when adding new layers it will always affect the next layer in 
    some way. I need a way to have the middle layer sometimes corelate with the input and sometimes not. In order to do
    this I will be using an Activation Function. An activation function is a non-linear equation that will do the task
    I'm wanting it to do. For this neural network I will be using the sigmoid function becausee the neural network 
    I'm creating is following the Bernoulli distribution, which limits the outputs to 0 or 1, and the sigmoid function
    is just the one to use because idk, it just is. There is also more activation functions othet then the sigmoid function.
Sigmoid Function: 
    S(x)=1/(1+e^-x) #I will using np.exp(x) to calculate e^x
Layer functions:
    Layer 1:
            Layer 1 will take in the input and weights, it will then computate the dot product, add them together,
            and finally will add the bias into them. They will then send the layer_1_result into the layer 2.
    Layer 2:
            Layers 2 will then take the layer_1_result, computate the raw prediction with the sigmoid function, and return the
            raw prediction.
Prediction:
    The prediction is either 1 or 0 based off the raw prediction. if the raw prediction >= 0.5, it returns a 1,
    else if the raw prediction < 0.5, it returns a 0.

"""

# Wrapping the vectors in NumPy arrays
input_vector_1 = np.array([1.66, 1.56])
input_vector_2 = np.array([2, 1.5])
weights_1 = np.array([1.45, -0.66])

# Haven't calculated bias yet
bias = np.array([0.0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

prediction1 = make_prediction(input_vector_1, weights_1, bias)
print(f"The prediction 1 result is: {prediction1}")
# The prediction result is: [0.7985731] Correct

prediction2 = make_prediction(input_vector_2, weights_1, bias)
print(f"The prediction 2 result is: {prediction2}")
# The prediction result is: [0.87101915] Wrong

"""

Task:
    Find a way to measure the error
Summary:
    I will be using a cost function(or loss function) to measure how much the ai is off by. For this network I will be using a
    Mean Squared Error. To calculate this you compute the difference between the prediction and the target, then square the result.
    Squaring this difference makes sure we always get a positive difference.


"""

# The target for input_vector_2 is 0
target = 0

# Squaring always gives a positive result
# Mean squared error
mse = np.square(prediction2 - target)

print(f"Prediction: {prediction2}; Error: {mse}")
# Prediction: [0.87101915]; Error: [0.7586743596667225]

"""
Task:
    Understand how to reduce the error using a 
    1. gradient decent
    2. backpropagation
Summary:
    First we need to create the gradient decent. The goal is to change the weights and bias variables to reduce the mse.
    For now I will just change the weights and leave the bias variables alone to try to understand it.
    
    To calculate mse we do (prediction-target)^2. If we look at this and just treat prediction-target as a variable we get
    (x)^2. This is a quadtratic function. The error is given by the y-axis, and in order to reduce the error if we're on the
    right side of the y-axis we subtract from the x-value, but if we're on the left side of the y-axis we add to the x-value.
    In order to know which way to go we calculate the derivative of the function.
Derivative(TODO):
    A derivative explains exactly how a pattern will change. Another word for derivtive is a gradient, and the gradient
    decent algorithim is the one I will be using for this netowrk.
    
    Derivative Rules:
        Power Rule:
            The power rule states that the derivative on x^n == to nx^(n-1), so, the derivative of x^2 == 2*x, and x == 1

"""

derivative = 2*(prediction2-target)
print(f'the derivative is {derivative}')

weights_1 -= derivative

new_prediction_2 = make_prediction(input_vector_2,weights_1,bias)
new_error = np.square(new_prediction_2-target)
print(f"The new prediction 2 result is: {new_prediction_2}; error is {new_error}")
# new_prediction_2: [0.01496248]; error: [0.00022388]

"""
Conclusion:
    Now that our error is close to zero our product is more accurate. Our derivative for this example was a fairly small number
    but sometimes our derivative is to big and will just go right over the desired error value, which is zero. So instead we will
    take a fraction of the derivative in order to ger as close as possible to zero. 
"""
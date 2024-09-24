/**
* Multilayer Perceptron Neural Network (MLP)
* By William Alves Jardim
* 
* This implementation is entirely original, written from scratch in JavaScript.
* It was inspired by various publicly available resources, including concepts 
* and explanations from the work of Jason Brownlee on backpropagation.
* 
* CREDITS && REFERENCE:
* Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
* For more details, see README.md.
*/

//Activation functions
net.activations = {};

net.activations.sigmoid = function(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

//Just the derivative of sigmoid
net.activations.sigmoid.derivative = function(functionResult){
    return functionResult * (1.0 - functionResult);
}

net.activations.relu = function(x) {
    return Math.max(0, x);
}

//Just the derivative of sigmoid
net.activations.relu.derivative = function(functionResult){
    return functionResult > 0 ? 1 : 0;
}
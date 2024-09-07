# MLP
Multilayer Perceptron Neural Network in JavaScript.
By William Alves Jardim

![Logo](./images/logo/logo256x256.png)

# CREDITS / REFERENCES
**Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.**

# Description
I wrote the functions in JavaScript. This project is an adaptation of Jason Brownlee's original code. I really liked the way his article taught backpropagation.

# Implementation Features
My implementation has the following features:

   - All are modularized in parts: Unit, Layer, MLP

   - Each part can be configurable via JSON attributes

   - Parameters like learning_rate are passed in a object called "hyperparameters"

   - The target(called desired value) of the samples are separated of features in two arrays: a array for the features, and another array for the targets

   - Do not convert class numbers to binary array. Instead, the code expects to receive an array containing the desired outputs for each output unit.

   - The heights and Bias are separated 

   - Use more aligned for loops instead of using more compact functions

   - Each layer can have a different activation function

   - The derivative of the activation function is stored inside the activation function, for example: sigmoid(x) is the function. Bot sigmoid.derivative(sigmoidOutput) is the derivative of the sigmoid

   - The Feedforward function have much intermediate variables, to be declarative

   - All the Backpropagation are centralized in a one function

   - In the Backpropagation, use L and L+1 indices for access the layers

   - In the Backpropagation function, in the end of method, will update the weights and bias

   - In the Backpropagation, do not use IF conditions to determine if is a output layer or if is a hidden layer. Instead, it follows a more sequential approach

   - Do not use Cross-Validation or k-fold

   - Not have a CSV reader included

   - Do not convert string to int or float

   - Do not have normalization

In some parts of the Feedforward and Backpropagation code, i used some similar strategies that are used by Jason Brownlee. Below are a list that describes this better:

  - **feedforward_sample:**

      - Use of a property called 'LAYER_INPUTS'(present in each layer object), to store the the outputs of the units of previous layer(L-1), that will be the inputs of the current layer(L).

      - Use of a variable property called 'ACTIVATION' to store the unit activation in the own unit object

      - Use of a variable property called 'INPUTS' to store the inputs that are used in unit in the own unit object

  - **backpropagate_sample:**
      
      - Using an aligned for to calculate unit deltas in the hidden layer

      - Use of a variable property called 'LOSS' to store the unit error in the own unit object

      - In final of code, they use the property INPUTS of the units in Gradient Descent(For update the weights)

# Examples that i used to test
I also used the same example dataset that he used in the article to test if my code follows the steps correctly. 

Also, the "examples" folder contains one example used in Jason Brownlee's article, I used it to test if my code worked

**I also left a copy of the original file in a folder called "original_code"**, [View the copy of the Jason Brownlee Code](./original_code/complete_original_code.py)

# Thank you note
Thank you, Jason! I'm grateful for having the opportunity to write an adaptation of your code, which was very good for my learning!
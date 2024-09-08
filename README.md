# MLP
Multilayer Perceptron Neural Network in JavaScript.
By William Alves Jardim

![Logo](./images/logo/logo256x256.png)

# CREDITS / REFERENCES
Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.

# Description
This project implements a Multilayer Perceptron (MLP) Neural Network in JavaScript. Users can configure the number of layers and units as desired, with support for ReLU and Sigmoid activation functions.

# Acknowledgment
This project drew inspiration from various sources, including general concepts and examples available in public domains, such as the work of Jason Brownlee on backpropagation, which was particularly informative for understanding certain neural network techniques.

# Implementation Features
My implementation has the following features:

   - All are modularized in parts: Unit, Layer, MLP, and each part can be configurable via JSON attributes

   - The entire Backpropagation and Gradient Descent processes are centralized in a single function called backpropagate_sample. At the end of this function, the weights and biases are updated. During Backpropagation, the layers are accessed using indices L and L+1.

In certain parts of the Feedforward and Backpropagation code, I implemented strategies that are similar to those outlined by Jason Brownlee. Below is a list detailing the specific areas where this approach was followed:

  - **feedforward_sample:**

      - Use of a property called 'LAYER_INPUTS'(present in each layer object), to store the the outputs of the units of previous layer(L-1), that will be the inputs of the current layer(L).

      - Use of a variable property called 'ACTIVATION' to store the unit activation in the own unit object

      - Use of a variable property called 'INPUTS' to store the inputs that are used in unit in the own unit object

  - **backpropagate_sample:**
      
      - Using an aligned for to calculate unit gradients in the hidden layer

      - Use of a variable property called 'LOSS' to store the unit error in the own unit object(each unit have a 'LOSS' property)

      - In final of code, they use the property INPUTS of the units in Gradient Descent(For update the weights)


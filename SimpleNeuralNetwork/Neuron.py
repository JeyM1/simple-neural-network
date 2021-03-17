import numpy as np
from SimpleNeuralNetwork.activation_functions import *


class Neuron:
    def __init__(self, n_inputs, learning_spd=1, activation_func=sigmoid, activation_func_deriv=sigmoid_derivative):
        self.weights = np.array([np.random.random() for i in range(n_inputs)])
        self.saved = self.weights.copy()
        self.activation_func = activation_func
        self.activation_func_deriv = activation_func_deriv
        self.learning_speed = learning_spd

    # going backwards through the network to update weights
    def backpropagation(self, inputs, outputs, results):
        error = outputs - results
        delta = self.learning_speed * error * self.activation_func_deriv(results)
        self.weights += np.dot(inputs.T, delta)

    def backpropagation_err(self, inputs, results, error):
        delta = self.learning_speed * error * np.array([self.activation_func_deriv(result) for result in results])
        self.weights += np.dot(inputs, delta)

    def train(self, inputs, outputs, epochs=25000):
        for epoch in range(epochs + 1):
            # flow forward and produce an output
            results = self.predict(inputs)
            if epoch == 1:
                print(results)
            loss = np.sum((outputs - results) ** 2)
            if epoch % 100 == 0:
                print("Epoch: {:7d} | Loss: {:f}".format(epoch, loss))
            # go back though the network to make corrections based on the output
            self.backpropagation(inputs, outputs, results)

    def save_weights(self):
        self.saved = self.weights.copy()

    def retrieve_saved(self):
        self.weights = self.saved.copy()

    def predict(self, new_input):
        return self.activation_func(np.dot(new_input, self.weights))

    def feed_forward(self, inputs):
        return [self.activation_func(np.dot(new_input, self.weights)) for new_input in inputs]

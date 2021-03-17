import numpy as np
from SimpleNeuralNetwork.Neuron import Neuron
from SimpleNeuralNetwork.activation_functions import *


class NeuralNetwork:
    def __init__(self, n_inputs, learning_spd=1):
        self.learning_speed = learning_spd
        self.layers = []
        self.n_inputs = n_inputs

    def add_layer(self, neurons_count, activation_func=sigmoid, activation_func_deriv=sigmoid_derivative):
        if not self.layers:
            # initialize neurons with n_inputs
            layer = [Neuron(self.n_inputs, self.learning_speed, activation_func, activation_func_deriv) for _ in
                     range(neurons_count)]
        else:
            layer = [Neuron(len(self.layers[-1]), self.learning_speed, activation_func, activation_func_deriv) for _ in
                     range(neurons_count)]
        self.layers.append(layer)

    def train(self, inputs, outputs, epochs=25000):
        loss_min = 10000000
        best_epoch = -1
        for epoch in range(epochs + 1):
            # flow forward and produce an output
            layers_results = [inputs.T, np.array([n.feed_forward(inputs) for n in self.layers[0]])]
            for layer in self.layers[1:]:
                layers_results.append(np.array([n.feed_forward(layers_results[-1].T) for n in layer]))
            loss = np.sum((outputs - layers_results[-1]) ** 2)
            if epoch % 10000 == 0:
                print("Epoch: {:7d} | Loss: {:f}".format(epoch, loss))

            if loss <= loss_min:
                loss_min = loss
                best_epoch = epoch
                for layer in self.layers:
                    for neuron in layer:
                        neuron.save_weights()

            # go back though the network to make corrections based on the output
            self.backpropagation(outputs, layers_results)
        print("Best epoch is:", best_epoch, f"({loss_min} loss)")
        for layer in self.layers:
            for neuron in layer:
                neuron.retrieve_saved()

    def backpropagation(self, outputs, layers_results):
        errors = outputs - layers_results[-1]
        # feed last output in last layer
        for i in range(len(self.layers[-1])):
            self.layers[-1][i].backpropagation_err(layers_results[-2], layers_results[-1][i], errors[i])

        # going through reversed layers WITHOUT last
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            layer_errors = []
            for n in range(len(self.layers[layer_idx + 1])):
                for w in self.layers[layer_idx + 1][n].weights:
                    layer_errors.append(errors[n] * w)

            # for each neuron on that layer - backpropagation
            for neuron_idx in range(len(self.layers[layer_idx])):
                self.layers[layer_idx][neuron_idx].backpropagation_err(
                    layers_results[layer_idx],
                    layers_results[layer_idx + 1][neuron_idx],
                    layer_errors[neuron_idx]
                )
            errors = layer_errors

    def predict(self, new_input):
        layer_res = np.array([n.predict(new_input) for n in self.layers[0]])
        for layer_idx in range(1, len(self.layers)):
            layer_res = np.array([n.predict(layer_res) for n in self.layers[layer_idx]])
        return layer_res

    def print_weights(self):
        weights = [[[ws for ws in ns.weights] for ns in layer] for layer in self.layers]
        for layer_idx in range(len(weights)):
            print("layer ", layer_idx)
            print(np.array(weights[layer_idx]))

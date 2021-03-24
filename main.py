import numpy as np
from SimpleNeuralNetwork import *


def main():
    inputs = np.array([[0.20, 5.14, 0.47],
                       [5.14, 0.47, 4.37],
                       [0.47, 4.37, 1.22],
                       [4.37, 1.22, 4.29],
                       [1.22, 4.29, 1.89],
                       [4.29, 1.89, 4.51],
                       [1.89, 4.51, 0.32],
                       [4.51, 0.32, 5.80],
                       [0.32, 5.80, 1.37],
                       [5.80, 1.37, 5.77]]) / 10
    outputs = np.array([4.37, 1.22, 4.29, 1.89, 4.51, 0.32, 5.80, 1.37, 5.77, 0.88]) / 10

    nn = NeuralNetwork(3, learning_spd=.1)

    def weights_init(x):
        return x * 0.1 + 0.1

    nn.add_layer(6, activation_func=activation_functions.relu,
                 activation_func_deriv=activation_functions.relu_deriv, neuron_weight_init=weights_init)
    nn.add_layer(1)
    nn.train(inputs, outputs, 10000)
    print("Final weights:")
    nn.print_weights()

    for i in range(len(inputs)):
        print('Predicted: ', nn.predict(inputs[i]) * 10,
              '\t- Correct: ', outputs[i] * 10)
    print("----------------------------------------------")
    print('Predicted: ', nn.predict(np.array([1.37, 5.77, 0.88]) / 10) * 10,
          '\t- Correct: ', 4.86)
    print('Predicted: ', nn.predict(np.array([5.77, 0.88, 4.86]) / 10) * 10,
          '\t- Correct: ', 1.94)


if __name__ == "__main__":
    main()

from layers import *


class NeuralNetwork:
    def __init__(self, layer_builders: list[LayerBuilder]):
        self.layers = []
        for num_inputs, num_neurons, in zip(layer_sizes, layer_sizes[1:]):
            self.layers.append(Layer(num_inputs, num_neurons, "relu"))
        self.layers[-1].activation_function = "sigmoid"

    def feed_forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.calc_outputs(outputs)
        self.outputs = outputs

    def update_parameters(self):
        for layer in self.layers:
            layer.update_parameters()

    def gradient_descent(self, training_inputs, training_outputs, learning_rate):
        pass

    def train(self, training_inputs, training_outputs, num_iterations, learning_rate):
        for iteration in range(num_iterations):
            self.gradient_descent(training_inputs, training_outputs, learning_rate)
            if (iteration + 1) % (num_iterations / 10) == 0:
                print(f"Iteration {iteration + 1}: ", end="")
                print(f"Cost = {self.average_cost}")

    def test(self, testing_inputs, testing_outputs):
        for inputs, expected_outputs in zip(testing_inputs, testing_outputs):
            self.feed_forward(inputs)

    def save_parameters(self):
        pass


network = Network([2, 3, 1])

inputs = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
outputs = [[0],
           [1],
           [1],
           [1]]

network.train(inputs, outputs, 1000, 0.001)

network.test(inputs, outputs)

network.save_parameters()

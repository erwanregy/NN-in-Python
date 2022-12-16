import math
import random


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    if x > 0:
        return x
    else:
        return 0.01 * x


class Neuron:
    def __init__(self, num_inputs, activation_function):
        self.weights = []
        for _ in range(num_inputs):
            self.weights.append(random.uniform(-1, 1))
        self.bias = 0
        self.activation_function = activation_function
        self.weight_nudges = [[]] * num_inputs
        self.bias_nudges = []

    def calc_output(self, inputs):
        output = 0.0
        for weight, input in zip(self.weights, inputs):
            output += weight * input
        if self.activation_function == "sigmoid":
            output = sigmoid(output + self.bias)
        elif self.activation_function == "relu":
            output = relu(output + self.bias)
        return output

    def update_parameters(self):
        for weight_num, weight_nudges in enumerate(self.weight_nudges):
            weight_nudges_sum = 0.0
            num_weight_nudges = len(weight_nudges)
            for weight_nudge in weight_nudges:
                weight_nudges_sum += weight_nudge
            weight_nudge_average = weight_nudges_sum / num_weight_nudges
            self.weights[weight_num] -= weight_nudge_average
        bias_nudges_sum = 0.0
        num_bias_nudges = len(self.bias_nudges)
        for bias_nudge in self.bias_nudges:
            bias_nudges_sum += bias_nudge
        bias_nudge_average = bias_nudges_sum / num_bias_nudges
        self.bias -= bias_nudge_average


class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(num_inputs, activation_function))

    def calc_outputs(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calc_output(inputs))
        return outputs

    def update_parameters(self):
        for neuron in self.neurons:
            neuron.update_parameters()


class Network:
    def __init__(self, layer_sizes):
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

    def calc_cost(self, expected_outputs):
        sum = 0.0
        for output, expected_output in zip(self.outputs, expected_outputs):
            sum += (output - expected_output) ** 2
        num_outputs = len(self.layers[-1].neurons)
        self.cost = sum / num_outputs

    def gradient_descent(self, training_inputs, training_outputs, learning_rate):
        nudge = 0.00000001
        costs_sum = 0.0
        for inputs, expected_outputs in zip(training_inputs, training_outputs):
            self.feed_forward(inputs)
            self.calc_cost(expected_outputs)
            original_cost = self.cost
            for _, layer in reversed(list(enumerate(self.layers))):
                for neuron_num, neuron in enumerate(layer.neurons):
                    for weight_num, _ in enumerate(neuron.weights):
                        neuron.weights[weight_num] += nudge
                        self.feed_forward(inputs)
                        self.calc_cost(expected_outputs)
                        new_cost = self.cost
                        change_in_cost = new_cost - original_cost
                        .append(
                            change_in_cost / nudge * learning_rate
                        )
                        neuron.weights[weight_num] -= nudge
                                             
                        neuron.weight_nudges[weight_num] = sigmoid_derivative(neuron.weights[weight_num])
                        neuron.weight_nudges[weight_num] = sigmoid_derivative(neuron.weights[weight_num]) * 
                        
                    layer.neurons[neuron_num].bias += nudge
                    self.feed_forward(inputs)
                    self.calc_cost(expected_outputs)
                    new_cost = self.cost
                    change_in_cost = new_cost - original_cost
                    layer.neurons[neuron_num].bias_nudges.append(
                        change_in_cost / nudge * learning_rate
                    )
                    layer.neurons[neuron_num].bias -= nudge
            self.feed_forward(inputs)
            self.calc_cost(expected_outputs)
            costs_sum += self.cost
            self.update_parameters()
        self.average_cost = costs_sum / len(training_inputs)

    def train(self, training_inputs, training_outputs, num_iterations, learning_rate):
        for iteration in range(num_iterations):
            self.gradient_descent(training_inputs, training_outputs, learning_rate)
            if (iteration + 1) % (num_iterations / 10) == 0:
                print(f"Iteration {iteration + 1}: ", end="")
                print(f"Cost = {self.average_cost}")

    def save_parameters(self):
        file = open("parameters.txt", "w")
        for layer_num, layer in enumerate(self.layers):
            file.write(f"layer {layer_num}:\n")
            for neuron_num, neuron in enumerate(layer.neurons):
                file.write(f"    neuron {neuron_num}:\n")
                file.write("        weights:\n")
                for weight_num, weight in enumerate(neuron.weights):
                    file.write(f"           {weight_num}: {weight}\n")
                file.write(f"        bias: {neuron.bias}\n")
        file.close()

    def test(self, testing_inputs, testing_outputs):
        for inputs, expected_outputs in zip(testing_inputs, testing_outputs):
            self.feed_forward(inputs)
            self.calc_cost(expected_outputs)
            print(f"Inputs: {inputs} | Expected Outputs: {expected_outputs} | Outputs: {self.outputs}")

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

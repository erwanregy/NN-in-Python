from random import uniform
from math import exp
from keras import models
from keras.datasets import mnist

class Neuron:
    def sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + exp(-x))
        except OverflowError:
            return 0.0

    def relu(self, x: float) -> float:
        return max(x, 0.0)

    def __init__(self, num_inputs: int, activation: str | None) -> None:
        self.weights: list[float] = [uniform(-0.1, 0.1) for _ in range(num_inputs)]
        self.bias: float = 0.0
        self.activation = getattr(self, activation)
        self.output: float = 0.0

    def calc_output(self, inputs: list[float]) -> None:
        self.output = self.activation(
            sum(weight * input for weight, input in zip(self.weights, inputs))
            + self.bias
        )


class Layer:
    def __init__(self, num_inputs: int, num_neurons: int, activation: str) -> None:
        self.neurons: list[Neuron] = [
            Neuron(num_inputs, activation) for _ in range(num_neurons)
        ]
        self.outputs: list[float] = [0.0 for _ in range(num_neurons)]

    def calc_outputs(self, inputs: list[float]) -> None:
        for n, _ in enumerate(self.neurons):
            self.neurons[n].calc_output(inputs)
            self.outputs[n] = self.neurons[n].output


class ConvolutionalLayer:
    def __init__(self, num_filters: int, filter_size: int, stride: int) -> None:
        self.num_filters: int = num_filters
        self.filter_size: int = filter_size
        self.stride: int = stride
        self.filters: list[list[list[float]]] = [
            [[uniform(-0.1, 0.1) for _ in range(filter_size)] for _ in range(filter_size)]
            for _ in range(num_filters)
        ]
        self.outputs: list[list[list[float]]] = [
            [[0.0 for _ in range(28 // stride)] for _ in range(28 // stride)]
            for _ in range(num_filters)
        ]

    def calc_outputs(self, inputs: list[list[float]]) -> None:
        for f, _ in enumerate(self.filters):
            for i in range(0, len(inputs) - self.filter_size + 1, self.stride):
                for j in range(0, len(inputs[0]) - self.filter_size + 1, self.stride):
                    self.outputs[f][i // self.stride][j // self.stride] = sum(
                        sum(
                            self.filters[f][k][l] * inputs[i + k][j + l]
                            for l in range(self.filter_size)
                        )
                        for k in range(self.filter_size)
                    )


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], activation_functions: list[str]) -> None:
        self.num_inputs: int = layer_sizes[0]
        self.layers: list[Layer] = [
            Layer(num_inputs, num_neurons, activation)
            for num_inputs, num_neurons, activation in zip(
                layer_sizes[0:], layer_sizes[1:], activation_functions
            )
        ]
        self.num_outputs: int = layer_sizes[-1]

    def feed_forward(self, inputs: list[float]) -> None:
        for l, _ in enumerate(self.layers):
            self.layers[l].calc_outputs(inputs)
            inputs = self.layers[l].outputs
        self.outputs = inputs

    def calc_cost(self, inputs, expected_label: int) -> float:
        self.feed_forward(inputs)
        expected_outputs = [
            0.0 if output_num != expected_label else 1.0
            for output_num in range(self.num_outputs)
        ]
        return (
            sum(
                (output - expected_output) ** 2
                for output, expected_output in zip(self.outputs, expected_outputs)
            )
            / self.num_outputs
        )
        
    def train(
        self, train_inputs: list[list[float]], train_labels: list[int]
    ) -> None:
        # train network us stochastic gradient descent
        pass

    def test(
        self, test_inputs: list[list[float]], test_labels: list[int]
    ) -> None:
        cost = sum(
            self.calc_cost(inputs, expected_label)
            for inputs, expected_label in zip(test_inputs, test_labels)
        ) / len(test_inputs)
        print(f"Cost: {cost}")

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            for l, layer in enumerate(self.layers):
                file.write(f"layer {l}:\n")
                for n, neuron in enumerate(layer.neurons):
                    file.write(f"\tneuron {n}:\n")
                    file.write("\t\tweights:\n")
                    for _, weight in enumerate(neuron.weights):
                        file.write(f"\t\t\t{weight}\n")
                    file.write(f"\t\tbias:\n")
                    file.write(f"\t\t\t{neuron.bias}\n")
            file.close()

    def load(self, path: str) -> None:
        with open(path, "r") as file:
            for l, _ in enumerate(self.layers):
                file.readline()
                for n, _ in enumerate(self.layers[l].neurons):
                    file.readline()
                    file.readline()
                    for w, _ in enumerate(self.layers[l].neurons[n].weights):
                        self.layers[l].neurons[n].weights[w] = float(file.readline())
                    file.readline()
                    self.layers[l].neurons[n].bias = float(file.readline())
                    
    def predict(self, inputs: list[float]) -> int:
        self.feed_forward(inputs)
        return self.outputs.index(max(self.outputs))
    
    
def print_image(image: list[list[float]]) -> None:
    print("+" + "-" * (len(image[0]) * 2) + "+")
    for row in image:
        print("|", end="")
        for pixel in row:
            if pixel > 0.8:
                print("██", end="")
            elif pixel > 0.6:
                print("▓▓", end="") 
            elif pixel > 0.4:
                print("▒▒", end="") 
            elif pixel > 0.2:
                print("░░", end="") 
            else:
                print("  ", end="")
        print("|")
    print("+" + "-" * (len(image[0]) * 2) + "+")


def main():
    # Create network
    neural_network = NeuralNetwork([784, 16, 16, 10], ["relu", "relu", "sigmoid"])

    # Load trained model and load parameters into network
    neural_network.load("mnist_parameters.txt")

    # Load dataset
    _, (test_images, test_labels) = mnist.load_data()
    test_images = test_images / 255.0
    test_inputs = [
        [pixel for row in test_image for pixel in row]
        for test_image in test_images
    ]

    # Test network on dataset
    for test_image, test_input, test_label in zip(test_images, test_inputs, test_labels):
        print_image(test_image)
        prediction = neural_network.predict(test_input)
        print(f"Predicted: {prediction}")
        print(f"Expected:  {test_label}")


if __name__ == "__main__":
    main()

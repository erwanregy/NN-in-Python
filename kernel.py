from random import gauss
from pathlib import Path
from activation_functions import activation_functions

matrix = list[list[float]]


class Kernel:
    outputs = None
    deltas = None

    def __init__(
        self,
        dimensions: tuple,
        activation_function: str,
        parameters_directory: Path = None,
    ) -> None:
        self.activation_function, self.activation_derivative = activation_functions[
            activation_function.lower()
        ]
        if parameters_directory:
            self.load_parameters(parameters_directory)
        else:
            self.init_parameters(dimensions)

    def init_parameters(self, dimensions: tuple) -> None:
        self.weights = [
            [gauss(0.0, 0.5) for _ in range(dimensions[1])]
            for _ in range(dimensions[0])
        ]
        self.bias = 0.0

    def load_parameters(self, parameters_directory: Path) -> None:
        weights_file = parameters_directory / "weights.txt"
        with weights_file.open() as file:
            for line in file:
                self.weights.append([float(value) for value in line.split()])
        bias_file = parameters_directory / "bias.txt"
        with bias_file.open() as file:
            self.bias = float(file.read())

    def calc_outputs(self, inputs: matrix) -> matrix:
        self.outputs = [
            [
                self.activation_function(
                    sum(weight * input for weight, input in zip(row, column))
                    + self.bias
                )
                for column in zip(*inputs)
            ]
            for row in self.weights
        ]
        return self.outputs

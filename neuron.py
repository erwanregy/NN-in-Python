from random import gauss
from pathlib import Path
from activation_functions import activation_functions

Vector = list[float]


class Neuron:
    output = None
    delta = None

    def __init__(self, num_inputs: int, activation_function: str, parameters_directory: Path = None) -> None:
        self.activation_function, self.activation_derivative = activation_functions[
            activation_function.lower()
        ]
        if parameters_directory:
            self.load_parameters(parameters_directory)
        else:
            self.init_parameters(num_inputs)

    def init_parameters(self, num_inputs: int) -> None:
        self.weights = [gauss(0.0, 0.5) for _ in range(num_inputs)]
        self.bias = 0.0

    def load_parameters(self, parameters_directory: Path) -> None:
        weights_file = parameters_directory / "weights.txt"
        with weights_file.open() as file:
            self.weights = [float(line) for line in file]
        bias_file = parameters_directory / "bias.txt"
        with bias_file.open() as file:
            self.bias = float(file.read())

    def calc_output(self, inputs: Vector) -> float:
        self.output = self.activation_function(
            sum(weight * input for weight, input in zip(self.weights, inputs)) + self.bias)
        return self.output

    def calc_delta(self, error: float) -> None:
        self.delta = error * self.activation_derivative(self.output)

    def update_parameters(self, inputs: Vector, learning_rate: float) -> None:
        for weight, input in zip(self.weights, inputs):
            weight -= learning_rate * self.delta * input
        self.bias -= learning_rate * self.delta


# def test(inputs, outputs):
#     print()
#     msae = 0.0
#     mspe = 0.0
#     for input, output in zip(inputs, outputs):
#         neuron.calc_output(input)
#         print(f"{neuron.output:.5f}, {output}")
#         msae += (neuron.output - output) ** 2
#         mspe += (neuron.output - output) ** 2 / ((neuron.output + output) / 2)
#     msae /= len(inputs)
#     mspe /= len(inputs)
#     print(f"MSAE: {msae:.5f}")
#     print(f"MSPE: {mspe:.5f}%")
#     print()

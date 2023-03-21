from random import gauss
from pathlib import Path
from typing import Callable
from activation_functions import functions
from datatypes import Vector
from fixed_point import float_to_fixed, fixed_to_float


class Neuron:
    output: float = 0.0
    delta: float = 0.0

    def __init__(self, num_inputs: int, activation_function: str) -> None:
        self.activation_function = functions[activation_function]
        self.weights = [gauss(0, 1.0) for _ in range(num_inputs)]
        self.bias = 0.0

    def calc_output(self, inputs: Vector) -> None:
        self.output = self.activation_function(
            sum(weight * input for weight, input in zip(self.weights, inputs))
            + self.bias
        )

    def calc_delta(self, error: float) -> None:
        self.delta = error * self.activation_function.derivative(self.output)
        
    def calc_error(self, weight_num: int) -> float:
        return self.weights[weight_num] * self.delta
        
    def update_parameters(self, inputs: Vector, learning_rate: float) -> None:
        for w, input in enumerate(inputs):
            self.weights[w] -= learning_rate * self.delta * input
        self.bias -= learning_rate * self.delta

    def save_parameters(
        self, parameters_directory: Path, format: str = "binary"
    ) -> None:
        file_extension, _, to_str = handle_format(format)

        weights_file = parameters_directory / f"weights{file_extension}"
        if overwrite(weights_file):
            with weights_file.open("w") as file:
                for weight in self.weights:
                    file.write(to_str(weight) + "\n")
            print("Weights saved.")
        else:
            print("Weights not saved.")

        bias_file = parameters_directory / f"bias{file_extension}"
        if overwrite(bias_file):
            with bias_file.open("w") as file:
                file.write(to_str(self.bias) + "\n")
            print("Bias saved.")
        else:
            print("Bias not saved.")

    def load_parameters(
        self, parameters_directory: Path, format: str = "binary"
    ) -> None:
        file_extension, to_float, _ = handle_format(format)

        weights_file = parameters_directory / f"weights{file_extension}"
        with weights_file.open() as file:
            for w, line in enumerate(file):
                self.weights[w] = to_float(line)

        bias_file = parameters_directory / f"bias{file_extension}"
        with bias_file.open() as file:
            self.bias = to_float(file.read())


def handle_format(
    format: str,
) -> tuple[str, Callable[[str], float], Callable[[float], str]]:
    if format.lower() == "binary":
        file_extension = ".mem"
        to_float = fixed_to_float
        to_str = float_to_fixed
    elif format.lower() == "decimal":
        file_extension = ".txt"
        to_float = float
        to_str = str
    else:
        raise ValueError(f"Invalid format: {format}")
    return file_extension, to_float, to_str

def overwrite(
    file: Path
) -> bool:
    if not file.exists():
        return True
    elif input(f"File {file} already exists. Overwrite? [y/N] ") == "y":
        return True
    else:
        return False


if __name__ == "__main__":
    neuron = Neuron(16, "sigmoid")

    neuron.load_parameters(Path("models/3b1b/layer_2/neuron_3/"), "binary")

    neuron.calc_output([1.0] * 16)
    
    neuron.save_parameters(Path("test"), "decimal")
    
    neuron2 = Neuron(16, "sigmoid")

    neuron2.load_parameters(Path("test"), "decimal")
    
    neuron2.calc_output([1.0] * 16)
    
    assert(neuron2.output == neuron.output)
    
    print("Test passed.")
    
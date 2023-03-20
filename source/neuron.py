from random import gauss
from pathlib import Path
from activation_functions import functions
from datatypes import Vector
from fixed_point import fixed_to_float, float_to_fixed


class Neuron:
    output = 0.0
    delta = 0.0

    def __init__(self, num_inputs: int, activation: str, parameters_directory: Path = Path()) -> None:
        self.activation_function = functions[
            activation.lower()
        ]
        if parameters_directory != Path():
            self.load_parameters(parameters_directory)
        else:
            self.init_parameters(num_inputs)

    def init_parameters(self, num_inputs: int) -> None:
        self.weights = [gauss(0.0, 0.5) for _ in range(num_inputs)]
        self.bias = 0.0

    def load_parameters(self, parameters_directory: Path, format: str = "binary") -> None:
        if format.lower() == "binary":
            file_extension = ".mem"
            ? = fixed_to_float
        elif format.lower() == "decimal":
            file_extension = ".txt"

        else:
            raise ValueError(f"Invalid format: {format}")
            
        weights_file = parameters_directory / f"weights{file_extension}"
        with weights_file.open() as file:
            self.weights = [?(line) for line in file]

        bias_file = parameters_directory / f"bias{file_extension}"
        with bias_file.open() as file:
            self.bias = ?(file.read())
        
        
        # weights_file = parameters_directory / "weights.mem"
        # with weights_file.open() as file:
        #     if format.lower() == "binary":
        #         self.weights = [fixed_to_float(line) for line in file]
        #     elif format.lower() == "decimal":
        #         self.weights = [float(line) for line in file]
        # bias_file = parameters_directory / "bias.mem"
        # with bias_file.open() as file:
        #     if format.lower() == "binary":
        #         self.bias = fixed_to_float(file.read())
        #     elif format.lower() == "decimal":
        #         self.bias = float(file.read())

    def save_parameters(self, parameters_directory: Path, format: str = "binary") -> None:
        if format.lower() == "binary":
        weights_file = parameters_directory / "weights.mem"
        with weights_file.open("w") as file:
            file.write("\n".join(map(str, map(float_to_fixed, self.weights))))
        bias_file = parameters_directory / "bias.mem"
        with bias_file.open("w") as file:
            file.write(str(float_to_fixed(self.bias)))

    def calc_output(self, inputs: Vector) -> float:
        self.output = self.activation_function(
            sum(weight * input for weight, input in zip(self.weights, inputs)) + self.bias)
        return self.output

    def calc_delta(self, error: float) -> None:
        self.delta = error * self.activation_function.derivative(self.output)

    def update_parameters(self, inputs: Vector, learning_rate: float) -> None:
        for weight, input in zip(self.weights, inputs):
            weight -= learning_rate * self.delta * input
        self.bias -= learning_rate * self.delta


if __name__ == "__main__":
    neuron = Neuron(784, "sigmoid", Path("models/3b1b/layer_0/neuron_0"))
    
    print(neuron.weights)

    neuron.save_parameters(Path("test"), "decimal")

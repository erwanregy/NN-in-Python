from random import gauss
from pathlib import Path
from activation_functions import functions
from datatypes import Matrix
from fixed_point import fixed_to_float


class Kernel:
    outputs = None
    deltas = None

    def __init__(
        self,
        dimensions: tuple[int, int],
        activation: str,
        parameters_directory: Path = Path(),
    ) -> None:
        self.activation_function = functions[
            activation.lower()
        ]
        if parameters_directory != Path():
            self.load_parameters(parameters_directory)
        else:
            self.init_parameters(dimensions)

    def init_parameters(self, dimensions: tuple[int, int]) -> None:
        self.weights = [
            [gauss(0.0, 0.5) for _ in range(dimensions[1])]
            for _ in range(dimensions[0])
        ]
        self.bias = [
            [0.0 for _ in range(dimensions[1])]
            for _ in range(dimensions[0])
        ]

    def load_parameters(self, parameters_directory: Path) -> None:
        weights_file = parameters_directory / "weights.mem"
        with weights_file.open() as file:
            self.weights = [[fixed_to_float(value) for value in line.split()] for line in file]
        bias_file = parameters_directory / "bias.mem"
        with bias_file.open() as file:
            self.bias = [[fixed_to_float(value) for value in line.split()] for line in file]

    def calc_outputs(self, inputs: Matrix) -> Matrix:
        self.outputs = self.activation_function(convolve(inputs, self.weights) + self.bias)
        return self.outputs


def convolve(matrix: Matrix, kernel: Matrix) -> Matrix:
    m, n = len(matrix), len(matrix[0])
    p, q = len(kernel), len(kernel[0])

    result = [[0.0] * (n - q + 1) for _ in range(m - p + 1)]

    for i in range(m - p + 1):
        for j in range(n - q + 1):
            for k in range(p):
                for l in range(q):
                    result[i][j] += matrix[i + k][j + l] * kernel[k][l]

    return result


if __name__ == "__main__":
    from keras.datasets import mnist
    
    _, (images, _) = mnist.load_data()
    
    kernel = Kernel((3, 3), "sigmoid")
    
    kernel.weights = [
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ]
    
    kernel.calc_outputs(images[0])
    
    print(kernel.outputs)
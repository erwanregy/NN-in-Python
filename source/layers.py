from pathlib import Path
from neuron import Neuron
from kernel import Kernel
from datatypes import Vector, Matrix, Tensor


class LayerBuilder:
    type: str
    size: int | tuple[int, int]
    activation_function: str


class LayerBase:
    outputs: Tensor = []
    
    def __init__(self):
        raise NotImplementedError
    
    def calc_outputs(self, inputs: Tensor) -> None:
        raise NotImplementedError
    
    def calc_deltas(self, errors: Tensor) -> None:
        raise NotImplementedError
    
    def calc_errors(self) -> Tensor:
        raise NotImplementedError
    
    def update_parameters(self, inputs: Tensor, learning_rate: float) -> None:
        raise NotImplementedError
    
    def save_parameters(self, layer_directory: Path, format: str = "binary") -> None:
        raise NotImplementedError
    
    def load_parameters(self, layer_directory: Path, format: str = "binary") -> None:
        raise NotImplementedError


class DenseLayer(LayerBase):
    outputs: Vector = []
    
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: str) -> None:
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    def calc_outputs(self, inputs: Vector) -> None:
        for n, _ in enumerate(self.neurons):
            self.neurons[n].calc_output(inputs)
        self.outputs = [neuron.output for neuron in self.neurons]
    
    def calc_deltas(self, errors: Vector) -> None:
        for n, _ in enumerate(self.neurons):
            self.neurons[n].calc_delta(errors[n])
            
    def calc_errors(self) -> Vector:
        return [sum(neuron.calc_error(w) for neuron in self.neurons) for w, _ in enumerate(self.neurons[0].weights)]

    def update_parameters(self, inputs: Vector, learning_rate: float) -> None:
        for n, _ in enumerate(self.neurons):
            self.neurons[n].update_parameters(inputs, learning_rate)
    
    def save_parameters(self, layer_directory: Path, format: str = "binary") -> None:
        pass
    
    def load_parameters(self, layer_directory: Path, format: str = "binary") -> None:
        for n, _ in enumerate(self.neurons):
            neuron_directory = layer_directory / f"neuron_{n}"
            self.neurons[n].load_parameters(neuron_directory, format)
            

class ConvolutionalLayer(LayerBase):
    outputs: Matrix = []
    
    def __init__(self, num_feature_maps: int, input_dimensions: tuple[int, int], kernel_dimensions: tuple[int, int], stride: int, activation: str) -> None:
        self.feature_maps = [Kernel(input_dimensions, kernel_dimensions, stride, activation) for _ in range(num_feature_maps)]
    
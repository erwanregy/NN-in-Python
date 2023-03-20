from neuron import Neuron, Vector
from kernel import Kernel, Matrix


class LayerBuilder:
    type: str
    size: int | tuple[int, int]
    activation_function: str

class LayerBase:
    def __init__(self):
        pass


class DenseLayer(LayerBase):
    outputs = None
    
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: str) -> None:
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    def calc_outputs(self, inputs: Vector) -> Vector:
        self.outputs = [neuron.calc_output(inputs) for neuron in self.neurons]
        return self.outputs

    def update_parameters(self, inputs: Vector, learning_rate: float) -> None:
        for neuron in self.neurons:
            neuron.update_parameters(inputs, learning_rate)
            

class ConvolutionalLayer(LayerBase):
    outputs = None
    
    def __init__(self, num_feature_maps: int, size: tuple[int, int], kernel_size: tuple[int, int], stride: int, activation: str) -> None:
        self.kernels = [Kernel(size, activation) for _ in range(num_feature_maps)]
    
    
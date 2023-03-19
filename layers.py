from neuron import Neuron, Vector

class DenseLayer:
    outputs = None
    
    def __init__(self, num_inputs: int, num_neurons: int, activation_function: str) -> None:
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    def calc_outputs(self, inputs: Vector) -> Vector:
        self.outputs = [neuron.calc_output(inputs) for neuron in self.neurons]
        return self.outputs

    def update_parameters(self, inputs: Vector, learning_rate: float) -> None:
        for neuron in self.neurons:
            neuron.update_parameters(inputs, learning_rate)
            

class ConvolutionalLayer:
    outputs = None
    
    
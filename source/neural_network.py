from random import randint
from pathlib import Path
from layers import *


class NeuralNetwork:
    outputs: Vector = []

    def __init__(self, *layers: LayerBase):
        self.layers = layers

    def predict(self, inputs: Tensor) -> int:
        self.feed_forward(inputs)
        return self.label()

    def label(self) -> int:
        return self.outputs.index(max(self.outputs))

    def feed_forward(self, inputs: Tensor) -> None:
        for l, _ in enumerate(self.layers):
            self.layers[l].calc_outputs(inputs)
            inputs = self.layers[l].outputs
        self.outputs = inputs

    def back_propagate(self, label: int) -> None:
        errors = [output - float(i == label) for i, output in enumerate(self.outputs)]
        for l, _ in reversed(list(enumerate(self.layers))):
            self.layers[l].calc_deltas(errors)
            errors = self.layers[l].calc_errors()

    def update_parameters(self, inputs: Tensor, learning_rate: float) -> None:
        for l, _ in enumerate(self.layers):
            self.layers[l].update_parameters(inputs, learning_rate)
            inputs = self.layers[l].outputs

    def train(
        self,
        inputs: Tensor,
        labels: list[int],
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        test_data: tuple[Tensor, list[int]] | None = None,
    ) -> None:
        if test_data == None:
            test_data = (inputs, labels)
        test_inputs, test_labels = test_data

        for epoch_num in range(1, num_epochs + 1):
            for _ in range(batch_size):
                i = randint(0, len(inputs) - 1)
                self.feed_forward(inputs[i])
                self.back_propagate(labels[i])
                self.update_parameters(inputs[i], learning_rate)
            if (
                num_epochs < 10
                or not epoch_num % (num_epochs / 10)
                or epoch_num == 1
                or epoch_num == num_epochs
            ):
                print(f"Epoch {epoch_num}/{num_epochs}:", end=" ")
                self.test(test_inputs, test_labels)

    def test(self, inputs: Tensor, labels: list[int]) -> None:
        num_correct = 0
        for input, label in zip(inputs, labels):
            if self.predict(input) == label:
                num_correct += 1
        print(
            f"Accuracy: {num_correct / len(inputs) * 100:.2f}% ({num_correct}/{len(inputs)})"
        )

    def save_parameters(self, model_directory: Path, format: str = "binary") -> None:
        pass

    def load_parameters(self, model_directory: Path, format: str = "binary") -> None:
        for l, _ in enumerate(self.layers):
            layer_directory = model_directory / f"layer_{l}"
            self.layers[l].load_parameters(layer_directory, format)


from numpy import ndarray


def normalise(inputs: ndarray) -> ndarray:
    return inputs / 255.0


def flatten(inputs: ndarray) -> ndarray:
    return inputs.reshape(len(inputs), -1)


if __name__ == "__main__":
    neural_network = NeuralNetwork(
        DenseLayer(784, 16, "relu"),
        DenseLayer(16, 16, "relu"),
        DenseLayer(16, 10, "sigmoid"),
    )

    from keras.datasets import mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images, train_inputs = (
        train_images.tolist(),
        flatten(normalise(train_images)).tolist(),
    )
    test_images, test_inputs = (
        test_images.tolist(),
        flatten(normalise(test_images)).tolist(),
    )

    neural_network.load_parameters(Path("models/3b1b"))

    neural_network.test(test_inputs, test_labels)

    from visualisation import print_image

    wait = input

    for image, input, label in zip(test_images, test_inputs, test_labels):
        neural_network.predict(input)
        if neural_network.label() != label:
            print_image(image)
            print(f"Predicted: {neural_network.label()}")
            print(f"Expected:  {label}")
            wait()

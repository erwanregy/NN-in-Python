from math import exp

class ActivationFunction:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x: float) -> float:
        raise NotImplementedError
        
    def derivative(self, x: float) -> float:
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def __call__(self, x: float) -> float:
        if x <= -709.0:
            return 0.0
        else:
            return 1.0 / (1.0 + exp(-x))

    def derivative(self, x: float) -> float:
        return x * (1.0 - x)


class ReLU(ActivationFunction):
    def __call__(self, x: float) -> float:
        return max(0.0, x)

    def derivative(self, x: float) -> float:
        return float(x > 0.0)


functions: dict[
    str, ActivationFunction
] = {
    "sigmoid": Sigmoid(),
    "relu": ReLU()
}

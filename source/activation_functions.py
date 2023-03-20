from math import exp
from typing import Callable
from datatypes import Matrix

def is_matrix(x: object) -> bool:
    return isinstance(x, list) and all(isinstance(row, list) and all(isinstance(value, float) for value in row) for row in x)

def apply_function(function: Callable[[float], float], x: float | Matrix) -> float | Matrix:
    if is_matrix(x):
        return [[function(value) for value in row] for row in x]
    elif isinstance(x, float):
        return function(x)
    else:
        raise TypeError(f"Invalid type for activation function: {type(x)}")

class ActivationFunction:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x: float | Matrix) -> float | Matrix:
        return apply_function(self._function, x)    
    
    def derivative(self, x: float | Matrix) -> float | Matrix:
        return apply_function(self._derivative, x)

    def _function(self, x: float) -> float:
        raise NotImplementedError
        
    def _derivative(self, x: float) -> float:
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def _function(self, x: float) -> float:
        if x <= -710.0:
            return 0.0
        else:
            return 1.0 / (1.0 + exp(-x))

    def _derivative(self, x: float) -> float:
        return x * (1.0 - x)


class ReLU(ActivationFunction):
    def _function(self, x: float) -> float:
        return max(0.0, x)

    def _derivative(self, x: float) -> float:
        return float(x > 0.0)


functions: dict[
    str, ActivationFunction
] = {
    "sigmoid": Sigmoid(),
    "relu": ReLU()
}

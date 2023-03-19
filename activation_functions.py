from math import exp

activation_functions = {
    "sigmoid": (lambda x: 1.0 / (1.0 + exp(-x)), lambda x: x * (1.0 - x)),
    "relu": (lambda x: max(0.0, x), lambda x: float(x > 0.0)),
}
from pennylane.operation import Operation
from pennylane.ops import IsingXX as XX
from pennylane.ops import IsingYY as YY
from pennylane.ops import IsingZZ as ZZ

__all__ = ["GPI", "GPI2", "MS", "XX", "YY", "ZZ"]


class GPI(Operation):
    num_params = 1
    num_wires = 1
    grad_method = None


class GPI2(Operation):
    num_params = 1
    num_wires = 1
    grad_method = None


class MS(Operation):
    num_params = 3
    num_wires = 2
    grad_method = None

    def __init__(self, phi0, phi1, theta=0.25, wires=None):
        super().__init__(phi0, phi1, theta, wires=wires)

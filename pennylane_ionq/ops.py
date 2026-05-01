"""Custom IonQ operations exposed to PennyLane users.

GPI / GPI2 / MS / IonQZZ are IonQ's native trapped-ion gates. Phase parameters
(``phi``, ``phi0``, ``phi1``) are in turns (fractions of 2*pi); interaction
parameters (``theta``, ``angle``) are in units of pi - matching the wire
convention of the IonQ Cloud API and ``ionq_core.gates``.
"""

import numpy as np
from ionq_core import gpi2_matrix, gpi_matrix, ms_matrix, zz_matrix
from pennylane.operation import Operation
from pennylane.ops import IsingXX as XX
from pennylane.ops import IsingYY as YY
from pennylane.ops import IsingZZ as ZZ

__all__ = ["GPI", "GPI2", "MS", "XX", "YY", "ZZ", "IonQZZ"]


class GPI(Operation):
    num_params = 1
    num_wires = 1
    grad_method = None

    @staticmethod
    def compute_matrix(phi):
        return np.asarray(gpi_matrix(float(phi)), dtype=complex)

    def adjoint(self):
        return GPI(self.parameters[0], wires=self.wires)


class GPI2(Operation):
    num_params = 1
    num_wires = 1
    grad_method = None

    @staticmethod
    def compute_matrix(phi):
        return np.asarray(gpi2_matrix(float(phi)), dtype=complex)

    def adjoint(self):
        return GPI2(self.parameters[0] + 0.5, wires=self.wires)


class MS(Operation):
    num_params = 3
    num_wires = 2
    grad_method = None

    def __init__(self, phi0, phi1, theta=0.25, wires=None):
        super().__init__(phi0, phi1, theta, wires=wires)

    @staticmethod
    def compute_matrix(phi0, phi1, theta=0.25):
        return np.asarray(ms_matrix(float(phi0), float(phi1), float(theta)), dtype=complex)

    def adjoint(self):
        phi0, phi1, theta = self.parameters
        return MS(phi0, phi1, -theta, wires=self.wires)


class IonQZZ(Operation):
    """Native IonQ ZZ interaction; ``angle`` is in units of pi."""

    num_params = 1
    num_wires = 2
    grad_method = None

    @staticmethod
    def compute_matrix(angle):
        return np.asarray(zz_matrix(float(angle)), dtype=complex)

    def adjoint(self):
        return IonQZZ(-self.parameters[0], wires=self.wires)

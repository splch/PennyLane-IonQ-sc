"""Native IonQ trapped-ion gates exposed to PennyLane.

Phase parameters (``phi``, ``phi0``, ``phi1``) are in **turns** (fractions of
2*pi); interaction parameters (``theta``, ``angle``) are in **units of pi** -
matching ``ionq_core.gates`` and the IonQ Cloud API wire format.
"""

from __future__ import annotations

import numpy as np
from ionq_core import gpi2_matrix, gpi_matrix, ms_matrix, zz_matrix
from pennylane.operation import Operation

__all__ = ["GPI", "GPI2", "MS", "IonQZZ"]


class GPI(Operation):
    """Single-qubit GPI gate (involutory; X at ``phi=0``)."""

    num_params = 1
    num_wires = 1
    grad_method = None

    @staticmethod
    def compute_matrix(phi):
        return np.asarray(gpi_matrix(float(phi)), dtype=complex)

    def adjoint(self):
        return GPI(self.parameters[0], wires=self.wires)


class GPI2(Operation):
    """Single-qubit GPI2 gate (pi/2 rotation about an XY-plane axis)."""

    num_params = 1
    num_wires = 1
    grad_method = None

    @staticmethod
    def compute_matrix(phi):
        return np.asarray(gpi2_matrix(float(phi)), dtype=complex)

    def adjoint(self):
        return GPI2(self.parameters[0] + 0.5, wires=self.wires)


class MS(Operation):
    """Two-qubit Molmer-Sorensen gate; ``angle=0.25`` is maximally entangling."""

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

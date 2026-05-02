# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the native IonQ gate operations."""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml
import pytest
from ionq_core import gpi2_matrix, gpi_matrix, ms_matrix, zz_matrix

from pennylane_ionq import GPI, GPI2, MS, IonQZZ


class TestGPI:
    def test_metadata(self):
        assert GPI.num_params == 1
        assert GPI.num_wires == 1
        assert GPI.grad_method is None

    def test_matrix_matches_ionq_core(self):
        np.testing.assert_allclose(
            GPI.compute_matrix(0.123),
            np.asarray(gpi_matrix(0.123), dtype=complex),
        )

    def test_x_at_zero(self):
        # GPI(0) is the Pauli-X gate.
        np.testing.assert_allclose(
            GPI.compute_matrix(0.0),
            np.array([[0, 1], [1, 0]], dtype=complex),
            atol=1e-12,
        )

    def test_self_inverse(self):
        m = GPI.compute_matrix(0.37)
        np.testing.assert_allclose(m @ m, np.eye(2), atol=1e-12)

    def test_adjoint_returns_self_form(self):
        op = GPI(0.42, wires=0)
        adj = op.adjoint()
        assert isinstance(adj, GPI)
        assert adj.parameters == [0.42]
        assert adj.wires == op.wires


class TestGPI2:
    def test_metadata(self):
        assert GPI2.num_params == 1
        assert GPI2.num_wires == 1

    def test_matrix_matches_ionq_core(self):
        np.testing.assert_allclose(
            GPI2.compute_matrix(0.7),
            np.asarray(gpi2_matrix(0.7), dtype=complex),
        )

    def test_adjoint_shifts_phase_by_half(self):
        op = GPI2(0.1, wires=0)
        adj = op.adjoint()
        assert isinstance(adj, GPI2)
        assert adj.parameters == [0.6]


class TestMS:
    def test_metadata(self):
        assert MS.num_params == 3
        assert MS.num_wires == 2

    def test_default_theta_is_max_entangling(self):
        op = MS(0.0, 0.0, wires=[0, 1])
        # Third parameter defaults to 0.25.
        assert op.parameters[2] == 0.25

    def test_matrix_matches_ionq_core(self):
        np.testing.assert_allclose(
            MS.compute_matrix(0.1, 0.2, 0.25),
            np.asarray(ms_matrix(0.1, 0.2, 0.25), dtype=complex),
        )

    def test_adjoint_flips_theta(self):
        op = MS(0.1, 0.2, 0.25, wires=[0, 1])
        adj = op.adjoint()
        assert isinstance(adj, MS)
        assert adj.parameters == [0.1, 0.2, -0.25]


class TestIonQZZ:
    def test_metadata(self):
        assert IonQZZ.num_params == 1
        assert IonQZZ.num_wires == 2

    def test_matrix_matches_ionq_core(self):
        np.testing.assert_allclose(
            IonQZZ.compute_matrix(0.3),
            np.asarray(zz_matrix(0.3), dtype=complex),
        )

    def test_zero_angle_is_identity(self):
        np.testing.assert_allclose(IonQZZ.compute_matrix(0.0), np.eye(4), atol=1e-12)

    def test_adjoint_flips_angle(self):
        op = IonQZZ(0.5, wires=[0, 1])
        adj = op.adjoint()
        assert isinstance(adj, IonQZZ)
        assert adj.parameters == [-0.5]


def test_native_gates_are_pennylane_operations():
    for op_cls, args in [
        (GPI, (0.0,)),
        (GPI2, (0.5,)),
        (MS, (0.1, 0.2)),
        (IonQZZ, (0.25,)),
    ]:
        wires = list(range(op_cls.num_wires))
        op = op_cls(*args, wires=wires)
        assert isinstance(op, qml.operation.Operation)


@pytest.mark.parametrize(
    "gate_cls,params",
    [
        (GPI, (0.0,)),
        (GPI2, (0.25,)),
        (IonQZZ, (0.1,)),
    ],
)
def test_matrices_are_unitary(gate_cls, params):
    m = gate_cls.compute_matrix(*params)
    np.testing.assert_allclose(m.conj().T @ m, np.eye(m.shape[0]), atol=1e-10)


def test_ms_matrix_is_unitary():
    m = MS.compute_matrix(0.1, 0.2, 0.25)
    np.testing.assert_allclose(m.conj().T @ m, np.eye(4), atol=1e-10)


def test_gpi_squared_phi_is_diagonal():
    # GPI(phi)^2 == identity for any phi (involutory).
    for phi in [0.0, 0.13, math.pi / 7]:
        m = GPI.compute_matrix(phi)
        np.testing.assert_allclose(m @ m, np.eye(2), atol=1e-12)

# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the native-gate decomposition rules."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from pennylane_ionq import GPI, GPI2, IonQZZ
from pennylane_ionq.decompositions import _decompose_native

NATIVE_GATESET = {"GPI", "GPI2", "MS", "IonQZZ"}


def _decompose(ops):
    """Run ``_decompose_native`` over a single-tape queue of ops."""
    with qml.queuing.AnnotatedQueue() as q:
        for op in ops:
            qml.apply(op)
    tape = qml.tape.QuantumScript.from_queue(q)
    (decomposed,), _ = _decompose_native(tape, gate_set=NATIVE_GATESET)
    return decomposed


@pytest.mark.parametrize(
    "op",
    [
        qml.Hadamard(wires=0),
        qml.PauliX(wires=0),
        qml.PauliY(wires=0),
        qml.PauliZ(wires=0),
        qml.S(wires=0),
        qml.T(wires=0),
        qml.SX(wires=0),
        qml.adjoint(qml.S(wires=0)),
        qml.adjoint(qml.T(wires=0)),
        qml.adjoint(qml.SX(wires=0)),
        qml.RX(0.3, wires=0),
        qml.RY(0.4, wires=0),
        qml.RZ(0.5, wires=0),
    ],
)
def test_single_qubit_decomp_uses_only_native_gates(op):
    tape = _decompose([op])
    for gate in tape.operations:
        assert gate.name in NATIVE_GATESET, (
            f"{op.name} decomposed to {gate.name}, which isn't a native IonQ gate"
        )


def test_isingzz_decomposes_to_ionqzz():
    tape = _decompose([qml.IsingZZ(0.6, wires=[0, 1])])
    assert any(isinstance(g, IonQZZ) for g in tape.operations)
    for g in tape.operations:
        assert g.name in NATIVE_GATESET


def test_decomp_preserves_unitary():
    # Decomposing PauliX should produce a circuit equivalent to PauliX.
    tape = _decompose([qml.PauliX(wires=0)])
    matrix = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=[0])
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    # Allow a global phase difference (decompositions are equivalent up to a phase).
    overlap = np.abs(np.trace(matrix.conj().T @ pauli_x)) / 2
    assert overlap == pytest.approx(1.0, abs=1e-8)


def test_rz_decomp_preserves_unitary_up_to_global_phase():
    theta = 0.7
    tape = _decompose([qml.RZ(theta, wires=0)])
    matrix = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=[0])
    expected = qml.matrix(qml.RZ(theta, wires=0))
    overlap = np.abs(np.trace(matrix.conj().T @ expected)) / 2
    assert overlap == pytest.approx(1.0, abs=1e-8)


def test_native_gate_passes_through_unchanged():
    op = GPI(0.25, wires=0)
    tape = _decompose([op])
    assert len(tape.operations) == 1
    assert isinstance(tape.operations[0], GPI)


def test_global_phase_is_dropped():
    # GlobalPhase is mapped to ``null_decomp`` (it's irrelevant for samples).
    tape = _decompose([qml.GlobalPhase(0.3), GPI2(0.0, wires=0)])
    # Only the native gate should remain.
    names = [g.name for g in tape.operations]
    assert "GlobalPhase" not in names
    assert "GPI2" in names

# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the IonQ serialization layer."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from ionq_core.models import (
    GateCnot,
    GateH,
    GateNativeGate,
    GatePauliexp,
    GateRx,
    GateSi,
    GateSwap,
    GateX,
    GateXX,
    GateZZ,
)

from pennylane_ionq import GPI, GPI2, MS, IonQZZ
from pennylane_ionq.serialize import (
    _native_op,
    _pauliexp_op,
    _qis_op,
    _samples_from_probs,
    _to_standard_wires,
)


class TestQisSerialization:
    def test_pauli_x(self):
        gate = _qis_op(qml.PauliX(wires=2))
        assert isinstance(gate, GateX)
        assert gate.gate == "x"
        assert gate.targets == [2]

    def test_hadamard(self):
        gate = _qis_op(qml.Hadamard(wires=0))
        assert isinstance(gate, GateH)
        assert gate.gate == "h"
        assert gate.targets == [0]

    def test_rotation_forwards_data(self):
        gate = _qis_op(qml.RX(0.42, wires=1))
        assert isinstance(gate, GateRx)
        assert gate.rotation == pytest.approx(0.42)
        assert gate.targets == [1]

    def test_cnot_splits_target_and_control(self):
        # CNOT in PennyLane: wires=[control, target]; on the wire it goes
        # control -> "controls", target -> "targets".
        gate = _qis_op(qml.CNOT(wires=[0, 1]))
        assert isinstance(gate, GateCnot)
        assert gate.controls == [0]
        assert gate.targets == [1]

    def test_swap(self):
        gate = _qis_op(qml.SWAP(wires=[1, 0]))
        assert isinstance(gate, GateSwap)
        assert gate.targets == [1, 0]

    def test_ising_xx(self):
        gate = _qis_op(qml.IsingXX(0.7, wires=[0, 1]))
        assert isinstance(gate, GateXX)
        assert gate.rotation == pytest.approx(0.7)

    def test_ising_zz(self):
        gate = _qis_op(qml.IsingZZ(0.3, wires=[0, 1]))
        assert isinstance(gate, GateZZ)
        assert gate.rotation == pytest.approx(0.3)

    def test_adjoint_s(self):
        gate = _qis_op(qml.adjoint(qml.S(wires=0)))
        assert isinstance(gate, GateSi)
        assert gate.gate == "si"


class TestNativeSerialization:
    def test_gpi(self):
        gate = _native_op(GPI(0.123, wires=0))
        assert isinstance(gate, GateNativeGate)
        assert gate.gate == "gpi"
        assert gate.target == 0
        assert gate.phase == pytest.approx(0.123)

    def test_gpi2(self):
        gate = _native_op(GPI2(0.5, wires=2))
        assert gate.gate == "gpi2"
        assert gate.target == 2
        assert gate.phase == pytest.approx(0.5)

    def test_ms(self):
        gate = _native_op(MS(0.1, 0.2, 0.25, wires=[0, 1]))
        assert gate.gate == "ms"
        assert gate.targets == [0, 1]
        assert list(gate.phases) == [pytest.approx(0.1), pytest.approx(0.2)]
        assert gate.angle == pytest.approx(0.25)

    def test_ionqzz(self):
        gate = _native_op(IonQZZ(0.4, wires=[0, 1]))
        assert gate.gate == "zz"
        assert gate.targets == [0, 1]
        assert gate.angle == pytest.approx(0.4)


class TestPauliexp:
    def test_simple_pauli_word(self):
        op = qml.evolve(qml.PauliX(0) @ qml.PauliY(1), 0.3)
        gate = _pauliexp_op(op)
        assert isinstance(gate, GatePauliexp)
        assert gate.time == pytest.approx(0.3)
        # The Pauli string is big-endian over targets.
        assert "XY" in gate.terms
        assert all(c == 1.0 for c in gate.coefficients)

    def test_negative_time_is_folded_into_coefficient_sign(self):
        op = qml.evolve(qml.PauliZ(0), -0.5)
        gate = _pauliexp_op(op)
        assert gate.time == pytest.approx(0.5)
        assert gate.coefficients[0] == pytest.approx(-1.0)

    def test_zero_time_returns_none(self):
        op = qml.evolve(qml.PauliZ(0), 0.0)
        assert _pauliexp_op(op) is None

    def test_identity_only_word_dropped(self):
        # An Identity-only Pauli word contributes only a global phase.
        op = qml.evolve(qml.Identity(0), 0.5)
        assert _pauliexp_op(op) is None

    def test_complex_coefficient_raises(self):
        op = qml.evolve(qml.s_prod(1j, qml.PauliX(0)), 0.2)
        with pytest.raises(ValueError, match="real Pauli coefficients"):
            _pauliexp_op(op)

    def test_no_pauli_rep_raises(self):
        # A Hermitian observable has no Pauli decomposition by name.
        H = qml.Hermitian(np.array([[1.0, 0.0], [0.0, -1.0]]), wires=0)
        op = qml.evolve(H, 0.1)
        with pytest.raises(ValueError, match="pauliexp requires a generator"):
            _pauliexp_op(op)


class TestStandardWires:
    def test_remaps_to_zero_indexed_wires(self):
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(wires="a")
            qml.PauliZ(wires="b")
        tape = qml.tape.QuantumScript.from_queue(q)
        (mapped,), _ = _to_standard_wires(tape)
        assert sorted(mapped.wires.tolist()) == [0, 1]


class TestSamplesFromProbs:
    def test_endianness_flip(self):
        # IonQ keys probabilities little-endian over qubits (qubit 0 is LSB).
        # PennyLane sample_probs expects big-endian (qubit 0 is MSB).
        # Probability mass on integer key 1 (qubit 0 = 1, qubit 1 = 0) should
        # produce sample state |10> in PennyLane's MSB convention.
        shots = qml.measurements.Shots(100)
        samples = _samples_from_probs({1: 1.0}, n=2, shots=shots)
        # Every sample should be the same basis state because all mass is on one outcome.
        assert samples.shape == (100, 2)
        unique = np.unique(samples, axis=0)
        assert len(unique) == 1
        # Big-endian bits of integer key 1 in 2 qubits, little-endian to big-endian
        # flip yields [1, 0].
        np.testing.assert_array_equal(unique[0], np.array([1, 0]))

    def test_partitioned_shots(self):
        shots = qml.measurements.Shots([10, 20])
        out = _samples_from_probs({0: 1.0}, n=1, shots=shots)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == (10, 1)
        assert out[1].shape == (20, 1)

    def test_normalizes_unsummed_probs(self):
        # Real cloud responses can have small numerical drift; the helper
        # renormalizes if the mass doesn't sum to 1.
        shots = qml.measurements.Shots(50)
        samples = _samples_from_probs({0: 0.4, 1: 0.4}, n=1, shots=shots)
        assert samples.shape == (50, 1)

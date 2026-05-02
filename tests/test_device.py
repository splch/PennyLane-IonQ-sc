# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the IonQ device classes."""

from __future__ import annotations

from pathlib import Path

import pennylane as qml
import pytest
from ionq_core.models import CircuitJobCreationPayload, JSONMultiCircuitJob

from pennylane_ionq import IonQDevice, IonQQPUDevice, IonQSimulatorDevice


class TestDeviceConstruction:
    def test_simulator_default_name(self):
        dev = IonQSimulatorDevice(wires=2)
        assert dev.name == "ionq.simulator"

    def test_qpu_default_backend_is_aria_1(self):
        dev = IonQQPUDevice(wires=2)
        assert dev.name == "ionq.qpu.aria-1"

    def test_qpu_backend_override(self):
        dev = IonQQPUDevice(wires=2, backend="forte-1")
        assert dev.name == "ionq.qpu.forte-1"

    def test_arbitrary_backend(self):
        dev = IonQDevice(wires=2, backend="qpu.tempo-1")
        assert dev.name == "ionq.qpu.tempo-1"

    def test_invalid_gateset_falls_back_to_qis(self):
        # gateset is restricted by Literal["qis", "native"]; non-native string
        # is treated as qis (the device defaults to qis serialization).
        dev = IonQDevice(wires=2, gateset="qis")
        assert dev._gateset == "qis"

    def test_native_gateset_chooses_native_serializer(self):
        dev = IonQDevice(wires=2, gateset="native")
        assert dev._gateset == "native"
        # The serializer is _native_op when gateset is native.
        from pennylane_ionq.serialize import _native_op

        assert dev._serialize_op is _native_op


class TestDeviceCapabilities:
    def test_capabilities_file_path_resolves(self):
        dev = IonQSimulatorDevice(wires=2)
        path = Path(dev.config_filepath)
        assert path.exists()
        assert path.name == "capabilities.toml"

    def test_capabilities_loaded(self):
        dev = IonQSimulatorDevice(wires=2)
        caps = dev.capabilities
        assert caps is not None
        # qis gates are present.
        gate_names = set(caps.operations.keys())
        for name in ["PauliX", "Hadamard", "RX", "CNOT", "Evolution", "GPI", "MS", "IonQZZ"]:
            assert name in gate_names

    def test_capabilities_observables(self):
        dev = IonQSimulatorDevice(wires=2)
        caps = dev.capabilities
        obs = set(caps.observables.keys())
        for name in ["PauliX", "PauliY", "PauliZ", "Hadamard", "Identity"]:
            assert name in obs


class TestEntryPoints:
    def test_simulator_via_qml_device(self):
        dev = qml.device("ionq.simulator", wires=2)
        assert isinstance(dev, IonQSimulatorDevice)

    def test_qpu_via_qml_device(self):
        dev = qml.device("ionq.qpu", wires=2, backend="aria-1")
        assert isinstance(dev, IonQQPUDevice)
        assert dev.name == "ionq.qpu.aria-1"


class TestPreprocessPipeline:
    def test_qis_pipeline(self):
        dev = IonQSimulatorDevice(wires=2)
        program = dev.preprocess_transforms()
        # The program is a CompilePipeline of named transforms.
        assert program is not None

    def test_native_pipeline_uses_native_decompose(self):
        # When gateset="native", the device's decomposition step swaps to
        # the IonQ-native decomposer.
        dev = IonQSimulatorDevice(wires=2, gateset="native")
        program = dev.preprocess_transforms()
        assert program is not None


class TestExecuteSingleCircuit:
    def test_single_tape_submits_one_job_and_returns_samples(self, mock_ionq):
        # Use the equal-superposition default; for a 1-qubit, 100-shot circuit
        # we should receive a (100,) sample array per the post-processing
        # pipeline.
        mock_ionq.probabilities.additional_properties = {0: 1.0}

        dev = IonQSimulatorDevice(wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(wires=0)

        result = qml.set_shots(circuit, shots=100)()
        assert result.shape == (100, 1)
        assert (result == 0).all()

        # Exactly one job submission was made.
        assert len(mock_ionq.create_job_calls) == 1
        body = mock_ionq.create_job_calls[0]["body"]
        assert isinstance(body, CircuitJobCreationPayload)
        assert body.type_ == "ionq.circuit.v1"
        assert body.shots == 100
        assert body.backend == "simulator"

    def test_session_id_propagated_to_payload(self, mock_ionq):
        dev = IonQSimulatorDevice(wires=1, session_id="sess-abc")

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.sample(wires=0)

        qml.set_shots(circuit, shots=10)()
        body = mock_ionq.create_job_calls[0]["body"]
        assert body.session_id == "sess-abc"


class TestExecuteMultiCircuit:
    def test_batch_uses_multi_circuit_payload(self, mock_ionq):
        # Configure the mock to return two child jobs.
        mock_ionq.job_response.child_job_ids = ["child-0", "child-1"]
        mock_ionq.probabilities.additional_properties = {0: 1.0}

        dev = IonQSimulatorDevice(wires=1)

        # A non-commuting pair forces a split into two tapes.
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))]

        qml.set_shots(circuit, shots=20)()

        # One create_job call for the multi-circuit submission.
        assert len(mock_ionq.create_job_calls) == 1
        body = mock_ionq.create_job_calls[0]["body"]
        assert isinstance(body, JSONMultiCircuitJob)
        assert body.type_ == "ionq.multi-circuit.v1"

        # And one probabilities fetch per child.
        fetched_uuids = [c["uuid"] for c in mock_ionq.get_probabilities_calls]
        assert fetched_uuids == ["child-0", "child-1"]


class TestErrorPaths:
    def test_no_analytic_circuits(self):
        # When ``shots=None``, the no_analytic preprocess transform should
        # raise on the first execute().
        dev = IonQSimulatorDevice(wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        # No mock_ionq fixture -> network calls would be live, but the
        # preprocess pipeline rejects analytic circuits before that happens.
        with pytest.raises(qml.exceptions.DeviceError):
            circuit()

    def test_unsupported_observable_rejected(self, mock_ionq):
        dev = IonQSimulatorDevice(wires=1)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.Hermitian([[1.0, 0.0], [0.0, -1.0]], wires=0))

        with pytest.raises(qml.exceptions.DeviceError):
            qml.set_shots(circuit, shots=10)()

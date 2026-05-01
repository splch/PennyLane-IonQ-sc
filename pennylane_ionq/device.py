from pathlib import Path

import numpy as np
from ionq_core import ClientExtension, IonQClient, wait_for_job
from ionq_core.api.default import create_job, get_job_probabilities
from ionq_core.models.circuit_job_creation_payload import CircuitJobCreationPayload
from pennylane import CompilePipeline
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.exceptions import DeviceError
from pennylane.transforms import broadcast_expand, diagonalize_measurements, split_non_commuting

_OBS = {"PauliZ", "Identity", "Prod", "SProd", "Sum", "LinearCombination"}

GATES = {
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "S": "s",
    "T": "t",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CNOT": "cnot",
}


def _ok(v, msg):
    if v is None:
        raise DeviceError(msg)
    return v


def _gate(op):
    g, w = GATES[op.name], list(op.wires)
    d: dict = {"gate": g}
    if g == "cnot":
        d["control"], d["target"] = w
    else:
        d["target"] = w[0]
    if op.data:
        d["rotation"] = float(op.data[0])
    return d


@simulator_tracking
@single_tape_support
class IonQDevice(Device):
    config_filepath = str(Path(__file__).parent / "capabilities.toml")

    def __init__(
        self, wires=None, *, backend: str = "simulator", api_key: str | None = None
    ) -> None:
        super().__init__(wires=wires)
        self._backend = backend
        self._client = IonQClient(
            api_key=api_key, extension=ClientExtension(user_agent_token="pennylane-ionq")
        )
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        return f"ionq.{self._backend}"

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None):
        p = CompilePipeline()
        p.add_transform(validate_device_wires, self.wires, name=self.name)
        p.add_transform(decompose, stopping_condition=lambda op: op.name in GATES, name=self.name)
        p.add_transform(split_non_commuting)
        p.add_transform(diagonalize_measurements)
        p.add_transform(broadcast_expand)
        p.add_transform(validate_measurements, name=self.name)
        p.add_transform(
            validate_observables, stopping_condition=lambda o: o.name in _OBS, name=self.name
        )
        return p

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        return tuple(self._run(t) for t in circuits)

    def _run(self, tape):
        if not tape.shots or tape.shots.has_partitioned_shots:
            raise DeviceError(f"{self.name} requires finite, non-partitioned shots")
        n = len(self.wires) if self.wires else tape.num_wires
        body = CircuitJobCreationPayload.from_dict(
            {
                "type": "ionq.circuit.v1",
                "backend": self._backend,
                "shots": tape.shots.total_shots,
                "input": {
                    "gateset": "qis",
                    "qubits": n,
                    "circuit": [_gate(op) for op in tape.operations],
                },
            }
        )
        job = _ok(create_job.sync(client=self._client, body=body), "submission failed")
        wait_for_job(self._client, job.id)
        result = _ok(get_job_probabilities.sync(uuid=job.id, client=self._client), "no results")

        probs = np.zeros(2**n)
        for k, p in result.additional_properties.items():
            probs[int(format(int(k), f"0{n}b")[::-1], 2)] = p
        idx = self._rng.choice(2**n, size=tape.shots.total_shots, p=probs)
        bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int64)
        wire_order = self.wires or tape.wires
        out = tuple(mp.process_samples(bits, wire_order) for mp in tape.measurements)
        return out[0] if len(tape.measurements) == 1 else out

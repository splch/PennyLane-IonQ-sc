from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from ionq_core import UNSET, IonQClient, wait_for_job
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
from pennylane.devices.qubit.sampling import sample_probs
from pennylane.exceptions import DeviceError
from pennylane.transforms import broadcast_expand, diagonalize_measurements, split_non_commuting

_OBS = {"PauliZ", "Identity", "Prod", "SProd", "Sum", "LinearCombination"}

_QIS = {
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "S": "s",
    "T": "t",
    "SX": "v",
    "SWAP": "swap",
    "Adjoint(S)": "si",
    "Adjoint(T)": "ti",
    "Adjoint(SX)": "vi",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CNOT": "cnot",
    "IsingXX": "xx",
    "IsingYY": "yy",
    "IsingZZ": "zz",
}

_NATIVE = {"GPI": "gpi", "GPI2": "gpi2", "MS": "ms"}


def _ok(v, msg):
    if v is None:
        raise DeviceError(msg)
    return v


def _qis_gate(op):
    g, w = _QIS[op.name], list(op.wires)
    d: dict = {"gate": g}
    if g == "cnot":
        d["control"], d["target"] = w
    elif len(w) == 2:
        d["targets"] = w
    else:
        d["target"] = w[0]
    if op.data:
        d["rotation"] = float(op.data[0])
    return d


def _native_gate(op):
    g, w = _NATIVE[op.name], list(op.wires)
    d: dict = {"gate": g, "targets": w} if len(w) == 2 else {"gate": g, "target": w[0]}
    if op.name == "MS":
        d["phases"] = [float(x) for x in op.data[:2]]
        if len(op.data) > 2:
            d["angle"] = float(op.data[2])
    elif op.data:
        d["phase"] = float(op.data[0])
    return d


@simulator_tracking
@single_tape_support
class IonQDevice(Device):
    config_filepath = str(Path(__file__).parent / "capabilities.toml")

    def __init__(
        self,
        wires=None,
        *,
        backend: str = "simulator",
        gateset: Literal["qis", "native"] = "qis",
        api_key: str | None = None,
        job_name: str | None = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        sharpen: bool | None = None,
        dry_run: bool = False,
        noise: dict | None = None,
        metadata: dict | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__(wires=wires)
        self._backend = backend
        self._gateset = gateset
        self._gate = _native_gate if gateset == "native" else _qis_gate
        self._allowed = _NATIVE if gateset == "native" else _QIS
        self._job_name = job_name
        self._compilation = compilation
        self._error_mitigation = error_mitigation
        self._sharpen = UNSET if sharpen is None else sharpen
        self._dry_run = dry_run
        self._noise = noise
        self._metadata = metadata
        self._client = IonQClient(
            api_key=api_key,
            timeout=httpx.Timeout(timeout) if timeout is not None else None,
            max_retries=max_retries,
            additional_user_agent="pennylane-ionq",
        )

    @property
    def name(self) -> str:
        return f"ionq.{self._backend}"

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None):
        p = CompilePipeline()
        p.add_transform(validate_device_wires, self.wires, name=self.name)
        p.add_transform(
            decompose, stopping_condition=lambda op: op.name in self._allowed, name=self.name
        )
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

    def _build_payload(self, tape, n):
        body: dict = {
            "type": "ionq.circuit.v1",
            "backend": self._backend,
            "shots": tape.shots.total_shots,
            "input": {
                "gateset": self._gateset,
                "qubits": n,
                "circuit": [self._gate(op) for op in tape.operations],
            },
        }
        settings = {
            k: v
            for k, v in (
                ("compilation", self._compilation),
                ("error_mitigation", self._error_mitigation),
            )
            if v
        }
        for k, v in (
            ("settings", settings),
            ("dry_run", self._dry_run or None),
            ("noise", self._noise),
            ("metadata", self._metadata),
            ("name", self._job_name),
        ):
            if v:
                body[k] = v
        return body

    def _run(self, tape):
        if not tape.shots or tape.shots.has_partitioned_shots:
            raise DeviceError(f"{self.name} requires finite, non-partitioned shots")
        tape = tape.map_to_standard_wires()
        n = tape.num_wires
        body = CircuitJobCreationPayload.from_dict(self._build_payload(tape, n))
        job = _ok(create_job.sync(client=self._client, body=body), "submission failed")
        if self._dry_run:
            return []
        wait_for_job(self._client, job.id)
        result = _ok(
            get_job_probabilities.sync(uuid=job.id, client=self._client, sharpen=self._sharpen),
            "no results",
        )

        probs = np.zeros(2**n)
        total = sum(result.additional_properties.values()) or 1.0
        for k, p in result.additional_properties.items():
            probs[int(format(int(k), f"0{n}b")[::-1], 2)] = p / total
        bits = sample_probs(probs, tape.shots.total_shots, n, False, None)
        out = tuple(mp.process_samples(bits, tape.wires) for mp in tape.measurements)
        return out[0] if len(tape.measurements) == 1 else out


class IonQSimulatorDevice(IonQDevice):
    def __init__(self, wires=None, **kw):
        super().__init__(wires=wires, backend="simulator", **kw)


class IonQQPUDevice(IonQDevice):
    def __init__(self, wires=None, *, backend: str = "aria-1", **kw):
        super().__init__(wires=wires, backend=f"qpu.{backend}", **kw)

"""PennyLane Device implementation backed by ``ionq_core``."""

import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import pennylane as qml
from ionq_core import (
    UNSET,
    APIError,
    IonQClient,
    JobFailedError,
    JobTimeoutError,
    Unset,
    wait_for_job,
)
from ionq_core.api.default import (
    create_job,
    get_job_cost,
    get_job_probabilities,
    get_variant_probabilities,
)
from ionq_core.models import (
    CircuitJobCreationPayload,
    CircuitJobCreationPayloadSettings,
    CircuitJobCreationPayloadSettingsCompilation,
    CircuitJobCreationPayloadSettingsErrorMitigation,
    GateNativeGate,
    GateQisGate,
    JobMetadata,
    JsonMultiCircuitInput,
    JSONMultiCircuitJob,
    JSONMultiCircuitJobSettings,
    NativeCircuit,
    NativeCircuitInput,
    Noise,
    QISCircuit,
    QisCircuitInput,
)
from pennylane import transform
from pennylane.decomposition import (
    add_decomps,
    disable_graph,
    enable_graph,
    enabled_graph,
    null_decomp,
    register_resources,
)
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    measurements_from_samples,
    no_analytic,
    null_postprocessing,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.devices.qubit.sampling import sample_probs
from pennylane.exceptions import DeviceError
from pennylane.transforms import broadcast_expand, split_non_commuting
from pennylane.transforms import decompose as graph_decompose
from pennylane.transforms.core import CompilePipeline

from .ops import GPI, GPI2, IonQZZ

try:
    _VERSION = version("pennylane-ionq")
except PackageNotFoundError:
    _VERSION = "0.0.0"

_OBSERVABLES = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "Identity",
    "Prod",
    "SProd",
    "Sum",
    "LinearCombination",
}

_QIS_GATES = {
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

_NATIVE_GATES = {"GPI": "gpi", "GPI2": "gpi2", "MS": "ms", "IonQZZ": "zz"}


def _qis_op(op) -> GateQisGate:
    name = _QIS_GATES[op.name]
    wires = list(op.wires)
    rotation = float(op.data[0]) if op.data else UNSET
    if name == "cnot":
        return GateQisGate(gate=name, control=wires[0], target=wires[1])
    if len(wires) == 2:
        return GateQisGate(gate=name, targets=wires, rotation=rotation)
    return GateQisGate(gate=name, target=wires[0], rotation=rotation)


def _native_op(op) -> GateNativeGate:
    name = _NATIVE_GATES[op.name]
    wires = list(op.wires)
    if op.name == "MS":
        return GateNativeGate(
            gate=name,
            targets=wires,
            phases=[float(op.data[0]), float(op.data[1])],
            angle=float(op.data[2]),
        )
    if op.name == "IonQZZ":
        return GateNativeGate(gate=name, targets=wires, angle=float(op.data[0]))
    return GateNativeGate(gate=name, target=wires[0], phase=float(op.data[0]))


@transform
def _to_standard_wires(tape):
    """Map heterogeneous wire labels onto consecutive integers ``0..n-1``."""
    return [tape.map_to_standard_wires()], null_postprocessing


_TURN = 4 * math.pi  # RZ(theta) = GPI(0) GPI(theta/(4*pi))


# Graph-based decomposition rules: register IonQ-native paths for every standard
# PennyLane gate the device must accept. The graph picks our paths only when the
# target gate set contains GPI/GPI2/MS/IonQZZ (i.e. on the native gateset).
@register_resources({GPI2: 1, GPI: 1})
def _native_h(wires):
    GPI2(0.25, wires=wires)
    GPI(0, wires=wires)


@register_resources({GPI: 1})
def _native_x(wires):
    GPI(0, wires=wires)


@register_resources({GPI: 1})
def _native_y(wires):
    GPI(0.25, wires=wires)


@register_resources({GPI: 2})
def _native_z(wires):
    GPI(0, wires=wires)
    GPI(0.25, wires=wires)


@register_resources({GPI: 2})
def _native_s(wires):
    GPI(0, wires=wires)
    GPI(0.125, wires=wires)


@register_resources({GPI: 2})
def _native_t(wires):
    GPI(0, wires=wires)
    GPI(0.0625, wires=wires)


@register_resources({GPI2: 1})
def _native_sx(wires):
    GPI2(0, wires=wires)


@register_resources({GPI: 2})
def _native_rz(theta, wires):
    GPI(0, wires=wires)
    GPI(theta / _TURN, wires=wires)


@register_resources({GPI2: 2, GPI: 2})
def _native_rx(theta, wires):
    GPI2(-0.25, wires=wires)
    GPI(0, wires=wires)
    GPI(theta / _TURN, wires=wires)
    GPI2(0.25, wires=wires)


@register_resources({GPI2: 2, GPI: 2})
def _native_ry(theta, wires):
    GPI2(0.5, wires=wires)
    GPI(0, wires=wires)
    GPI(-theta / _TURN, wires=wires)
    GPI2(0, wires=wires)


@register_resources({IonQZZ: 1})
def _native_isingzz(theta, wires):
    IonQZZ(theta / (2 * math.pi), wires=wires)


@register_resources({GPI: 2})
def _native_s_dagger(wires, **_):
    GPI(0.125, wires=wires)
    GPI(0, wires=wires)


@register_resources({GPI: 2})
def _native_t_dagger(wires, **_):
    GPI(0.0625, wires=wires)
    GPI(0, wires=wires)


@register_resources({GPI2: 1})
def _native_sx_dagger(wires, **_):
    GPI2(0.5, wires=wires)


for _op_type, _decomp in (
    (qml.Hadamard, _native_h),
    (qml.PauliX, _native_x),
    (qml.PauliY, _native_y),
    (qml.PauliZ, _native_z),
    (qml.S, _native_s),
    (qml.T, _native_t),
    (qml.SX, _native_sx),
    (qml.RZ, _native_rz),
    (qml.RX, _native_rx),
    (qml.RY, _native_ry),
    (qml.IsingZZ, _native_isingzz),
    ("Adjoint(S)", _native_s_dagger),
    ("Adjoint(T)", _native_t_dagger),
    ("Adjoint(SX)", _native_sx_dagger),
):
    add_decomps(_op_type, _decomp)


@transform
def _ionq_decompose(tape, gate_set):
    """Decompose to ``gate_set`` using the graph-based system, scoped to this call.

    ``qml.decomposition.enable_graph`` mutates global state, so we save/restore
    it around the underlying ``qml.transforms.decompose`` call to leave the
    process state untouched for code outside the device.
    """
    was_on = enabled_graph()
    if not was_on:
        enable_graph()
    try:
        return graph_decompose(
            tape,
            gate_set=gate_set,
            fixed_decomps={qml.GlobalPhase: null_decomp},
        )
    finally:
        if not was_on:
            disable_graph()


def _set(value):
    return None if isinstance(value, Unset) else value


@simulator_tracking
@single_tape_support
class IonQDevice(Device):
    """Base PennyLane device targeting IonQ Quantum Cloud."""

    config_filepath = str(Path(__file__).parent / "capabilities.toml")

    def __init__(
        self,
        wires=None,
        *,
        backend: str = "simulator",
        gateset: Literal["qis", "native"] = "qis",
        api_key: str | None = None,
        base_url: str | None = None,
        job_name: str | None = None,
        compilation: dict | None = None,
        error_mitigation: dict | None = None,
        sharpen: bool | None = None,
        noise: dict | None = None,
        metadata: dict | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__(wires=wires)
        self._backend = backend
        self._gateset = gateset
        self._serialize_op = _native_op if gateset == "native" else _qis_op
        self._allowed = _NATIVE_GATES if gateset == "native" else _QIS_GATES
        self._job_name = job_name or UNSET
        self._compilation = compilation
        self._error_mitigation = error_mitigation
        self._sharpen = UNSET if sharpen is None else sharpen
        self._noise = Noise.from_dict(noise) if noise else UNSET
        self._metadata = JobMetadata.from_dict(metadata) if metadata else UNSET
        self._client = IonQClient(
            api_key=api_key,
            base_url=base_url or "https://api.ionq.co/v0.4",
            timeout=httpx.Timeout(timeout) if timeout is not None else None,
            max_retries=max_retries,
            additional_user_agent=f"pennylane-ionq/{_VERSION}",
            raise_on_unexpected_status=True,
        )

    @property
    def name(self) -> str:
        return f"ionq.{self._backend}"

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None):
        program = CompilePipeline()
        program.add_transform(no_analytic, name=self.name)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(_to_standard_wires)
        program.add_transform(
            validate_observables,
            stopping_condition=lambda obs: obs.name in _OBSERVABLES,
            name=self.name,
        )
        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(split_non_commuting)
        program.add_transform(measurements_from_samples)
        program.add_transform(_ionq_decompose, gate_set=set(self._allowed.keys()))
        program.add_transform(broadcast_expand)
        return program

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        if len(circuits) == 1:
            return (self._run_one(circuits[0]),)
        return self._run_many(circuits)

    def _settings(self, settings_cls):
        if not (self._compilation or self._error_mitigation):
            return UNSET
        return settings_cls(
            compilation=(
                CircuitJobCreationPayloadSettingsCompilation.from_dict(self._compilation)
                if self._compilation
                else UNSET
            ),
            error_mitigation=(
                CircuitJobCreationPayloadSettingsErrorMitigation.from_dict(self._error_mitigation)
                if self._error_mitigation
                else UNSET
            ),
        )

    def _gates(self, tape) -> tuple[int, list]:
        n = max(tape.num_wires, 1)
        return n, [self._serialize_op(op) for op in tape.operations]

    def _input(self, qubits, gates):
        cls = NativeCircuitInput if self._gateset == "native" else QisCircuitInput
        return cls(gateset=self._gateset, qubits=qubits, circuit=gates)

    def _child(self, qubits, gates, name):
        cls = NativeCircuit if self._gateset == "native" else QISCircuit
        return cls(qubits=qubits, circuit=gates, name=name)

    def _tape_name(self, tape, fallback=UNSET):
        return getattr(tape, "name", None) or fallback

    def _run_one(self, tape):
        n, gates = self._gates(tape)
        payload = CircuitJobCreationPayload(
            type_="ionq.circuit.v1",
            backend=self._backend,
            input_=self._input(n, gates),
            shots=tape.shots.total_shots,
            name=self._tape_name(tape, self._job_name),
            metadata=self._metadata,
            noise=self._noise,
            settings=self._settings(CircuitJobCreationPayloadSettings),
        )
        job_id, _ = self._submit(payload)
        try:
            result = get_job_probabilities.sync(
                uuid=job_id, client=self._client, sharpen=self._sharpen
            )
        except APIError as err:
            raise DeviceError(f"{self.name}: fetching results failed: {err}") from err
        return self._samples(result.additional_properties, n, tape.shots)

    def _run_many(self, tapes):
        children = [self._gates(t) for t in tapes]
        max_n = max(n for n, _ in children)
        base_name = self._job_name if not isinstance(self._job_name, Unset) else "circuit"
        child_models = [
            self._child(n, gates, name=self._tape_name(t, f"{base_name}-{i}"))
            for i, (t, (n, gates)) in enumerate(zip(tapes, children, strict=True))
        ]
        payload = JSONMultiCircuitJob(
            type_="ionq.multi-circuit.v1",
            backend=self._backend,
            input_=JsonMultiCircuitInput(
                gateset=self._gateset, qubits=max_n, circuits=child_models
            ),
            shots=tapes[0].shots.total_shots,
            name=self._job_name,
            metadata=self._metadata,
            noise=self._noise,
            settings=self._settings(JSONMultiCircuitJobSettings),
        )
        job_id, response = self._submit(payload)
        child_ids = response.child_job_ids or []
        if len(child_ids) != len(tapes):
            raise DeviceError(f"{self.name}: expected {len(tapes)} variants, got {len(child_ids)}")
        results = []
        for variant_id, tape, (n, _) in zip(child_ids, tapes, children, strict=True):
            try:
                result = get_variant_probabilities.sync(
                    uuid=job_id, variant_id=variant_id, client=self._client
                )
            except APIError as err:
                raise DeviceError(f"{self.name}: variant {variant_id} failed: {err}") from err
            results.append(self._samples(result.additional_properties, n, tape.shots))
        return tuple(results)

    def _submit(self, payload):
        try:
            job = create_job.sync(client=self._client, body=payload)
            response = wait_for_job(self._client, job.id)
        except (JobFailedError, JobTimeoutError, APIError) as err:
            raise DeviceError(f"{self.name} job failed: {err}") from err
        self._track(response)
        return job.id, response

    def _track(self, response):
        if not self.tracker.active:
            return
        update: dict = {"job_id": response.id}
        for k in ("predicted_wait_time_ms", "execution_duration_ms"):
            v = getattr(response, k, None)
            if v is not None:
                update[k] = v
        stats = response.stats
        for k in (
            "billed_quantum_compute_time_us",
            "predicted_quantum_compute_time_us",
        ):
            v = _set(getattr(stats, k, UNSET))
            if v is not None:
                update[k] = v
        gate_counts = _set(stats.gate_counts)
        if gate_counts is not None:
            update["gate_counts"] = dict(gate_counts.additional_properties)
        try:
            cost_resp = get_job_cost.sync(uuid=response.id, client=self._client)
        except APIError:
            cost_resp = None
        if cost_resp is not None:
            cost = _set(cost_resp.cost)
            if cost is not None:
                update["cost"] = cost.value
                update["cost_unit"] = cost.unit
        self.tracker.update(**update)

    def _samples(self, raw_probs, n, shots):
        # IonQ returns probabilities keyed by integer-encoded basis states with
        # qubit 0 as the least-significant bit; PennyLane's ``sample_probs``
        # expects the most-significant bit to correspond to wire 0.
        probs = np.zeros(2**n)
        for key, p in raw_probs.items():
            probs[int(format(int(key), f"0{n}b")[::-1], 2)] = p
        total = probs.sum()
        if total > 0:
            probs /= total
        samples = sample_probs(probs, shots.total_shots, n, False, None)
        if shots.has_partitioned_shots:
            return tuple(samples[lo:hi] for lo, hi in shots.bins())
        return samples


class IonQSimulatorDevice(IonQDevice):
    """IonQ noiseless / noisy state-vector simulator."""

    def __init__(self, wires=None, **kwargs):
        super().__init__(wires=wires, backend="simulator", **kwargs)


class IonQQPUDevice(IonQDevice):
    """IonQ trapped-ion QPU (Aria / Forte / Forte-Enterprise)."""

    def __init__(self, wires=None, *, backend: str = "aria-1", **kwargs):
        super().__init__(wires=wires, backend=f"qpu.{backend}", **kwargs)

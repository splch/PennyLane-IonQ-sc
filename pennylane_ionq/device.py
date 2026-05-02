"""PennyLane Device implementation backed by ``ionq_core``."""

from __future__ import annotations

import contextlib
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import pennylane as qml
from ionq_core import (
    UNSET,
    IonQClient,
    IonQError,
    Unset,
    wait_for_job,
)
from ionq_core.api.default import (
    create_job,
    estimate_job_cost,
    get_job_cost,
    get_job_probabilities,
)
from ionq_core.models import (
    CircuitJobCreationPayload,
    CircuitJobCreationPayloadSettings,
    CircuitJobCreationPayloadSettingsCompilation,
    CircuitJobCreationPayloadSettingsErrorMitigation,
    GateCnot,
    GateH,
    GateNativeGate,
    GatePauliexp,
    GateRx,
    GateRy,
    GateRz,
    GateS,
    GateSi,
    GateSwap,
    GateT,
    GateTi,
    GateV,
    GateVi,
    GateX,
    GateXX,
    GateY,
    GateYY,
    GateZ,
    GateZZ,
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
from pennylane.transforms import broadcast_expand, decompose, split_non_commuting

from .ops import GPI, GPI2, IonQZZ

try:
    _VERSION = version("pennylane-ionq")
except PackageNotFoundError:
    _VERSION = "0.0.0"

_OBSERVABLES = frozenset(
    {
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
)

# PennyLane op name -> (ionq-core gate model class, gate literal). Op data
# (rotation) is forwarded automatically when the model accepts it; CNOT is
# the one shape exception (split target/control).
_QIS_GATES: dict[str, tuple[type, str]] = {
    "Hadamard": (GateH, "h"),
    "PauliX": (GateX, "x"),
    "PauliY": (GateY, "y"),
    "PauliZ": (GateZ, "z"),
    "S": (GateS, "s"),
    "T": (GateT, "t"),
    "SX": (GateV, "v"),
    "Adjoint(S)": (GateSi, "si"),
    "Adjoint(T)": (GateTi, "ti"),
    "Adjoint(SX)": (GateVi, "vi"),
    "RX": (GateRx, "rx"),
    "RY": (GateRy, "ry"),
    "RZ": (GateRz, "rz"),
    "SWAP": (GateSwap, "swap"),
    "CNOT": (GateCnot, "cnot"),
    "IsingXX": (GateXX, "xx"),
    "IsingYY": (GateYY, "yy"),
    "IsingZZ": (GateZZ, "zz"),
    "Evolution": (GatePauliexp, "pauliexp"),
}

_NATIVE_GATES: dict[str, str] = {"GPI": "gpi", "GPI2": "gpi2", "MS": "ms", "IonQZZ": "zz"}

# Tolerance for treating PauliSentence coefficients as real.
_PAULIEXP_IMAG_ATOL = 1e-12


def _pauliexp_op(op) -> GatePauliexp | None:
    """Serialize :class:`pennylane.ops.op_math.Evolution` to IonQ ``pauliexp``.

    ``Evolution(H, t)`` is :math:`e^{-i t H}` and IonQ's ``pauliexp`` is
    :math:`e^{-i \\, time \\sum_j c_j P_j}`. The schema requires ``time > 0``,
    so a negative ``t`` is folded into the sign of every coefficient; ``t == 0``
    is the identity and emits no gate. Identity-only Pauli words contribute
    only a global phase and are dropped.

    Pauli strings are big-endian over ``targets``: ``terms[k][i]`` acts on
    ``targets[i]``.
    """
    sentence = op.base.pauli_rep
    if sentence is None:
        raise ValueError(
            f"{op.name}: pauliexp requires a generator with a Pauli "
            f"decomposition (got {op.base!r}). Express the generator as a "
            "LinearCombination of Pauli words."
        )

    targets = list(op.wires)
    terms: list[str] = []
    coefficients: list[float] = []
    for word, coeff in sentence.items():
        if not word:
            continue
        c = complex(coeff)
        if abs(c.imag) > _PAULIEXP_IMAG_ATOL:
            raise ValueError(f"{op.name}: pauliexp requires real Pauli coefficients (got {coeff}).")
        terms.append("".join(word.get(w, "I") for w in targets))
        coefficients.append(c.real)

    time = float(op.data[0])
    if not terms or time == 0:
        return None
    if time < 0:
        time = -time
        coefficients = [-c for c in coefficients]
    return GatePauliexp(
        gate="pauliexp", targets=targets, terms=terms, coefficients=coefficients, time=time
    )


def _qis_op(op):
    if op.name == "Evolution":
        return _pauliexp_op(op)
    cls, gate = _QIS_GATES[op.name]
    wires = list(op.wires)
    if cls is GateCnot:
        return cls(gate=gate, targets=[wires[1]], controls=[wires[0]])
    if op.data:
        return cls(gate=gate, targets=wires, rotation=float(op.data[0]))
    return cls(gate=gate, targets=wires)


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
    return [tape.map_to_standard_wires()], null_postprocessing


# Native-gate decompositions for standard PennyLane gates. Phases are in
# turns; RZ(theta) = GPI(0)*GPI(theta/4pi). Used only when the target gate
# set is native (GPI/GPI2/MS/IonQZZ).
_TURN = 4 * math.pi


def _native(op_type, resources):
    def wrap(fn):
        add_decomps(op_type, register_resources(resources)(fn))
        return fn

    return wrap


@_native(qml.Hadamard, {GPI2: 1, GPI: 1})
def _native_h(wires):
    GPI2(0.25, wires=wires)
    GPI(0, wires=wires)


@_native(qml.PauliX, {GPI: 1})
def _native_x(wires):
    GPI(0, wires=wires)


@_native(qml.PauliY, {GPI: 1})
def _native_y(wires):
    GPI(0.25, wires=wires)


@_native(qml.PauliZ, {GPI: 2})
def _native_z(wires):
    GPI(0, wires=wires)
    GPI(0.25, wires=wires)


@_native(qml.S, {GPI: 2})
def _native_s(wires):
    GPI(0, wires=wires)
    GPI(0.125, wires=wires)


@_native(qml.T, {GPI: 2})
def _native_t(wires):
    GPI(0, wires=wires)
    GPI(0.0625, wires=wires)


@_native(qml.SX, {GPI2: 1})
def _native_sx(wires):
    GPI2(0, wires=wires)


@_native(qml.RZ, {GPI: 2})
def _native_rz(theta, wires):
    GPI(0, wires=wires)
    GPI(theta / _TURN, wires=wires)


@_native(qml.RX, {GPI2: 2, GPI: 2})
def _native_rx(theta, wires):
    GPI2(-0.25, wires=wires)
    GPI(0, wires=wires)
    GPI(theta / _TURN, wires=wires)
    GPI2(0.25, wires=wires)


@_native(qml.RY, {GPI2: 2, GPI: 2})
def _native_ry(theta, wires):
    GPI2(0.5, wires=wires)
    GPI(0, wires=wires)
    GPI(-theta / _TURN, wires=wires)
    GPI2(0, wires=wires)


@_native(qml.IsingZZ, {IonQZZ: 1})
def _native_isingzz(theta, wires):
    IonQZZ(theta / (2 * math.pi), wires=wires)


@_native("Adjoint(S)", {GPI: 2})
def _native_s_dagger(wires, **_):
    GPI(0.125, wires=wires)
    GPI(0, wires=wires)


@_native("Adjoint(T)", {GPI: 2})
def _native_t_dagger(wires, **_):
    GPI(0.0625, wires=wires)
    GPI(0, wires=wires)


@_native("Adjoint(SX)", {GPI2: 1})
def _native_sx_dagger(wires, **_):
    GPI2(0.5, wires=wires)


@contextlib.contextmanager
def _graph_enabled():
    if enabled_graph():
        yield
        return
    enable_graph()
    try:
        yield
    finally:
        disable_graph()


@transform
def _decompose_native(tape, gate_set):
    """Decompose ``tape`` to the IonQ native gate set under graph mode."""
    with _graph_enabled():
        return decompose(tape, gate_set=gate_set, fixed_decomps={qml.GlobalPhase: null_decomp})


def _samples_from_probs(raw_probs: dict, n: int, shots) -> np.ndarray | tuple:
    """Convert IonQ integer-keyed probabilities into PennyLane samples.

    IonQ keys probabilities by integer state with qubit 0 as LSB; ``sample_probs``
    expects qubit 0 as MSB, so reverse the bit order.
    """
    probs = np.zeros(2**n)
    for key, p in raw_probs.items():
        probs[int(format(int(key), f"0{n}b")[::-1], 2)] = p
    if (total := probs.sum()) > 0:
        probs /= total
    samples = sample_probs(probs, shots.total_shots, n, False, None)
    if shots.has_partitioned_shots:
        return tuple(samples[lo:hi] for lo, hi in shots.bins())
    return samples


@simulator_tracking
@single_tape_support
class IonQDevice(Device):
    """PennyLane device targeting IonQ Quantum Cloud.

    Args:
        wires: Device wires (int, iterable, or ``None`` for any).
        backend: IonQ backend identifier (e.g. ``"simulator"``, ``"qpu.aria-1"``).
        gateset: ``"qis"`` (default; logical) or ``"native"`` (gpi/gpi2/ms/zz).
        api_key: IonQ API key. Defaults to ``$IONQ_API_KEY``.
        base_url: API base URL. Defaults to ``https://api.ionq.co/v0.4``.
        job_name: Optional name attached to every submitted job.
        compilation: ``{"opt": float, "precision": str}`` server-side compilation knobs.
        error_mitigation: ``{"debiasing": bool}`` (debiasing requires >= 500 shots).
        sharpen: Apply IonQ's debiasing sharpening when fetching probabilities.
        noise: ``{"model": str, "seed": int}`` cloud-simulator noise (simulator only).
        metadata: User metadata dict (string -> string) attached to each job.
        session_id: Submit jobs into an existing IonQ session.
        dry_run: If ``True``, server only validates and bills nothing.
        timeout: HTTP request timeout in seconds.
        max_retries: Max retries for transient HTTP failures.
        job_timeout: Max wall-clock seconds to wait for a submitted job.
        job_poll_interval: Initial polling interval (seconds) for ``wait_for_job``.
    """

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
        session_id: str | None = None,
        dry_run: bool | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        job_timeout: float = 300.0,
        job_poll_interval: float = 1.0,
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
        self._session_id = session_id or UNSET
        self._dry_run = UNSET if dry_run is None else dry_run
        self._job_timeout = job_timeout
        self._job_poll_interval = job_poll_interval
        self._client = IonQClient(
            api_key=api_key,
            base_url=base_url or "https://api.ionq.co/v0.4",
            timeout=httpx.Timeout(timeout) if timeout is not None else None,
            max_retries=max_retries,
            additional_user_agent=f"pennylane-ionq/{_VERSION}",
        )

    @property
    def name(self) -> str:
        return f"ionq.{self._backend}"

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None):
        program = qml.CompilePipeline()
        program.add_transform(no_analytic, name=self.name)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(
            validate_observables,
            stopping_condition=lambda obs: obs.name in _OBSERVABLES,
            name=self.name,
        )
        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(split_non_commuting)
        program.add_transform(measurements_from_samples)
        program.add_transform(_to_standard_wires)
        if self._gateset == "native":
            program.add_transform(_decompose_native, gate_set=set(self._allowed))
        else:
            program.add_transform(decompose, gate_set=set(self._allowed))
        program.add_transform(broadcast_expand)
        return program

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        if len(circuits) == 1:
            return (self._run_one(circuits[0]),)
        return self._run_many(circuits)

    def estimate_cost(self, qnode_or_tape, *args, **kwargs) -> dict:
        """Server-side cost / queue / runtime estimate.

        Accepts a :class:`~pennylane.QNode` (with ``args``/``kwargs`` for its
        parameters) or a single :class:`~pennylane.tape.QuantumScript`. Returns a
        dict with ``cost``, ``cost_unit``, ``execution_time_s``, ``queue_time_s``,
        and ``rate_information``.
        """
        if isinstance(qnode_or_tape, qml.QNode):
            batch, _ = qml.workflow.construct_batch(qnode_or_tape, level="device")(*args, **kwargs)
        else:
            batch = (qnode_or_tape,)
        if not batch:
            raise ValueError("estimate_cost requires at least one tape")
        shots = batch[0].shots.total_shots if batch[0].shots else 1000
        resp = estimate_job_cost.sync(
            client=self._client,
            backend=self._backend,
            qubits=max(t.num_wires for t in batch),
            shots=shots,
            field_1q_gates=sum(1 for t in batch for op in t.operations if len(op.wires) == 1),
            field_2q_gates=sum(1 for t in batch for op in t.operations if len(op.wires) == 2),
            error_mitigation=bool(self._error_mitigation),
        )
        if resp is None:
            raise DeviceError(f"{self.name}: cost estimate request returned no response")
        return {
            "cost": resp.estimated_cost,
            "cost_unit": resp.cost_unit,
            "execution_time_s": resp.estimated_execution_time,
            "queue_time_s": resp.current_predicted_queue_time,
            "rate_information": resp.rate_information.to_dict(),
        }

    # -- internals ----------------------------------------------------------

    def _settings(self, settings_cls):
        if not (self._compilation or self._error_mitigation):
            return UNSET
        return settings_cls(
            compilation=(
                CircuitJobCreationPayloadSettingsCompilation(**self._compilation)
                if self._compilation
                else UNSET
            ),
            error_mitigation=(
                CircuitJobCreationPayloadSettingsErrorMitigation(**self._error_mitigation)
                if self._error_mitigation
                else UNSET
            ),
        )

    def _common_kwargs(self) -> dict:
        return {
            "backend": self._backend,
            "name": self._job_name,
            "metadata": self._metadata,
            "noise": self._noise,
            "session_id": self._session_id,
            "dry_run": self._dry_run,
        }

    def _gates(self, tape) -> tuple[int, list]:
        n = max(tape.num_wires, 1)
        gates = (self._serialize_op(op) for op in tape.operations)
        return n, [g for g in gates if g is not None]

    def _run_one(self, tape):
        n, gates = self._gates(tape)
        input_cls = NativeCircuitInput if self._gateset == "native" else QisCircuitInput
        payload = CircuitJobCreationPayload(
            type_="ionq.circuit.v1",
            input_=input_cls(gateset=self._gateset, qubits=n, circuit=gates),
            shots=tape.shots.total_shots,
            settings=self._settings(CircuitJobCreationPayloadSettings),
            **self._common_kwargs(),
        )
        job_id, _ = self._submit(payload)
        try:
            result = get_job_probabilities.sync(
                uuid=job_id, client=self._client, sharpen=self._sharpen
            )
        except IonQError as err:
            raise DeviceError(f"{self.name}: fetching results failed: {err}") from err
        return _samples_from_probs(result.additional_properties, n, tape.shots)

    def _run_many(self, tapes):
        children = [self._gates(t) for t in tapes]
        max_n = max(n for n, _ in children)
        child_cls = NativeCircuit if self._gateset == "native" else QISCircuit
        base_name = self._job_name if not isinstance(self._job_name, Unset) else "circuit"
        child_models = [
            child_cls(qubits=n, circuit=gates, name=f"{base_name}-{i}")
            for i, (n, gates) in enumerate(children)
        ]
        payload = JSONMultiCircuitJob(
            type_="ionq.multi-circuit.v1",
            input_=JsonMultiCircuitInput(
                gateset=self._gateset, qubits=max_n, circuits=child_models
            ),
            shots=tapes[0].shots.total_shots,
            settings=self._settings(JSONMultiCircuitJobSettings),
            **self._common_kwargs(),
        )
        _, response = self._submit(payload)
        child_ids = response.child_job_ids or []
        if len(child_ids) != len(tapes):
            raise DeviceError(f"{self.name}: expected {len(tapes)} children, got {len(child_ids)}")
        # child_job_ids are full standalone job UUIDs; fetch each child's
        # probabilities via the per-job endpoint (children complete with the parent).
        results = []
        for child_id, tape, (n, _) in zip(child_ids, tapes, children, strict=True):
            try:
                result = get_job_probabilities.sync(
                    uuid=child_id, client=self._client, sharpen=self._sharpen
                )
            except IonQError as err:
                raise DeviceError(f"{self.name}: child {child_id} failed: {err}") from err
            results.append(_samples_from_probs(result.additional_properties, n, tape.shots))
        return tuple(results)

    def _submit(self, payload):
        try:
            job = create_job.sync(client=self._client, body=payload)
            response = wait_for_job(
                self._client,
                job.id,
                timeout=self._job_timeout,
                poll_interval=self._job_poll_interval,
            )
        except IonQError as err:
            raise DeviceError(f"{self.name} job failed: {err}") from err
        self._track(response)
        return job.id, response

    def _track(self, response):
        if not self.tracker.active:
            return
        s = response.stats
        update: dict = {"job_id": response.id}
        for k, v in {
            "predicted_wait_time_ms": response.predicted_wait_time_ms,
            "predicted_execution_duration_ms": response.predicted_execution_duration_ms,
            "execution_duration_ms": response.execution_duration_ms,
            "predicted_quantum_compute_time_us": s.predicted_quantum_compute_time_us,
            "billed_quantum_compute_time_us": s.billed_quantum_compute_time_us,
            "kwh": s.kwh,
        }.items():
            if v is not None and not isinstance(v, Unset):
                update[k] = v
        if not isinstance(s.gate_counts, Unset):
            update["gate_counts"] = dict(s.gate_counts.additional_properties)
        try:
            cost_resp = get_job_cost.sync(uuid=response.id, client=self._client)
        except IonQError:
            cost_resp = None
        if cost_resp is not None and not isinstance(cost := cost_resp.cost, Unset):
            update["cost"] = cost.value
            update["cost_unit"] = cost.unit
        self.tracker.update(**update)


class IonQSimulatorDevice(IonQDevice):
    """IonQ noiseless / noisy state-vector simulator."""

    def __init__(self, wires=None, **kwargs):
        super().__init__(wires=wires, backend="simulator", **kwargs)


class IonQQPUDevice(IonQDevice):
    """IonQ trapped-ion QPU (Aria / Forte / Forte-Enterprise)."""

    def __init__(self, wires=None, *, backend: str = "aria-1", **kwargs):
        super().__init__(wires=wires, backend=f"qpu.{backend}", **kwargs)

# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""PennyLane device backed by the ``ionq_core`` REST client."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import httpx
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
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    measurements_from_samples,
    no_analytic,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.exceptions import DeviceError
from pennylane.transforms import broadcast_expand, decompose, split_non_commuting

from .decompositions import _decompose_native
from .serialize import (
    _NATIVE_GATES,
    _QIS_GATES,
    _native_op,
    _qis_op,
    _samples_from_probs,
    _to_standard_wires,
)

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
        """Server-side estimate of cost, queue time, runtime, and rate info.

        Accepts a :class:`~pennylane.QNode` (with its ``args``/``kwargs``) or a
        :class:`~pennylane.tape.QuantumScript`. Returns a dict with ``cost``,
        ``cost_unit``, ``execution_time_s``, ``queue_time_s``, and
        ``rate_information``.
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
        Comp = CircuitJobCreationPayloadSettingsCompilation
        Mit = CircuitJobCreationPayloadSettingsErrorMitigation
        return settings_cls(
            compilation=Comp(**self._compilation) if self._compilation else UNSET,
            error_mitigation=Mit(**self._error_mitigation) if self._error_mitigation else UNSET,
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
        return self._fetch_samples(job_id, n, tape.shots)

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
        # child_job_ids are standalone job UUIDs; fetch each child's
        # probabilities via the per-job endpoint (children complete with parent).
        return tuple(
            self._fetch_samples(cid, n, tape.shots)
            for cid, tape, (n, _) in zip(child_ids, tapes, children, strict=True)
        )

    def _fetch_samples(self, job_id, n, shots):
        try:
            result = get_job_probabilities.sync(
                uuid=job_id, client=self._client, sharpen=self._sharpen
            )
        except IonQError as err:
            raise DeviceError(f"{self.name}: fetching {job_id} failed: {err}") from err
        return _samples_from_probs(result.additional_properties, n, shots)

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
    """IonQ cloud state-vector simulator (registered as ``ionq.simulator``).

    Runs noiselessly by default; pass ``noise={"model": "<backend>"}`` to
    emulate the noise profile of a specific QPU. See
    :class:`IonQDevice` for the full keyword reference.
    """

    def __init__(self, wires=None, **kwargs):
        super().__init__(wires=wires, backend="simulator", **kwargs)


class IonQQPUDevice(IonQDevice):
    """IonQ trapped-ion QPU (registered as ``ionq.qpu``).

    The ``backend`` keyword selects a specific QPU -- ``"aria-1"`` (default),
    ``"aria-2"``, ``"forte-1"``, ``"forte-enterprise-1"`` / ``-2`` / ``-3``,
    or ``"tempo-1"``. See :class:`IonQDevice` for the full keyword reference.
    """

    def __init__(self, wires=None, *, backend: str = "aria-1", **kwargs):
        super().__init__(wires=wires, backend=f"qpu.{backend}", **kwargs)

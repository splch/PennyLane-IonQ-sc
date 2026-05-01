"""Augment ``qml.specs`` output with IonQ predicted cost / queue / execution time.

Importing this module monkey-patches ``pennylane.resource.specs._specs_qnode``
so that ``qml.specs(qnode)(...)`` for an IonQ device returns a subclass of
``CircuitSpecs`` populated with the result of ``estimate_job_cost``. Calls
on non-IonQ devices are passed through untouched.
"""

import importlib
from dataclasses import dataclass

import pennylane as qml
from ionq_core import APIError
from ionq_core.api.default import estimate_job_cost
from pennylane.resource import CircuitSpecs

from .device import IonQDevice

# ``pennylane.resource.specs`` is shadowed by the ``specs`` function inside the
# package's ``__init__.py``; resolve the underlying module via ``importlib``.
_specs_module = importlib.import_module("pennylane.resource.specs")


@dataclass(frozen=True)
class IonQCircuitSpecs(CircuitSpecs):
    """``CircuitSpecs`` extended with IonQ predicted cost and timing."""

    estimated_cost: float | None = None
    cost_unit: str | None = None
    estimated_execution_time_s: float | None = None
    estimated_queue_time_s: float | None = None

    def __str__(self) -> str:
        base = super().__str__()
        if self.estimated_cost is None:
            return base
        return (
            f"{base}\n\n"
            f"IonQ estimate:\n"
            f"  Cost: {self.estimated_cost:.4f} {self.cost_unit}\n"
            f"  Execution time: {self.estimated_execution_time_s:.2f} s\n"
            f"  Queue time: {self.estimated_queue_time_s:.2f} s"
        )


def _device_batch_resources(qnode, args, kwargs):
    """Return (n_qubits, n_1q_gates, n_2q_gates, n_shots) at device level."""
    batch, _ = qml.workflow.construct_batch(qnode, level="device")(*args, **kwargs)
    if not batch:
        return None
    n_qubits = max(t.num_wires for t in batch)
    n_1q = sum(1 for t in batch for op in t.operations if len(op.wires) == 1)
    n_2q = sum(1 for t in batch for op in t.operations if len(op.wires) == 2)
    n_shots = batch[0].shots.total_shots if batch[0].shots else 1000
    return n_qubits, n_1q, n_2q, n_shots


def _estimate(qnode, args, kwargs):
    device = qnode.device
    counts = _device_batch_resources(qnode, args, kwargs)
    if counts is None:
        return None
    n_qubits, n_1q, n_2q, n_shots = counts
    try:
        return estimate_job_cost.sync(
            client=device._client,
            backend=device._backend,
            qubits=n_qubits,
            shots=n_shots,
            field_1q_gates=n_1q,
            field_2q_gates=n_2q,
            error_mitigation=bool(device._error_mitigation),
        )
    except APIError:
        return None


_original_specs_qnode = _specs_module._specs_qnode


def _patched_specs_qnode(qnode, level, compute_depth, *args, **kwargs):
    base = _original_specs_qnode(qnode, level, compute_depth, *args, **kwargs)
    if not isinstance(qnode.device, IonQDevice):
        return base

    fields: dict = dict(
        device_name=base.device_name,
        num_device_wires=base.num_device_wires,
        shots=base.shots,
        level=base.level,
        resources=base.resources,
    )
    estimate = _estimate(qnode, args, kwargs)
    if estimate is not None:
        fields.update(
            estimated_cost=estimate.estimated_cost,
            cost_unit=estimate.cost_unit,
            estimated_execution_time_s=estimate.estimated_execution_time,
            estimated_queue_time_s=estimate.current_predicted_queue_time,
        )
    return IonQCircuitSpecs(**fields)


_specs_module._specs_qnode = _patched_specs_qnode

# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the PennyLane-IonQ test suite.

The ``mock_ionq`` fixture patches the three ``ionq_core`` call sites
(``create_job.sync``, ``wait_for_job``, ``get_job_probabilities.sync``) at
the ``pennylane_ionq.device`` import site, so unit tests never hit the
network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

# Tests should not pick up real credentials from the developer's shell.
os.environ.setdefault("IONQ_API_KEY", "test-key-not-real")


@dataclass
class _CreatedJob:
    id: str = "job-123"


@dataclass
class _Stats:
    predicted_quantum_compute_time_us: Any = None
    billed_quantum_compute_time_us: Any = None
    kwh: Any = None
    gate_counts: Any = None


@dataclass
class _JobResponse:
    """Subset of ``ionq_core`` job-response fields read by ``IonQDevice``."""

    id: str = "job-123"
    child_job_ids: list[str] = field(default_factory=list)
    predicted_wait_time_ms: Any = None
    predicted_execution_duration_ms: Any = None
    execution_duration_ms: Any = None
    stats: _Stats = field(default_factory=_Stats)


@dataclass
class _Probabilities:
    additional_properties: dict[int, float]


@pytest.fixture
def mock_ionq(monkeypatch):
    """Patch the IonQ Cloud calls used by ``IonQDevice``.

    Returns a handle so tests can override return values or assert on calls.
    """

    handle = MagicMock()
    handle.created_job = _CreatedJob()
    handle.job_response = _JobResponse()
    # Default: equal superposition over 2 qubits. IonQ keys probabilities
    # by integer state with qubit 0 as LSB.
    handle.probabilities = _Probabilities(additional_properties={0: 0.5, 1: 0.5})

    def fake_create_job_sync(*, client, body):
        handle.create_job_calls.append({"client": client, "body": body})
        return handle.created_job

    def fake_wait_for_job(client, job_id, *, timeout=None, poll_interval=None):
        handle.wait_for_job_calls.append(
            {"client": client, "job_id": job_id, "timeout": timeout, "poll_interval": poll_interval}
        )
        return handle.job_response

    def fake_get_job_probabilities_sync(*, uuid, client, sharpen=None):
        handle.get_probabilities_calls.append({"uuid": uuid, "client": client, "sharpen": sharpen})
        return handle.probabilities

    handle.create_job_calls = []
    handle.wait_for_job_calls = []
    handle.get_probabilities_calls = []

    monkeypatch.setattr("pennylane_ionq.device.create_job.sync", fake_create_job_sync, raising=True)
    monkeypatch.setattr("pennylane_ionq.device.wait_for_job", fake_wait_for_job, raising=True)
    monkeypatch.setattr(
        "pennylane_ionq.device.get_job_probabilities.sync",
        fake_get_job_probabilities_sync,
        raising=True,
    )
    return handle

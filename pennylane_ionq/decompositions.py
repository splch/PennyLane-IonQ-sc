# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native-gate decompositions for standard PennyLane gates.

Registers rules with PennyLane's graph decomposition system. Phases are in
turns (e.g. ``RZ(theta) = GPI(0) * GPI(theta/4pi)``). Used only when the
device's gate set is ``"native"``.
"""

from __future__ import annotations

import math

import pennylane as qml
from pennylane import transform
from pennylane.decomposition import (
    add_decomps,
    disable_graph,
    enable_graph,
    enabled_graph,
    null_decomp,
    register_resources,
)
from pennylane.transforms import decompose

from .ops import GPI, GPI2, IonQZZ

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


# GlobalPhase is irrelevant for samples; pre-register so callers don't need
# to pass ``fixed_decomps={qml.GlobalPhase: null_decomp}`` to ``decompose``.
add_decomps(qml.GlobalPhase, null_decomp)


@transform
def _decompose_native(tape, gate_set):
    """Decompose ``tape`` to the IonQ native gate set, enabling graph mode if needed."""
    was_enabled = enabled_graph()
    enable_graph()
    try:
        return decompose(tape, gate_set=gate_set)
    finally:
        if not was_enabled:
            disable_graph()

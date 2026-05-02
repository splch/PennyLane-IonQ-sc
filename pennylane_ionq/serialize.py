"""Serialization between PennyLane ops and the ionq-core wire format.

Per-gate translators convert :class:`~pennylane.operation.Operation` instances
to :mod:`ionq_core.models` payload objects. The probabilities-to-samples
helper closes the loop on the result side.
"""

from __future__ import annotations

import numpy as np
from ionq_core.models import (
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
)
from pennylane import transform
from pennylane.devices.preprocess import null_postprocessing
from pennylane.devices.qubit.sampling import sample_probs

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

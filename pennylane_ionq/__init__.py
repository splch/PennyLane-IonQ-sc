# Copyright 2019 Xanadu Quantum Technologies Inc.
# SPDX-License-Identifier: Apache-2.0
"""PennyLane plugin for IonQ Quantum Cloud.

Re-exports the device classes (registered as ``ionq.simulator`` and
``ionq.qpu``) and IonQ's native trapped-ion gates.
"""

from .device import _VERSION, IonQDevice, IonQQPUDevice, IonQSimulatorDevice
from .ops import GPI, GPI2, MS, IonQZZ

__version__ = _VERSION

__all__ = [
    "GPI",
    "GPI2",
    "MS",
    "IonQDevice",
    "IonQQPUDevice",
    "IonQSimulatorDevice",
    "IonQZZ",
    "__version__",
]

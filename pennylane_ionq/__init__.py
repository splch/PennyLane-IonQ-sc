"""PennyLane plugin for IonQ Quantum Cloud."""

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

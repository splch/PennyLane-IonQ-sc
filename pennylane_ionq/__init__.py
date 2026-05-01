"""PennyLane plugin for IonQ Quantum Cloud."""

from . import specs as _specs  # noqa: F401  # patches qml.specs to include IonQ cost
from .device import _VERSION, IonQDevice, IonQQPUDevice, IonQSimulatorDevice
from .ops import GPI, GPI2, MS, XX, YY, ZZ, IonQZZ

__version__ = _VERSION

__all__ = [
    "GPI",
    "GPI2",
    "MS",
    "XX",
    "YY",
    "ZZ",
    "IonQDevice",
    "IonQQPUDevice",
    "IonQSimulatorDevice",
    "IonQZZ",
    "__version__",
]

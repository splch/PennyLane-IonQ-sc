# Release 0.45.0-dev

### New features since last release

* Added a `"native"` gate set: `gateset="native"` compiles circuits to
  `GPI` / `GPI2` / `MS` / `IonQZZ` via PennyLane's graph decomposition
  before submission.

* Added `IonQDevice.estimate_cost(qnode_or_tape, *args, **kwargs)`,
  returning IonQ Cloud's server-side estimate of cost, queue time,
  runtime, and rate information.

* Multi-circuit submission now uses the `ionq.multi-circuit.v1` payload,
  so a tape batch ships as a single job. No API change for users.

* Added a `session_id` device keyword to submit jobs into an existing
  IonQ Cloud session.

* Added `IonQZZ`, the native ZZ interaction. `qml.IsingZZ` decomposes to
  `IonQZZ` when the gate set is `"native"`.

* Tracker integration: `device.tracker` records job ID, predicted/actual
  runtimes, billed compute time, energy (`kwh`), gate counts, and cost.

### Improvements 🛠

* Migrated to PennyLane's new device API (`pennylane.devices.Device`).
  Capabilities are declared in `capabilities.toml` (schema 3); preprocessing
  composes the standard `no_analytic`, `validate_device_wires`,
  `validate_observables`, `validate_measurements`, `split_non_commuting`,
  `measurements_from_samples`, and `broadcast_expand` transforms with the
  plugin's gate-set decomposition step.

* Replaced the in-tree REST client with
  [`ionq-core`](https://github.com/ionq/ionq-core-python), IonQ's official
  typed Python client for the v0.4 API. The plugin no longer ships
  `api_client.py`, custom resource abstractions, or hand-rolled retry logic.

* `Evolution` operations are serialized via PennyLane's `pauli_rep`,
  replacing the legacy `_decompose_evolution` path. Negative evolution
  times are folded into coefficient signs to satisfy IonQ's `time > 0`
  schema; identity-only Pauli words (global phases) are dropped.

* Native-gate decompositions for `Hadamard`, `Pauli{X,Y,Z}`, `S`, `T`,
  `SX`, `Adjoint(S/T/SX)`, `RX`, `RY`, `RZ`, and `IsingZZ` are registered
  with `add_decomps`.

### Breaking changes 💔

* Renamed device classes to `IonQDevice`, `IonQSimulatorDevice`, and
  `IonQQPUDevice`. Entry-point names (`ionq.simulator`, `ionq.qpu`) are
  unchanged, so `qml.device("ionq.simulator", ...)` still works.

* Removed the `XX`, `YY`, `ZZ` operation classes. Use `IsingXX` / `IsingYY`
  / `IsingZZ` (or `qml.evolve(...)` for arbitrary Pauli evolutions).

* Removed `api_client.py` and its exception hierarchy
  (`CircuitIndexNotSetException`, `MethodNotSupportedException`,
  `ObjectAlreadyCreatedException`, `JobNotQueuedError`,
  `JobExecutionError`). Errors now surface as
  `pennylane.exceptions.DeviceError` wrapping `ionq_core.IonQError`.

* Removed the legacy Evolution-specific exceptions
  (`NotSupportedEvolutionInstance`,
  `OperatorNotSupportedInEvolutionGateGenerator`,
  `ComplexEvolutionCoefficientsNotSupported`). Ill-formed evolutions now
  raise `ValueError` from the serializer.

* Minimum supported PennyLane is `0.44`; minimum supported Python is
  `3.11`.

### Deprecations 👋

### Documentation 📝

* The README, Sphinx documentation, and CI workflows have been refreshed
  for the new backend.

### Bug fixes 🐛

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Spencer Churchill.

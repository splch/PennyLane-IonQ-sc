IonQ Devices
============

PennyLane-IonQ provides two devices: ``ionq.simulator`` and ``ionq.qpu``.
Both subclass :class:`~pennylane_ionq.IonQDevice` and accept the same
options; they differ only in the cloud backend they target.

Set ``IONQ_API_KEY`` in your environment, or pass ``api_key="..."`` to the
constructor.

.. _simulator:

Cloud trapped-ion simulator
---------------------------

:class:`~pennylane_ionq.IonQSimulatorDevice` (``ionq.simulator``) runs on
IonQ's cloud state-vector simulator. By default it is noiseless. Pass
``noise={"model": "<backend>"}`` to emulate a specific QPU; ``seed`` makes
the noise realisation deterministic.

.. code-block:: python

    import pennylane as qml

    dev = qml.device(
        "ionq.simulator",
        wires=2,
        shots=1000,
        noise={"model": "aria-1", "seed": 42},
    )

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.IsingYY(y, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

.. _qpu:

Trapped-ion QPU
---------------

:class:`~pennylane_ionq.IonQQPUDevice` (``ionq.qpu``) runs on IonQ
hardware. Pick a QPU with the ``backend`` argument; jobs are queued and
dispatched by IonQ Cloud.

.. code-block:: python

    import pennylane as qml

    dev = qml.device("ionq.qpu", backend="aria-1", wires=2, shots=1000)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.IsingXX(x, wires=[0, 1])
        qml.IsingYY(y, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

Available backends: ``aria-1``, ``aria-2``, ``forte-1``,
``forte-enterprise-1`` / ``-2`` / ``-3``, and ``tempo-1``. See the
`IonQ API reference
<https://docs.ionq.com/api-reference/v0.4/introduction>`_ for canonical
capabilities.

Gate sets
---------

The ``gateset`` keyword selects how the device accepts circuits:

* ``"qis"`` (default) -- Pauli, Hadamard, S, T, SX (and adjoints),
  RX/RY/RZ, CNOT, SWAP, IsingXX/YY/ZZ, plus ``qml.evolve(...)``
  (lowered to IonQ's ``pauliexp``).

* ``"native"`` -- :class:`~pennylane_ionq.GPI`,
  :class:`~pennylane_ionq.GPI2`, :class:`~pennylane_ionq.MS`,
  :class:`~pennylane_ionq.IonQZZ`. Standard gates auto-decompose to native
  ones via PennyLane's graph decomposition.

.. code-block:: python

    from pennylane_ionq import GPI2, MS

    dev = qml.device("ionq.simulator", wires=2, shots=1000, gateset="native")

    @qml.qnode(dev)
    def circuit():
        GPI2(0.0, wires=0)
        MS(0.0, 0.0, theta=0.25, wires=[0, 1])
        return qml.probs(wires=[0, 1])

IonQ-native operations
~~~~~~~~~~~~~~~~~~~~~~

Phases (``phi``, ``phi0``, ``phi1``) are in **turns** (fractions of
:math:`2\pi`); interaction parameters (``theta``, ``angle``) are in
**units of** :math:`\pi`. This matches ``ionq_core.gates`` and the IonQ
Cloud wire format.

.. autosummary::

    ~pennylane_ionq.GPI
    ~pennylane_ionq.GPI2
    ~pennylane_ionq.MS
    ~pennylane_ionq.IonQZZ

Configuring jobs
----------------

Device keyword arguments:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Keyword
     - Purpose
   * - ``wires``
     - Number of qubits or wire labels.
   * - ``backend``
     - IonQ backend identifier (``"simulator"``, ``"qpu.aria-1"``, ...).
       The simulator and QPU device classes set this automatically.
   * - ``gateset``
     - ``"qis"`` (default) or ``"native"``.
   * - ``shots``
     - Sample count per circuit. Required; analytic execution is rejected.
   * - ``api_key``
     - IonQ API key. Defaults to ``$IONQ_API_KEY``.
   * - ``base_url``
     - API base URL. Defaults to ``https://api.ionq.co/v0.4``.
   * - ``job_name``
     - Optional name attached to each submitted job.
   * - ``compilation``
     - ``{"opt": float, "precision": str}``: server-side compilation knobs.
   * - ``error_mitigation``
     - ``{"debiasing": bool}``. Debiasing requires ``shots >= 500``.
   * - ``sharpen``
     - Apply IonQ's sharpening when fetching probabilities.
   * - ``noise``
     - ``{"model": str, "seed": int}``: simulator-only noise model.
   * - ``metadata``
     - ``dict[str, str]`` attached to each job.
   * - ``session_id``
     - Submit into an existing IonQ Cloud session.
   * - ``dry_run``
     - Server validates only; nothing is billed.
   * - ``timeout``
     - HTTP request timeout in seconds.
   * - ``max_retries``
     - Retries for transient HTTP failures.
   * - ``job_timeout``
     - Seconds to wait for a job (default ``300``).
   * - ``job_poll_interval``
     - Initial poll interval for ``wait_for_job`` (default ``1.0``).

Multi-circuit batching
----------------------

When PennyLane hands the device a batch of tapes -- for example, after a
non-commuting measurement split -- the plugin packs them into a single
``ionq.multi-circuit.v1`` job. IonQ Cloud returns one parent job whose
``child_job_ids`` are fetched per-circuit. This is automatic.

Server-side cost estimation
---------------------------

:meth:`~pennylane_ionq.IonQDevice.estimate_cost` returns IonQ's estimate
of cost, queue time, runtime, and rate information for a QNode or tape:

.. code-block:: python

    dev = qml.device("ionq.qpu", backend="aria-1", wires=2, shots=1000)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    dev.estimate_cost(circuit, 0.5)
    # {'cost': ..., 'cost_unit': 'USD',
    #  'execution_time_s': ..., 'queue_time_s': ...,
    #  'rate_information': {...}}

Job tracking
------------

Wrap a call in :class:`~pennylane.Tracker` to capture per-job metrics:

.. code-block:: python

    with qml.Tracker(dev) as tracker:
        circuit(0.5)

    tracker.history
    # {'job_id': [...], 'execution_duration_ms': [...],
    #  'billed_quantum_compute_time_us': [...], 'kwh': [...],
    #  'gate_counts': [...], 'cost': [...], 'cost_unit': [...]}

Fields IonQ does not return for a backend (e.g.
``billed_quantum_compute_time_us`` on the simulator) are omitted rather
than recorded as ``None``.

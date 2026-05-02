PennyLane-IonQ Plugin
#####################

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-ionq/tests.yml?branch=master&logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane-ionq/actions?query=workflow%3ATests

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-ionq/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-ionq

.. image:: https://readthedocs.com/projects/xanaduai-pennylane-ionq/badge/?version=latest&style=flat-square
    :alt: Read the Docs
    :target: https://docs.pennylane.ai/projects/ionq

.. image:: https://img.shields.io/pypi/v/PennyLane-ionq.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-ionq

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-ionq.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-ionq

.. header-start-inclusion-marker-do-not-remove

The PennyLane-IonQ plugin connects `PennyLane <https://pennylane.ai>`_ to
IonQ Cloud, giving QNodes access to IonQ's trapped-ion QPUs and cloud
simulator.

.. header-end-inclusion-marker-do-not-remove

Documentation: https://docs.pennylane.ai/projects/ionq

Features
========

* Two devices: ``ionq.simulator`` (cloud state-vector simulator, optionally
  with a noise model) and ``ionq.qpu`` (Aria, Forte, Forte-Enterprise,
  Tempo).

* Two gate sets via ``gateset``:

  - ``"qis"`` (default) -- standard PennyLane gates plus ``qml.evolve(...)``,
    lowered to IonQ's ``pauliexp`` instruction.
  - ``"native"`` -- :class:`~.GPI`, :class:`~.GPI2`, :class:`~.MS`,
    :class:`~.IonQZZ`. Standard gates auto-decompose to native ones.

* Server-side cost / queue / runtime estimates via
  :meth:`~.IonQDevice.estimate_cost`.

* Automatic multi-circuit batching via IonQ's
  ``ionq.multi-circuit.v1`` payload.

* Job options as device keywords: ``compilation``, ``error_mitigation``,
  ``sharpen``, ``noise``, ``metadata``, ``session_id``, ``dry_run``.

* Tracker integration: ``device.tracker`` records job ID, runtime, billed
  compute time, kWh, gate counts, and cost.

* Built on `ionq-core <https://github.com/ionq/ionq-core-python>`_, IonQ's
  typed Python client for the v0.4 API.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

PennyLane-IonQ requires Python 3.11+ and PennyLane 0.44+. Install from PyPI:

.. code-block:: bash

    $ python -m pip install pennylane-ionq

Or from a source checkout:

.. code-block:: bash

    $ python -m pip install .

Run the tests with ``make test`` and build the docs with ``make docs``
(output in ``doc/_build/html/``).

.. installation-end-inclusion-marker-do-not-remove

Getting started
===============

Set ``IONQ_API_KEY`` in your environment, or pass ``api_key="..."`` to the
device constructor.

.. code-block:: python

    import pennylane as qml

    dev = qml.device("ionq.simulator", wires=2, shots=1000)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    circuit(0.123)

Submitting to a real QPU is a one-line change:

.. code-block:: python

    dev = qml.device("ionq.qpu", backend="aria-1", wires=2, shots=1000,
                     error_mitigation={"debiasing": True}, sharpen=True)

See the `documentation <https://docs.pennylane.ai/projects/ionq>`_ for the
full device reference.

Contributing
============

Fork the repository and open a pull request against ``master``. Bug
reports and feature requests are welcome on the
`issue tracker <https://github.com/PennyLaneAI/pennylane-ionq/issues>`_.

Contributors
============

PennyLane-IonQ is the work of many `contributors
<https://github.com/PennyLaneAI/pennylane-ionq/graphs/contributors>`_.

If you use PennyLane in your research, please cite:

    Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid
    quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

    Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and
    Nathan Killoran. *Evaluating analytic gradients on quantum hardware.* 2018.
    `Phys. Rev. A 99, 032331 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331>`_

.. support-start-inclusion-marker-do-not-remove

Support
=======

- Source: https://github.com/PennyLaneAI/pennylane-ionq
- Issues: https://github.com/PennyLaneAI/pennylane-ionq/issues

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

Apache License, Version 2.0.

.. license-end-inclusion-marker-do-not-remove

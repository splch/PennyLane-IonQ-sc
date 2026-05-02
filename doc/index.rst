PennyLane-IonQ Plugin
#####################

:Release: |release|

.. include:: ../README.rst
  :start-after: header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Once installed, the two devices are available through ``qml.device(...)``
without any extra imports.

Devices
=======

.. title-card::
    :name: 'ionq.simulator'
    :description: Cloud trapped-ion simulator (noiseless or noisy).
    :link: devices.html#simulator

.. title-card::
    :name: 'ionq.qpu'
    :description: IonQ trapped-ion QPUs (Aria, Forte, Forte-Enterprise, Tempo).
    :link: devices.html#qpu

.. raw:: html

    <div style='clear:both'></div>
    </br>

Both devices accept the same options: gate sets (``"qis"`` or
``"native"``), multi-circuit batching, server-side cost estimation, and
tracker integration.

Remote backend access
=====================

Set ``IONQ_API_KEY`` in your environment, or pass ``api_key="..."`` to the
device constructor.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/__init__
   code/ops

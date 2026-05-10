Examples
=========

The in-depth examples below are meant to demonstrate ASSET's capabilities as a generic optimal control and trajectory design software.
Many more examples may also be found in the `examples <https://github.com/GrantHecht/tycho/tree/master/examples>`_ folder of the ASSET Github Repo.

.. warning::

   These example pages largely pre-date the Tycho Python-API snake_case
   rename and the legacy OCP link-API removal. Inline code samples often
   use camelCase identifiers (``addBoundaryValue``, ``addPhase``, ``XVar``,
   ``vf.Stack``, ``setLinkParams``, ``LinkConstraint``) that no longer
   exist in the bindings. Copy-pasting samples will fail with
   ``AttributeError`` or ``TypeError``. For runnable, current code see the
   matching scripts under ``examples/python_examples/`` (e.g.
   ``Delta3Launch.py``, ``Reentry.py``, ``CartPole.py``,
   ``HyperSens.py``, ``Heteroclinic.py``, ``Zermelo.py``,
   ``MultiPhaseZermelo.py``, ``MultiSpacecraftOptimization.py``).

.. toctree::
    :maxdepth: 2

    Delta3
    ReentryExample
    CartPole
    HyperSensitive
    halo
    zermelo
    zermelolink
    MultiTarg
    
    

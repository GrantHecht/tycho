.. ASSET documentation master file, created by
   sphinx-quickstart on Mon Jan 27 14:56:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ASSET's documentation!
=================================

ASSET (Astrodynamics Software and Science Enabling Toolkit) is a modular, extensible library for trajectory design and optimal control.
It uses a custom implementation of vector math formalisms to enable rapid implementation of dynamical systems and automatic differentiation.
The phase object is the core of the optimal control functionality, and by linking multiple phases together, the user can construct scenarios of arbitrary complexity.
A newly developed high-performance interior-point optimizer (PSIOPT) is included with the library, which enables quick turnaround from concept to solution.

.. warning::

   Many code samples in this documentation pre-date the Tycho Python-API
   snake_case rename and use camelCase identifiers (``addBoundaryValue``,
   ``XVar``, ``setControlMode``, ``vf.Stack``, ``LinkConstraint``, etc.) and
   old module paths (``tycho.Astro``, ``tycho.OptimalControl``). Copy-pasting
   will fail with ``AttributeError``. For runnable, current code see the
   scripts under ``examples/python_examples/``. Heavily-affected pages carry
   their own per-file warning banners; in their absence, prefer the
   snake_case API exposed by the runtime.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    tutorials/tutorials.rst
    examples/examples.rst
    python/python.rst

Citation
########

Cite the following `paper <https://www.researchgate.net/publication/357567956_ASSET_Astrodynamics_Software_and_Science_Enabling_Toolkit>`_ if you used ASSET in your work.

.. note::
    .. raw:: latex

    @inproceedings{Pezent2022,
        author = {James B. Pezent and Jared Sikes and William Ledbetter and Rohan Sood and Kathleen C. Howell and Jeffrey R. Stuart},
        title = {ASSET: Astrodynamics Software and Science Enabling Toolkit},
        booktitle = {AIAA SCITECH 2022 Forum},
        pages = {AIAA 2022-1131},
        year={2022},
        doi = {10.2514/6.2022-1131}
    }



If you have questions, please email any of the following or post an issue on the GitHub:

    ajhouin@crimson.ua.edu

    cgsandel@crimson.ua.edu

    rsood@eng.ua.edu

    jbpezent@crimson.ua.edu

    jdsikes1@crimson.ua.edu

    wgledbetter@crimson.ua.edu

Funding Acknowledgment
######################
Open source development and distribution of ASSET is funded by NASA under Grant No. 80NSSC19K1643 as part of the
`C.29 element of the 2018 ROSES program <https://nspires.nasaprs.com/external/solicitations/summary!init.do?solId=CEB5907A-57A0-379C-7B48-2F538EEB716E>`_.
 

Copyright, Licensing, and Legal Notices
#######################################
ASSET is distributed under the permissive Apache 2.0 license that can be found in the `LICENSE <https://github.com/GrantHecht/tycho/blob/master/LICENSE.txt>`_ file on the Github repo.

The license and copyright notices of ASSET's source and redistributable dependencies are shown below. The full text for each respective license can be found in the 
`notices <https://github.com/GrantHecht/tycho/tree/master/notices>`_ folder on the Github repo.

    Nanobind      - BSD License : Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

    Intel MKL     - Intel Simplified Software License : Copyright (c) 2022 Intel Corporation.

    Eigen         - MPL-2.0 License : Copyright (c) Eigen Developers

    fmt           - MIT License : Copyright (c) 2012 - present, Victor Zverovich

    autodiff      - MIT License : Copyright (c) 2018–2024 Allan Leal

    boost-threads - Boost Software License : Copyright (c) 2003-2022

    rubber_types  - MIT License : Copyright (C) 2014 Andreas J. Herrmann

    kepler propagator - MIT License : Copyright (C) 2014 Darin Koblick

    lambert solver - MIT License : Copyright (C) 2018 Rody Oldenhuis

..
    cpp/cpp_root.rst
    install
    contrib



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

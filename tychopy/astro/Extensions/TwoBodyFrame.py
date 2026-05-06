# =============================================================================
# Originally from ASSET (AlabamaASRL/asset_asrl)
# Copyright 2020-present The University of Alabama-Astrodynamics and Space
#   Research Lab. Licensed under the Apache License, Version 2.0
# License: notices/asset-apache2.txt.
# Source: https://github.com/AlabamaASRL/asset_asrl
# Original Developer: James B. Pezent
#
# Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
#   Apache 2.0 — see LICENSE.txt):
#   - Package renamed: asset_asrl -> tycho
#   - Module renamed: asset_asrl (pybind11) -> _tychopy (nanobind)
#   - Imports updated accordingly
# =============================================================================

import _tychopy as ast
import numpy as np

vf = ast.vector_functions
oc = ast.optimal_control
Args = vf.Arguments


class TwoBodyFrame:
    """
    Two body dynamics

    Attributes
    ----------
    mu : float
        system non-dimensional gravity parameter
    P1mu : float
        system dimensional gravity parameter
    lstar : float
        system characteristic length
    tstar : float
        system characteristic time
    vstar : float
        system characteristic velocity
    astar : float
        system characteristic acceleration
    mustar : float
        system characteristic gravity parameter
    """

    def __init__(self, P1mu, lstar):
        """
        TwoBodyFrame init function. Be sure that P1mu and lstar are of the same units.

        Parameters
        ----------
        P1mu : float
            gravitational constant for primary
        lstar : float
            characteristic length of system
        """
        self.mu = 1
        self.P1mu = P1mu
        self.lstar = lstar
        self.tstar = np.sqrt((lstar**3) / (P1mu))
        self.vstar = lstar / self.tstar
        self.astar = (P1mu) / (lstar**2)
        self.mustar = P1mu

    def TwoBodyEOMs(self, r, v, otherAccs=[], otherEOMs=[]):
        """
        Two body equations of motion

        Parameters
        ----------
        r : ASSET VectorFunction
            3x1 Particle position from primary
        v : ASSET VectorFunction
            3x1 particle velocity
        otherAccs : list
            list of other accelerations to add to model
        otherEOMS : list
            list of other equations of motion driving model

        Returns
        -------
        Full equations of motion : ASSET VectorFunction
            All equations of motion for two body model

        """
        accG = (-self.mu) * r.normalized_power3()
        accT = vf.sum([accG] + otherAccs)
        return vf.stack([v, accT] + otherEOMs)

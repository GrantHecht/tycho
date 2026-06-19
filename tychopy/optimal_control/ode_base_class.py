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

import inspect
import numbers

import _tychopy as _tychopy
import numpy as np


class ODEBase:
    """Base class for user-defined ordinary differential equations.

    Subclass :class:`ODEBase` and, in ``__init__``, build the right-hand side
    of the dynamics as a :class:`~tychopy.vector_functions.VectorFunction` over
    an :class:`~tychopy.optimal_control.ODEArguments` input, then call
    ``super().__init__(ode, Xvars, Uvars)``. The resulting object can spawn
    collocation phases (:meth:`phase`) and integrators (:meth:`integrator`) for
    the dynamics.

    The ODE input is the ordered vector ``[x, t, u, p]`` -- the state ``x``
    (size ``Xvars``), time ``t``, control ``u`` (size ``Uvars``), and ODE
    parameters ``p`` (size ``Pvars``); the right-hand side returns the state
    derivative ``xdot`` (size ``Xvars``).

    Examples
    --------
    Define the brachistochrone dynamics as an ``ODEBase`` subclass::

        class Brachistochrone(oc.ODEBase):
            def __init__(self, g):
                args = oc.ODEArguments(3, 1)
                x, y, v = args.x_vec().tolist()
                theta = args.u_var(0)
                ode = vf.stack([vf.sin(theta) * v, -vf.cos(theta) * v, g * vf.cos(theta)])
                super().__init__(ode, 3, 1)
    """

    def __init__(self, odefunc, Xvars, Uvars=None, Pvars=None, Vgroups=None):
        """Bind a right-hand-side function as a sized ODE.

        Parameters
        ----------
        odefunc : VectorFunction
            The right-hand side ``xdot = f(x, t, u, p)``, an output of size
            ``Xvars`` defined over the packed ``[x, t, u, p]`` input.
        Xvars : int
            Number of state variables.
        Uvars : int, optional
            Number of control variables. ``None`` or ``0`` means no controls.
        Pvars : int, optional
            Number of ODE parameters. ``None`` or ``0`` means no parameters.
        Vgroups : dict, optional
            Named variable groups, a dict mapping a name (str) or sequence of
            names (tuple of str) to the indices they label. Each value may be an
            int, a list of ints, or a VectorFunction whose ``input_domain``
            encodes the indices. For use with :meth:`make_input` /
            :meth:`get_vars`.
        """

        mlist = inspect.getmembers(_tychopy.optimal_control)

        hasUvars = Uvars != None and Uvars != 0
        hasPvars = Pvars != None and Pvars != 0
        name = "ode_" + str(Xvars)

        if hasUvars:
            name += "_" + str(Uvars)
        if hasPvars:
            if not hasUvars:
                name += "_0"
            name += "_" + str(Pvars)

        constsize = False
        for m in mlist:
            if name == m[0]:
                self.ode = m[1].ode(odefunc)
                constsize = True
                break
        if constsize == False:
            if hasUvars and hasPvars:
                self.ode = _tychopy.optimal_control.ode_x_u_p.ode(
                    odefunc, Xvars, Uvars, Pvars
                )
            elif hasUvars:
                self.ode = _tychopy.optimal_control.ode_x_u.ode(odefunc, Xvars, Uvars)
            elif hasPvars:
                self.ode = _tychopy.optimal_control.ode_x_u_p.ode(
                    odefunc, Xvars, 0, Pvars
                )
            else:
                self.ode = _tychopy.optimal_control.ode_x.ode(odefunc, Xvars)

        if Vgroups != None:
            self.add_Vgroups(Vgroups)

    def _make_index_set(self, idxs):

        if (
            hasattr(idxs, "input_domain")
            and hasattr(idxs, "vf")
            and hasattr(idxs, "is_linear")
        ):
            ## Handle Vector Functions
            input_domain = idxs.input_domain()
            idxstmp = []
            for i in range(0, input_domain.shape[1]):
                start = input_domain[0, i]
                size = input_domain[1, i]
                idxstmp += list(range(start, start + size))
            return idxstmp
        elif isinstance(idxs, (int, np.int32, np.intc)):
            return [idxs]
        elif hasattr(idxs, "__iter__") and not isinstance(idxs, str):
            if len(idxs) == 0:
                raise Exception("Index list is empty")
            idxtmp = []
            for idx in idxs:
                idxtmp += self._make_index_set(idx)
            return idxtmp
        else:
            raise Exception("Invalid index: {}".format(str(idxs)))

    def add_Vgroups(self, Vgroups):
        for name in Vgroups:
            idxs = self._make_index_set(Vgroups[name])
            if isinstance(name, str):
                self.ode.add_idx(name, idxs)
            elif hasattr(name, "__iter__"):
                for n in name:
                    self.ode.add_idx(n, idxs)

    def make_units(self, **kwargs):
        """
        Makes a vector of non-dim units from variable group names and unit values

        Parameters
        ----------
        **kwargs : float,list,ndarray
            Keyword list of the names of the variable groups and corresponding units.
        Returns
        -------
        units : np.ndarray
            Vector of units.
        """
        units = np.ones((self.xtup_vars()))

        for key, value in kwargs.items():
            idx = self.idx(key)

            if isinstance(value, numbers.Number):
                units_t = np.ones((len(idx))) * value
            elif hasattr(value, "__iter__") and not isinstance(value, str):
                units_t = value
            else:
                raise Exception("Invalid unit: {}".format(str(value)))

            for i in range(0, len(idx)):
                units[idx[i]] = units_t[i]

        return units

    def make_input(self, **kwargs):
        """
        Makes an ODE input vector. Assigns sub-components to variable group names
        and assembles them into  single vector with proper ordering

        Parameters
        ----------
        **kwargs : float,list,ndarray
            Keyword list of the sub-vectors of the full input.

        Returns
        -------
        state : np.ndarray
            ODE input vector.

        """
        state = np.zeros((self.xtup_vars()))
        for key, value in kwargs.items():
            idx = self.idx(key)

            if isinstance(value, numbers.Number):
                state_t = np.ones((len(idx))) * value
            elif hasattr(value, "__iter__") and not isinstance(value, str):
                state_t = value
            else:
                raise Exception("Invalid unit: {}".format(str(value)))

            for i in range(0, len(idx)):
                state[idx[i]] = state_t[i]
        return state

    def get_vars(self, names, source, retscalar=False):
        """
        Retrieve variables from a ODE input vector or trajectory(2darray/listoflists)

        Parameters
        ----------
        names : str or list
            A single group name (str), or a list of group names (str) and/or
            integer indices, identifying the variables or sub-vectors to
            retrieve. A bare integer is not accepted; wrap it in a list.
        source : np.ndarray,list
            Source vector or 2d-array for retreiving variables.
        retscalar : bool, optional
            Return a scalar instead of vector if result has only one element.
            The default is False.

        Returns
        -------
        np.ndarray
            Concatenated array of the specified variables from source
        """
        #############################################################

        idxs = []
        if hasattr(names, "__iter__") and not isinstance(names, str):
            for name in names:
                if isinstance(name, str):
                    idxs += list(self.idx(name))
                elif isinstance(name, (int, np.int32, np.intc)):
                    idxs.append(name)
                else:
                    raise Exception("Invalid name specifier: {}".format(str(name)))
        elif isinstance(names, str):
            idxs += list(self.idx(names))
        else:
            raise Exception("Invalid name specifier: {}".format(str(names)))

        #############################################################

        if hasattr(source, "__iter__") and isinstance(source[0], numbers.Number):
            # Its a single vector
            output = np.zeros(len(idxs))
            for i, idx in enumerate(idxs):
                output[i] = source[idx]

            if len(output) == 1 and retscalar:
                return output[0]
            else:
                return output

        else:
            # Its a trajectory, nd array or list of lists/arrays
            size = len(source)
            output = np.zeros((size, len(idxs)))
            for j in range(0, size):
                for i, idx in enumerate(idxs):
                    output[j][i] = source[j][idx]

            return output

    def idx(self, Vname):
        return self.ode.idx(Vname)

    def phase(self, *args):
        """Create a collocation phase for these dynamics.

        Parameters
        ----------
        *args
            Forwarded to the underlying phase constructor. The common form is
            ``phase(transcription_mode, traj, num_segments)`` where
            ``transcription_mode`` is a :class:`TranscriptionModes` value (or its
            string name, e.g. ``"LGL3"``), ``traj`` is an initial-guess
            trajectory (a list of packed ``[x, t, u, p]`` node vectors), and
            ``num_segments`` is the number of collocation segments.

        Returns
        -------
        PhaseInterface
            A phase ready to be configured with boundary values, bounds,
            constraints, and objectives.

        Examples
        --------
        Create an LGL3 phase with 32 segments from an initial guess::

            phase = ode.phase("LGL3", Xs, 32)
        """
        return self.ode.phase(*args)

    def integrator(self, *args):
        """Create a numerical integrator for these dynamics.

        Parameters
        ----------
        *args
            Forwarded to the underlying integrator constructor. Common forms
            are ``integrator(dt)``, ``integrator(alg_name, dt)`` (e.g.
            ``"DOPRI87"``), and ``integrator(dt, control_func, indices)``.

        Returns
        -------
        Integrator
            An integrator for the ODE right-hand side.
        """
        return self.ode.integrator(*args)

    def vf(self):
        """Return the ODE right-hand side as a VectorFunction.

        Returns
        -------
        VectorFunction
            The dynamics function ``xdot = f(x, t, u, p)``.
        """
        return self.ode.vf()

    def x_vars(self):
        return self.ode.x_vars()

    def u_vars(self):
        return self.ode.u_vars()

    def p_vars(self):
        return self.ode.p_vars()

    def t_var(self):
        return self.ode.t_var()

    def xt_vars(self):
        return self.ode.xt_vars()

    def xtu_vars(self):
        return self.ode.xtu_vars()

    def xtup_vars(self):
        return self.ode.xtup_vars()

    def x_idxs(self, *args):
        if len(args) > 1:
            return self.ode.x_idxs(list(args))
        else:
            return self.ode.x_idxs(*args)

    def xt_idxs(self, *args):
        if len(args) > 1:
            return self.ode.xt_idxs(list(args))
        else:
            return self.ode.xt_idxs(*args)

    def xtu_idxs(self, *args):
        if len(args) > 1:
            return self.ode.xtu_idxs(list(args))
        else:
            return self.ode.xtu_idxs(*args)

    def u_idxs(self, *args):
        if len(args) > 1:
            return self.ode.u_idxs(list(args))
        else:
            return self.ode.u_idxs(*args)

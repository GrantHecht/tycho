// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "function_registry.h"
#include "tycho/detail/astro/kepler/lambert_solvers.h"

#include <fmt/format.h>

#include <stdexcept>
#include <string>

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
void lambert_solvers_build(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::lambert_solvers_build(FunctionRegistry &reg, nb::module_ &m) {

    // The scalar lambert_izzo overloads NaN-poison V1/V2 when the underlying
    // iteration fails to converge within maxiters=20.  Translate at the
    // binding boundary to RuntimeError, mirroring the propagate_cartesian
    // pattern in kepler_utils.cpp.  (The vectorized overload below returns
    // exitcodes per call, so it does not need translation.)
    auto check_finite_pair = [](const std::array<Vector3<double>, 2> &result, const char *name) {
        if (!result[0].allFinite() || !result[1].allFinite())
            throw std::runtime_error(std::string(name) +
                                     ": iteration did not converge within 20 iterations");
        return nb::make_tuple(result[0], result[1]);
    };

    m.def("lambert_izzo", [check_finite_pair](const Vector3<double> &R1, const Vector3<double> &R2,
                                              double dt, double mu, bool longway) {
        return check_finite_pair(lambert_izzo(R1, R2, dt, mu, longway), "lambert_izzo");
    },
          nb::arg("R1"), nb::arg("R2"), nb::arg("dt"), nb::arg("mu"), nb::arg("longway"),
          R"doc(Solve a Lambert boundary-value problem (zero-revolution, scalar).

Uses Izzo's algorithm to find the transfer orbit connecting ``R1`` to ``R2``
in time ``dt``.

Parameters
----------
R1 : array_like, shape (3,)
    Departure position vector.  Units consistent with ``mu``.
R2 : array_like, shape (3,)
    Arrival position vector.  Same units as ``R1``.
dt : float
    Transfer time-of-flight (must be > 0).
mu : float
    Gravitational parameter (must be > 0).
longway : bool
    ``True`` for the long-way transfer (transfer angle > 180°);
    ``False`` for the short-way transfer.

Returns
-------
tuple[ndarray shape (3,), ndarray shape (3,)]
    ``(V1, V2)`` — departure and arrival velocity vectors.

Raises
------
RuntimeError
    If the iteration fails to converge within 20 iterations.

See Also
--------
lambert_izzo_multirev : Multi-revolution overload.
)doc");

    m.def("lambert_izzo",
          [check_finite_pair](const Vector3<double> &R1, const Vector3<double> &R2, double dt,
                              double mu, bool longway, int Nrevs, bool rightbranch) {
              return check_finite_pair(lambert_izzo(R1, R2, dt, mu, longway, Nrevs, rightbranch),
                                       "lambert_izzo");
          },
          nb::arg("R1"), nb::arg("R2"), nb::arg("dt"), nb::arg("mu"), nb::arg("longway"),
          nb::arg("Nrevs"), nb::arg("rightbranch"),
          R"doc(Solve a Lambert boundary-value problem (multi-revolution, scalar).

Overload that additionally accepts a revolution count and branch selector.
For zero revolutions (``Nrevs=0``) this is equivalent to the 5-argument
overload.

Parameters
----------
R1 : array_like, shape (3,)
    Departure position vector.
R2 : array_like, shape (3,)
    Arrival position vector.
dt : float
    Transfer time-of-flight (must be > 0).
mu : float
    Gravitational parameter (must be > 0).
longway : bool
    ``True`` for the long-way (>180°) transfer.
Nrevs : int
    Number of complete revolutions (0 = direct transfer).
rightbranch : bool
    For ``Nrevs > 0``: ``True`` selects the right branch of the
    multi-revolution solution family, ``False`` selects the left branch.

Returns
-------
tuple[ndarray shape (3,), ndarray shape (3,)]
    ``(V1, V2)`` — departure and arrival velocity vectors.

Raises
------
RuntimeError
    If the iteration fails to converge within 20 iterations.
)doc");

    m.def("lambert_izzo_multirev",
          [check_finite_pair](const Vector3<double> &R1, const Vector3<double> &R2, double dt,
                              double mu, bool longway, int Nrevs, bool rightbranch) {
              return check_finite_pair(lambert_izzo(R1, R2, dt, mu, longway, Nrevs, rightbranch),
                                       "lambert_izzo_multirev");
          },
          nb::arg("R1"), nb::arg("R2"), nb::arg("dt"), nb::arg("mu"), nb::arg("longway"),
          nb::arg("Nrevs"), nb::arg("rightbranch"),
          R"doc(Solve a multi-revolution Lambert problem (alias for ``lambert_izzo`` 7-arg overload).

Equivalent to ``lambert_izzo(R1, R2, dt, mu, longway, Nrevs, rightbranch)``.
Provided as a named alias to make call sites self-documenting.

Parameters
----------
R1 : array_like, shape (3,)
    Departure position vector.
R2 : array_like, shape (3,)
    Arrival position vector.
dt : float
    Transfer time-of-flight (must be > 0).
mu : float
    Gravitational parameter (must be > 0).
longway : bool
    ``True`` for the long-way (>180°) transfer.
Nrevs : int
    Number of complete revolutions (must be > 0 for a multi-rev solution).
rightbranch : bool
    ``True`` selects the right branch; ``False`` selects the left branch.

Returns
-------
tuple[ndarray shape (3,), ndarray shape (3,)]
    ``(V1, V2)`` — departure and arrival velocity vectors.

Raises
------
RuntimeError
    If the iteration fails to converge within 20 iterations.
)doc");

    using NumpyMat = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

    m.def("lambert_izzo",
          [](ConstEigenRef<NumpyMat> R1s, ConstEigenRef<NumpyMat> R2s,
             ConstEigenRef<VectorX<double>> dts, double mu, const std::vector<bool> &longways,
             EigenRef<NumpyMat> V1s, EigenRef<NumpyMat> V2s, int axis, bool vectorize) {
              using SuperScalar = Eigen::Array<double, 8, 1>;
              constexpr int vsize = SuperScalar::SizeAtCompileTime;

              // Newton-iteration non-convergence exit code, reused below to
              // also flag the "NaN-with-success" collinear-geometry case
              // (see the warning on lambert_izzo_impl in lambert_solvers.h):
              // exint==0 there means only "Newton converged", it does not
              // guarantee V1/V2 are finite.
              constexpr int kNonConvergenceCode = 1;

              if (axis != 0 && axis != 1) {
                  throw std::invalid_argument(
                      fmt::format("lambert_izzo: axis must be 0 or 1, got {}", axis));
              }
              const int NumCalls = (axis == 0) ? int(R1s.cols()) : int(R1s.rows());
              auto check_3xN = [&](const auto &M, const char *name) {
                  const bool ok = (axis == 0) ? (M.rows() == 3 && M.cols() == NumCalls)
                                              : (M.cols() == 3 && M.rows() == NumCalls);
                  if (!ok) {
                      throw std::invalid_argument(fmt::format(
                          "lambert_izzo: {} has shape ({}, {}); expected {} for axis={} with {} "
                          "problems",
                          name, M.rows(), M.cols(), axis == 0 ? "(3, N)" : "(N, 3)", axis,
                          NumCalls));
                  }
              };
              check_3xN(R1s, "R1s");
              check_3xN(R2s, "R2s");
              check_3xN(V1s, "V1s");
              check_3xN(V2s, "V2s");
              if (dts.size() != NumCalls) {
                  throw std::invalid_argument(
                      fmt::format("lambert_izzo: dts has {} entries but R1s implies {} problems",
                                  dts.size(), NumCalls));
              }
              if (int(longways.size()) != NumCalls) {
                  throw std::invalid_argument(fmt::format(
                      "lambert_izzo: longways has {} entries but R1s implies {} problems",
                      longways.size(), NumCalls));
              }

              int Packs = vectorize ? NumCalls / vsize : 0;

              Eigen::VectorXi exitcodes(NumCalls);
              Vector3<SuperScalar> R1ss;
              Vector3<SuperScalar> R2ss;
              std::array<bool, vsize> lwss;
              std::array<int, vsize> ecodess;

              Vector3<SuperScalar> V1ss;
              Vector3<SuperScalar> V2ss;

              for (int i = 0; i < Packs; i++) {
                  int V = i * vsize;
                  if (axis == 0) {
                      R1ss[0] = R1s.row(0).segment<vsize>(V, vsize).transpose();
                      R1ss[1] = R1s.row(1).segment<vsize>(V, vsize).transpose();
                      R1ss[2] = R1s.row(2).segment<vsize>(V, vsize).transpose();

                      R2ss[0] = R2s.row(0).segment<vsize>(V, vsize).transpose();
                      R2ss[1] = R2s.row(1).segment<vsize>(V, vsize).transpose();
                      R2ss[2] = R2s.row(2).segment<vsize>(V, vsize).transpose();
                  } else {

                      R1ss[0] = R1s.col(0).segment<vsize>(V, vsize);
                      R1ss[1] = R1s.col(1).segment<vsize>(V, vsize);
                      R1ss[2] = R1s.col(2).segment<vsize>(V, vsize);

                      R2ss[0] = R2s.col(0).segment<vsize>(V, vsize);
                      R2ss[1] = R2s.col(1).segment<vsize>(V, vsize);
                      R2ss[2] = R2s.col(2).segment<vsize>(V, vsize);
                  }

                  SuperScalar dtss = dts.segment<vsize>(V);
                  for (int j = 0; j < vsize; j++) {
                      lwss[j] = longways[V + j];
                  }

                  lambert_izzo_impl(R1ss, R2ss, dtss, mu, lwss, 0, false, V1ss, V2ss, ecodess);

                  if (axis == 0) {
                      V1s.row(0).segment<vsize>(V, vsize) = V1ss[0].transpose();
                      V1s.row(1).segment<vsize>(V, vsize) = V1ss[1].transpose();
                      V1s.row(2).segment<vsize>(V, vsize) = V1ss[2].transpose();

                      V2s.row(0).segment<vsize>(V, vsize) = V2ss[0].transpose();
                      V2s.row(1).segment<vsize>(V, vsize) = V2ss[1].transpose();
                      V2s.row(2).segment<vsize>(V, vsize) = V2ss[2].transpose();
                  } else {
                      V1s.col(0).segment<vsize>(V, vsize) = V1ss[0];
                      V1s.col(1).segment<vsize>(V, vsize) = V1ss[1];
                      V1s.col(2).segment<vsize>(V, vsize) = V1ss[2];

                      V2s.col(0).segment<vsize>(V, vsize) = V2ss[0];
                      V2s.col(1).segment<vsize>(V, vsize) = V2ss[1];
                      V2s.col(2).segment<vsize>(V, vsize) = V2ss[2];
                  }

                  for (int j = 0; j < vsize; j++) {
                      exitcodes[V + j] = ecodess[j];
                  }

                  // NaN-poison guard: collinear-geometry inputs can leave
                  // exint==0 (Newton "converged") while V1/V2 are NaN (see
                  // the lambert_izzo_impl warning). Check the just-written
                  // V1s/V2s slice per lane, respecting the axis layout, and
                  // promote the exit code so callers who "trust the exit
                  // codes" actually can.
                  for (int j = 0; j < vsize; j++) {
                      int idx = V + j;
                      bool finite = (axis == 0) ? (V1s.col(idx).allFinite() &&
                                                   V2s.col(idx).allFinite())
                                                : (V1s.row(idx).allFinite() &&
                                                   V2s.row(idx).allFinite());
                      if (!finite && exitcodes[idx] == 0) {
                          exitcodes[idx] = kNonConvergenceCode;
                      }
                  }
              }

              Vector3<double> R1;
              Vector3<double> R2;
              Vector3<double> V1;
              Vector3<double> V2;

              for (int i = Packs * vsize; i < NumCalls; i++) {

                  if (axis == 0) {
                      R1 = R1s.col(i);
                      R2 = R2s.col(i);
                  } else {
                      R1 = R1s.row(i);
                      R2 = R2s.row(i);
                  }

                  double dt = dts[i];
                  bool longway = longways[i];
                  int excode;

                  lambert_izzo_impl(R1, R2, dt, mu, longway, 0, false, V1, V2, excode);

                  exitcodes[i] = excode;

                  // Same NaN-poison guard as the packed loop above: collinear
                  // geometry can leave excode==0 with non-finite V1/V2.
                  if (exitcodes[i] == 0 && (!V1.allFinite() || !V2.allFinite())) {
                      exitcodes[i] = kNonConvergenceCode;
                  }

                  if (axis == 0) {
                      V1s.col(i) = V1;
                      V2s.col(i) = V2;
                  } else {
                      V1s.row(i) = V1.transpose();
                      V2s.row(i) = V2.transpose();
                  }
              }

              return exitcodes;
          },
          nb::call_guard<nb::gil_scoped_release>(),
          R"doc(Solve a batch of Lambert problems in-place (vectorized overload).

Solves ``N`` independent Lambert problems simultaneously, writing results
directly into pre-allocated output matrices.  Uses 8-wide SuperScalar SIMD
packing when ``vectorize=True`` for throughput-critical applications.

Parameters
----------
R1s : ndarray, shape (3, N) or (N, 3)
    Departure position vectors.  Layout selected by ``axis``.
R2s : ndarray, shape (3, N) or (N, 3)
    Arrival position vectors.  Same layout as ``R1s``.
dts : ndarray, shape (N,)
    Transfer times for each problem.
mu : float
    Gravitational parameter (scalar, same for all problems).
longways : list of bool, length N
    Long-way flag for each problem.
V1s : ndarray, shape (3, N) or (N, 3), writeable
    Output buffer for departure velocities.  Must be pre-allocated.
V2s : ndarray, shape (3, N) or (N, 3), writeable
    Output buffer for arrival velocities.  Must be pre-allocated.
axis : int
    ``0`` if vectors are stored as columns (shape ``(3, N)``);
    ``1`` if vectors are stored as rows (shape ``(N, 3)``).
vectorize : bool
    If ``True``, processes problems in packs of 8 using SuperScalar SIMD
    for higher throughput.  The remainder (``N mod 8``) is handled
    scalar-wise.

Returns
-------
ndarray of int, shape (N,)
    Per-problem exit codes: ``0`` = converged, ``1`` = not converged
    within 20 iterations, or collinear-geometry input produced a
    non-finite (NaN) result despite the Newton iteration itself
    reporting convergence.

Raises
------
ValueError
    If ``axis`` is not ``0`` or ``1``, or if ``R1s``/``R2s``/``V1s``/``V2s``
    do not have the shape implied by ``axis`` and the problem count, or if
    ``dts``/``longways`` do not have length ``N``.

Notes
-----
Unlike the scalar overloads this function does **not** raise on
non-convergence or collinear-geometry NaN results; the exit-code vector
now flags both, so it can be trusted without a separate ``allFinite()``
check on ``V1s``/``V2s``.  The GIL is released for the duration of the
call.
)doc");
}

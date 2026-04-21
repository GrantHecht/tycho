// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"
#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/vf/type_erasure/generic_function.h"

namespace tycho::integrators {

using oc::InterpFunction;
using oc::LGLInterpTable;
using vf::GenericFunction;

/// Named direction constants for EventPack::direction. Left as plain int to
/// preserve brace-init call-site syntax (`{gf, 0, 1}`) across C++ callers and
/// to let the nanobind tuple-based Python API pass through unchanged.
namespace event_direction {
inline constexpr int Falling = -1;
inline constexpr int Any = 0;
inline constexpr int Rising = 1;
} // namespace event_direction

/// Event spec: an event function, a direction filter, and a stop count.
///
/// `direction`: -1 (falling only), 0 (any crossing), +1 (rising only). See
///    `event_direction::Falling/Any/Rising`.
/// `stop_count`: 0 means "record crossings, never stop"; N > 0 means "stop
///    integration after the Nth recorded crossing".
struct EventPack {
    GenericFunction<-1, 1> vf;
    int direction = 0;
    int stop_count = 0;

    EventPack() = default;
    EventPack(GenericFunction<-1, 1> vf_, int direction_, int stop_count_)
        : vf(std::move(vf_)), direction(direction_), stop_count(stop_count_) {}
};

struct EventHandler {

    using EventPack = tycho::integrators::EventPack;

    /// Check for zero crossings between prev_vals and next_vals.
    /// Returns true if an event triggered a stop condition.
    template <class XState>
    static bool check_crossings(const std::vector<EventPack> &events,
                                const std::vector<Vector1<double>> &prev_vals,
                                std::vector<Vector1<double>> &next_vals, const XState &xnext,
                                int t_var, std::vector<std::vector<Eigen::Vector2d>> &eventtimes,
                                double t_prev) {
        bool eventbreak = false;
        for (int j = 0; j < static_cast<int>(events.size()); j++) {
            next_vals[j].setZero();
            events[j].vf.compute(xnext, next_vals[j]);

            double vprev = prev_vals[j][0];
            double vnext = next_vals[j][0];
            // Event VF output must be finite on finite states; otherwise
            // the sign comparison below silently drops the crossing under
            // IEEE 754 (NaN < 0 is false). Surface the failure immediately
            // with t/j context so the user can fix the event function.
            if (!std::isfinite(vnext) || !std::isfinite(vprev)) {
                throw std::runtime_error(
                    "Event function " + std::to_string(j) + " produced non-finite value at t=" +
                    std::to_string(static_cast<double>(xnext[t_var])) +
                    " (vprev=" + std::to_string(vprev) + ", vnext=" + std::to_string(vnext) +
                    "); event functions must be finite on finite states.");
            }
            int dir = events[j].direction;
            double vprod = vprev * vnext;

            if (vprod < 0.0) {
                if ((dir > 0 && vnext > 0) || (dir < 0 && vnext < 0) || dir == 0) {
                    Eigen::Vector2d times;
                    times[0] = t_prev;
                    times[1] = xnext[t_var];
                    eventtimes[j].push_back(times);
                    int stop = events[j].stop_count;

                    if (stop != 0) {
                        if (static_cast<int>(eventtimes[j].size()) == stop) {
                            eventbreak = true;
                        }
                    }
                }
            }
        }
        return eventbreak;
    }

    /// Refine event locations using bisection + Newton on an interpolation table.
    ///
    /// Returns a vector whose shape matches `eventtimes` exactly: for every
    /// crossing in `eventtimes[i]` there is a corresponding entry in
    /// `eventstates[i]`, either `std::optional<ODEState>` holding the refined
    /// state or `std::nullopt` if neither the fast bisect+Newton pass nor the
    /// wider-bracket retry could resolve the crossing. 1:1 alignment makes
    /// zip-iterating `eventtimes` against `eventstates` safe.
    ///
    /// `n_failed_refinements` is incremented once per unresolved crossing
    /// (i.e., per `std::nullopt` produced).
    template <class ODEState>
    static std::vector<std::vector<std::optional<ODEState>>>
    refine_events(std::shared_ptr<LGLInterpTable> tab, const std::vector<EventPack> &events,
                  const std::vector<std::vector<Eigen::Vector2d>> &eventtimes, int input_rows,
                  int max_iters, double tol, int &n_failed_refinements) {

        Eigen::VectorXi vars;
        vars.setLinSpaced(input_rows, 0, input_rows - 1);
        InterpFunction<-1> tabfunc(tab, vars);

        Eigen::Matrix<double, 1, 1> x;
        Eigen::Matrix<double, 1, 1> fx;
        Eigen::Matrix<double, 1, 1> fl;
        Eigen::Matrix<double, 1, 1> jx;

        std::vector<std::vector<std::optional<ODEState>>> eventstates(events.size());

        for (int i = 0; i < static_cast<int>(events.size()); i++) {
            if (eventtimes[i].size() > 0) {
                auto func = events[i].vf.eval(tabfunc);

                auto newton = [&](auto x0) {
                    x[0] = x0;
                    for (int k = 0; k < max_iters; k++) {
                        fx.setZero();
                        jx.setZero();
                        func.compute_jacobian(x, fx, jx);
                        if (std::abs(fx[0]) < std::abs(tol)) {
                            break;
                        }
                        // Guard against singular / non-finite Jacobian. Without
                        // NaN-ing x[0] here, break leaves x[0] at its pre-singular
                        // value — which can land inside the bracket and be
                        // indistinguishable from convergence at the bracket check
                        // below. Setting x[0] = NaN forces the bracket comparison
                        // (NaN > tlow, NaN < thigh both false under IEEE 754) into
                        // the wider-bracket retry deterministically, preserving
                        // the failure-counter contract.
                        if (!std::isfinite(jx[0]) || jx[0] == 0.0) {
                            x[0] = std::numeric_limits<double>::quiet_NaN();
                            break;
                        }
                        x[0] = x[0] - fx[0] / jx[0];
                    }
                    return x[0];
                };

                auto bisect = [&](auto tlow, auto thigh, int iters) {
                    double tm = (tlow + thigh) / 2.0;
                    x[0] = tlow;
                    fl = func.compute(x);
                    int sgnfl = (fl[0] >= 0) - (fl[0] <= 0);

                    x[0] = tm;
                    fx = func.compute(x);
                    int sgnfx = (fx[0] >= 0) - (fx[0] <= 0);

                    for (int bi = 0; bi < iters; bi++) {
                        if (sgnfx == sgnfl) {
                            tlow = tm;
                            sgnfl = sgnfx;
                        } else {
                            thigh = tm;
                        }

                        tm = (tlow + thigh) / 2.0;
                        if ((thigh - tlow) / 2.0 < std::abs(tol))
                            break;

                        x[0] = tm;
                        fx = func.compute(x);
                        sgnfx = (fx[0] >= 0) - (fx[0] <= 0);
                    }

                    return std::array<double, 3>{tm, tlow, thigh};
                };

                for (auto &eventtime : eventtimes[i]) {
                    double tlow = eventtime[0];
                    double thigh = eventtime[1];

                    if (thigh < tlow) {
                        std::swap(tlow, thigh);
                    }

                    auto res = bisect(tlow, thigh, 2);
                    double tig = res[0];
                    double tlow2 = res[1];
                    double thigh2 = res[2];

                    double tevent = newton(tig);

                    if (tevent > tlow && tevent < thigh) {
                        ODEState ei(input_rows);
                        ei.setZero();
                        tab->interpolate_ref(tevent, ei);
                        eventstates[i].emplace_back(std::move(ei));
                    } else {
                        res = bisect(tlow2, thigh2, max_iters);
                        tig = res[0];
                        tevent = newton(tig);

                        if (tevent > tlow && tevent < thigh) {
                            ODEState ei(input_rows);
                            ei.setZero();
                            tab->interpolate_ref(tevent, ei);
                            eventstates[i].emplace_back(std::move(ei));
                        } else {
                            // Neither the fast nor wide-bracket bisect+Newton
                            // retry could resolve this crossing. The slot is
                            // filled with std::nullopt to preserve 1:1
                            // alignment with eventtimes; the counter also
                            // increments for callers that summarise loss.
                            eventstates[i].emplace_back(std::nullopt);
                            ++n_failed_refinements;
                        }
                    }
                }
            }
        }

        return eventstates;
    }
};

} // namespace tycho::integrators

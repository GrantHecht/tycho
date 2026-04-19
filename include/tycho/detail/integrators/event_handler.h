// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <tuple>
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

struct EventHandler {

    using EventPack = std::tuple<GenericFunction<-1, 1>, int, int>;

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
            std::get<0>(events[j]).compute(xnext, next_vals[j]);

            double vprev = prev_vals[j][0];
            double vnext = next_vals[j][0];
            int dir = std::get<1>(events[j]);
            double vprod = vprev * vnext;

            if (vprod < 0.0) {
                if ((dir > 0 && vnext > 0) || (dir < 0 && vnext < 0) || dir == 0) {
                    Eigen::Vector2d times;
                    times[0] = t_prev;
                    times[1] = xnext[t_var];
                    eventtimes[j].push_back(times);
                    int stop = std::get<2>(events[j]);

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
    template <class ODEState>
    static std::vector<std::vector<ODEState>>
    refine_events(std::shared_ptr<LGLInterpTable> tab, const std::vector<EventPack> &events,
                  const std::vector<std::vector<Eigen::Vector2d>> &eventtimes, int input_rows,
                  int max_iters, double tol) {

        Eigen::VectorXi vars;
        vars.setLinSpaced(input_rows, 0, input_rows - 1);
        InterpFunction<-1> tabfunc(tab, vars);

        Eigen::Matrix<double, 1, 1> x;
        Eigen::Matrix<double, 1, 1> fx;
        Eigen::Matrix<double, 1, 1> fl;
        Eigen::Matrix<double, 1, 1> jx;

        std::vector<std::vector<ODEState>> eventstates(events.size());

        for (int i = 0; i < static_cast<int>(events.size()); i++) {
            if (eventtimes[i].size() > 0) {
                auto func = std::get<0>(events[i]).eval(tabfunc);

                auto newton = [&](auto x0) {
                    x[0] = x0;
                    for (int k = 0; k < max_iters; k++) {
                        fx.setZero();
                        jx.setZero();
                        func.compute_jacobian(x, fx, jx);
                        if (std::abs(fx[0]) < std::abs(tol)) {
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
                        eventstates[i].push_back(ei);
                    } else {
                        res = bisect(tlow2, thigh2, max_iters);
                        tig = res[0];
                        tevent = newton(tig);

                        if (tevent > tlow && tevent < thigh) {
                            ODEState ei(input_rows);
                            ei.setZero();
                            tab->interpolate_ref(tevent, ei);
                            eventstates[i].push_back(ei);
                        } // else give up
                    }
                }
            }
        }

        return eventstates;
    }
};

} // namespace tycho::integrators

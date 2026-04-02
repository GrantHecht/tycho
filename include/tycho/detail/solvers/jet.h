// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
//   - Thread pool replaced with global tycho::utils::thread_pool() singleton
//   - Removed `nt` parameter: parallelism is controlled by tycho::utils::set_num_threads()
// =============================================================================

#pragma once

#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include "tycho/detail/solvers/optimization_problem_base.h"
#include "tycho/detail/utils/thread_pool.h"
#include "tycho/detail/utils/timer.h"
#ifdef USE_ACCELERATE_SPARSE
#include "tycho/detail/solvers/linear/accelerate_utils.h"
#else
#include "mkl.h"
#endif

namespace tycho::solvers {

namespace detail {

template <class T, class GenFunc, class Args> struct JetInvoker {
    static std::shared_ptr<T> invoke(const GenFunc &gf, const Args &args) { return gf(args); }
};
} // namespace detail

struct Jet {

    static void print_beginning() {
        constexpr const char *jestr = "          __  ______  ______\n"
                                      "         / / / ____/ /_  __/\n"
                                      "    __  / / / __/     / /   \n"
                                      "   / /_/ / / /___    / /    \n"
                                      "   \\____/ /_____/   /_/     \n\n";

        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
        fmt::print(fmt::fg(fmt::color::crimson), jestr);
        fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Jet");
        fmt::print("\n");
    }

    static void print_progress(int i, double tsec, int NumJobs, int NumConv, int NumAcc,
                               int NumNoConv, int NumDiv) {
        double prog = 100 * double(i + 1) / double(NumJobs);
        int len = 76 * double(i + 1) / double(NumJobs);
        int wspace = 76 - len;
        double remtime = (tsec / double(i + 1)) * (NumJobs - i - 1);
        auto cyan = fmt::fg(fmt::color::cyan);
        auto green = fmt::fg(fmt::color::green);

        auto sminhrs = [cyan](double ts) {
            if (ts < 60.0) {
                fmt::print(cyan, "{0:>10.2f} s     \n", ts);
            } else if (ts < 3600.0) {
                fmt::print(cyan, "{0:>10.2f} min     \n", ts / 60.0);
            } else {
                fmt::print(cyan, "{0:>10.2f} hr     \n", ts / 3600.0);
            }
        };

        fmt::print("\n");

        fmt::print(" Remaining Time : ");
        sminhrs(remtime);
        fmt::print(" Elapsed Time   : ");
        sminhrs(tsec);
        fmt::print(" Progress       : ");
        fmt::print(cyan, "{0:>10.2f} %  \n", prog);
        fmt::print(" [");
        fmt::print(green, "{0:#^{1}}{0:.^{2}}", "", len, wspace);
        fmt::print("]");
        fmt::print("\n\n");
        fmt::print("  Completed        : ");
        fmt::print(cyan, "{0:>10}/{1:<10}   \n", (i + 1), NumJobs);

        fmt::print("    Optimal        : ");
        fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumConv, NumJobs);
        fmt::print("    Acceptable     : ");
        fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumAcc, NumJobs);
        fmt::print("    Not Converged  : ");
        fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumNoConv, NumJobs);
        fmt::print("    Diverged       : ");
        fmt::print(cyan, "{0:>10}/{1:<10}   \n", NumDiv, NumJobs);

        if (i < (NumJobs - 1))
            fmt::print("\033[11F");
        else
            fmt::print("\n");
    };

    static void print_finished() {

        fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Jet");
        fmt::print("\n");
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////// Map///////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////
    template <class T, class Args1, class Args2>
    static std::vector<std::shared_ptr<T>>
    map(const std::vector<std::function<std::shared_ptr<T>(Args1)>> &genfuncs,
        const std::vector<Args2> &args, const Eigen::VectorXi &genfidxes, bool verbose) {

#ifdef USE_ACCELERATE_SPARSE
        accelerate_set_num_threads(1);
#endif

        int NumJobs = args.size();
        int NumConv = 0;
        int NumAcc = 0;
        int NumNoConv = 0;
        int NumDiv = 0;

        std::vector<std::shared_ptr<T>> optprobs(NumJobs);
        tycho::utils::Timer t;

        auto Job = [&](int i) {
#ifdef USE_ACCELERATE_SPARSE
            // Per-thread single-threaded mode (uses BLASSetThreading on
            // macOS 15+, env var fallback on older systems)
            accelerate_set_num_threads(1);
#else
            mkl_set_num_threads_local(1);
#endif

            int gfidx = genfidxes[i];
            optprobs[i] =
                detail::JetInvoker<T, std::decay_t<decltype(genfuncs[gfidx])>, Args2>::invoke(
                    genfuncs[gfidx], args[i]);

            return optprobs[i]->jet_run();
        };

        if (verbose)
            print_beginning();
        t.start();

        auto track = [&](tycho::ConvergenceFlags flag, int i) {
            if (!verbose)
                return;
            switch (flag) {
            case tycho::ConvergenceFlags::CONVERGED:
                NumConv++;
                break;
            case tycho::ConvergenceFlags::ACCEPTABLE:
                NumAcc++;
                break;
            case tycho::ConvergenceFlags::NOTCONVERGED:
                NumNoConv++;
                break;
            case tycho::ConvergenceFlags::DIVERGING:
                NumDiv++;
                break;
            }
            double tsec = double(t.count<std::chrono::microseconds>()) / 1000000.0;
            print_progress(i, tsec, NumJobs, NumConv, NumAcc, NumNoConv, NumDiv);
        };

        if (tycho::utils::use_thread_pool()) {
            std::vector<std::future<tycho::ConvergenceFlags>> results;
            results.reserve(NumJobs);
            for (int i = 0; i < NumJobs; i++)
                results.push_back(
                    tycho::utils::thread_pool().submit_task([&Job, i] { return Job(i); }));
            // track() mutates local counters — must be called sequentially.
            // If .get() throws, drain remaining futures before rethrowing to
            // prevent use-after-free of stack-captured references (&Job, etc.).
            std::exception_ptr ex;
            for (int i = 0; i < NumJobs; i++) {
                try {
                    track(results[i].get(), i);
                } catch (...) {
                    if (!ex)
                        ex = std::current_exception();
                    int suppressed = 0;
                    for (int j = i + 1; j < NumJobs; j++) {
                        try {
                            results[j].get();
                        } catch (const std::exception &e) {
                            if (suppressed == 0)
                                fmt::print(stderr,
                                           "[Tycho] Jet::map: additional job also failed: {}\n",
                                           e.what());
                            ++suppressed;
                        } catch (...) {
                            ++suppressed;
                        }
                    }
                    if (suppressed > 1)
                        fmt::print(stderr,
                                   "[Tycho] Jet::map: {} additional exceptions suppressed\n",
                                   suppressed - 1);
                    break;
                }
            }
            if (ex)
                std::rethrow_exception(ex);
        } else {
            for (int i = 0; i < NumJobs; i++)
                track(Job(i), i);
        }
        if (verbose)
            print_finished();
        return optprobs;
    }

    template <class T, class Args1, class Args2>
    static std::vector<std::shared_ptr<T>> map(std::function<std::shared_ptr<T>(Args1)> genfunc,
                                               const std::vector<Args2> &args, bool verbose) {

        std::vector<std::function<std::shared_ptr<T>(Args1)>> genfuncs;
        genfuncs.push_back(genfunc);
        Eigen::VectorXi genfidxes(args.size());

        genfidxes.setConstant(0);

        return Jet::map(genfuncs, args, genfidxes, verbose);
    }

    template <class T>
    static std::vector<std::shared_ptr<T>> map(const std::vector<std::shared_ptr<T>> &optprobs,
                                               bool verbose) {

        std::function<std::shared_ptr<T>(std::shared_ptr<T>)> genfunc =
            [](std::shared_ptr<T> optprob) { return optprob; };

        return Jet::map(genfunc, optprobs, verbose);
    }
    ////////////////////////////////////////////////////////////////////////////////////
};

} // namespace tycho::solvers

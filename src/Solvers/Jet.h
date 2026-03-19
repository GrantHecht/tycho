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
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 / pybind11 header references removed
//   - Thread pool replaced with global Tycho::thread_pool() singleton
//   - Removed `nt` parameter: parallelism is controlled by Tycho::set_num_threads()
// =============================================================================

#pragma once
#include "OptimizationProblemBase.h"
#include "Utils/ThreadPool.h"
#include "Utils/Timer.h"
#include "Utils/fmtlib.h"
#ifdef USE_ACCELERATE_SPARSE
#include "AccelerateUtils.h"
#else
#include "mkl.h"
#endif

namespace Tycho {

namespace detail {

template <class T, class GenFunc, class Args> struct JetInvoker {
    static std::shared_ptr<T> invoke(const GenFunc &gf, const Args &args) { return gf(args); }
};
} // namespace detail

struct Jet {

    static void print_beginning() {
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
        fmt::print(fmt::fg(fmt::color::dim_gray), "Starting ");
        fmt::print(": ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "Jet");
        fmt::print("\n");
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
    }

    static void print_progress(int i, double t, int nj, int nc, int na, int nn, int nd) {
        auto jestr = fmt::format(fmt::runtime("({:>{}}/{})"), i + 1,
                                 static_cast<int>(std::to_string(nj).size()), nj);
        fmt::print(fmt::fg(fmt::color::dim_gray), "Job ");
        fmt::print(fmt::fg(fmt::color::royal_blue), "{}", jestr);
        fmt::print(fmt::fg(fmt::color::dim_gray), " Done");
        fmt::print(":");
        fmt::print(fmt::fg(fmt::color::green), " Conv:{}", nc);
        fmt::print(fmt::fg(fmt::color::yellow), " Acc:{}", na);
        fmt::print(fmt::fg(fmt::color::orange), " NConv:{}", nn);
        fmt::print(fmt::fg(fmt::color::red), " Div:{}", nd);
        fmt::print(fmt::fg(fmt::color::dim_gray), " Time:{:.4f} s", t);
        fmt::print("\n");
    }

    static void print_finished() {
        fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 79);
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
        Utils::Timer t;

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

        auto track = [&](PSIOPT::ConvergenceFlags flag, int i) {
            if (!verbose)
                return;
            switch (flag) {
            case PSIOPT::ConvergenceFlags::CONVERGED:
                NumConv++;
                break;
            case PSIOPT::ConvergenceFlags::ACCEPTABLE:
                NumAcc++;
                break;
            case PSIOPT::ConvergenceFlags::NOTCONVERGED:
                NumNoConv++;
                break;
            case PSIOPT::ConvergenceFlags::DIVERGING:
                NumDiv++;
                break;
            }
            double tsec = double(t.count<std::chrono::microseconds>()) / 1000000.0;
            print_progress(i, tsec, NumJobs, NumConv, NumAcc, NumNoConv, NumDiv);
        };

        if (Tycho::use_thread_pool()) {
            auto results = Tycho::thread_pool().submit_sequence(0, NumJobs, Job);
            for (int i = 0; i < NumJobs; i++)
                track(results[i].get(), i);
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

} // namespace Tycho

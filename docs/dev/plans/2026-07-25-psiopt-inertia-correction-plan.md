# PSIOPT Inertia Correction (IPOPT Algorithm IC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PSIOPT's classic inertia-correction ladder enforce the full IPOPT
condition — inertia exactly `(kkt_dim − m, m, 0)` — engaging the existing #103 dual
regularization on demand for singular factorizations, and abort the phase as
`SINGULAR_KKT` (after consulting the recovery chain) when correction exhausts.

**Architecture:** All solver logic lives in `PSIOPT::factor_impl` and its one call site
in `alg_impl` (src/solvers/psiopt.cpp). δ_c machinery (`perturb_kkt_c_diags`,
`dual_regularization(mu)`) is reused verbatim from PR #103. Exhaustion is signaled by a
new `bool &exhausted` out-param and routed by forcing the line-search verdict to
rejected so the existing `recovery_->on_step_rejected` chain gets its say; an
unresolved rejection aborts with a new `ConvergenceFlags::SINGULAR_KKT`.

**Tech Stack:** C++20, Eigen, gtest (leaf tests), nanobind (one enum value), the
standalone-probe pipeline from the PR #88 investigation for fast red/green.

**Spec:** `docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md` (rev 2).

## Global Constraints

- Branch: `fix/psiopt-inertia-correction` (already cut from `main` at `c9e8ddd`). All
  work on this Mac (16 GB Apple Silicon) — the defect only reproduces on
  macOS/Accelerate.
- `conda activate tycho` before any build or Python step.
- Builds: `cd /Users/granthec/Source/tycho/build && ninja -j4 all`. **ONE build at a
  time, ever. Never start a second ninja while one runs** — two concurrent builds OOM
  this machine. Launch long builds from the controller session (not from inside a
  subagent that will end its turn), in the background, and wait for completion.
- `psiopt.h`, `psiopt_fwd.h`, and `jet.h` are in the PCH include chain: touching them
  costs a full rebuild (~20-40 min at `-j4`). Task 1 and Task 2 each contain exactly
  one such rebuild; do not add extra ones.
- **Probe before building:** each code change is first verified through the standalone
  probe pipeline (recompile the 7 solver TUs + relink a probe, ~40 s total; commands
  given in the tasks). Only after the probe is green do you pay for the real build.
- **Never trust a `ctest` run against binaries older than your change** — stale test
  binaries reproduce stale behavior. Rebuild first, always.
- clangd diagnostics in this repo are unusable (it cannot parse `-fopenmp=libomp` and
  reports false missing-symbol errors). Ignore editor diagnostics; trust the compiler.
- Do not modify: the δ_w escalation constants/flow, `finalpert`/`Hpert0` warm-start
  accounting, `cumpert` display accounting, the proximal display/decay block
  (psiopt.cpp:2305-2319), anything in `notices/`.
- Format before each commit: `cd build && ninja clang-format`. Commit prefixes:
  `fix:` / `test:` / `docs:`.
- This PR touches PSIOPT optimizer internals and adds a Python-visible enum value —
  both flagged for explicit human review; the PR body must say so. Nothing merges
  without Grant's review plus the Linux CI corpus run.
- The expected macOS end state: `DivergencePersistence.MaratosCorpusConvergesAtDefaults`
  green, upgraded rank-deficient classic test green, new exhaustion test green,
  `GenuineDivergenceStillAborts` and all `InertiaRegularizationSolve` tests green,
  full ctest red ONLY on `cpp_example_optimal_docking_builder` (pre-existing,
  out of scope, documented in the spec).

---

### Task 1: Full IC condition + on-demand δ_c in `factor_impl`

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt.h:1005-1018` (doc comment + declaration)
- Modify: `src/solvers/psiopt.cpp:1639-1760` (`factor_impl` body)
- Modify: `src/solvers/psiopt.cpp:2279-2293` (call site: hoist `dual_shift`, thread
  `exhausted`)
- Test: `tests/cpp/solvers/test_inertia_regularization.cpp:130-166` (upgrade test (b))

**Interfaces:**
- Consumes: `NonLinearProgram::perturb_kkt_c_diags(double, Eigen::SparseMatrix<double,
  Eigen::RowMajor>&)`, `tycho::solvers::dual_regularization(double mu)` (both from
  #103, already in tree).
- Produces: `int factor_impl(bool docompute, bool ZFac, double ipurt, double incpurt0,
  double incpurt, double &finalpert, double &cumpert, double base_prox,
  double dual_shift, bool &exhausted)` — Task 2 consumes the `kkt_exhausted` local
  threaded at the call site.

- [ ] **Step 1: Upgrade the classic rank-deficient test to unconditional**

In `tests/cpp/solvers/test_inertia_regularization.cpp`, replace the whole test
`ClassicOnRankDeficientKktDocumented` (the `// (b) ...` comment block through its
closing brace, lines ~146-166) with:

```cpp
// (b) The SAME rank-deficient problem under classic. The full Ipopt IC condition
// engages the on-demand dual regularization when the factorization reports rank
// deficiency, so classic converges here too. (On MKL the static pivot
// perturbation may mask the deficiency instead; either road must reach the
// unique optimum.)
TEST(InertiaRegularizationSolve, ClassicConvergesOnRankDeficientKkt) {
    auto prob = build_duplicated_equality_nlp();
    prob->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    prob->optimizer_->set_max_iters(100);
    auto flag = prob->optimize();

    EXPECT_EQ(flag, tycho::ConvergenceFlags::CONVERGED);
    const auto &r = prob->optimizer_->result();
    EXPECT_NEAR(r.obj_val_, 0.5, 1e-5);
    ASSERT_EQ(r.primals_.size(), 2);
    EXPECT_NEAR(r.primals_[0], 0.5, 1e-4);
    EXPECT_NEAR(r.primals_[1], 0.5, 1e-4);
}
```

Do not run it yet (the binary is stale until Step 6's rebuild); the probe in Step 4 is
the fast red/green.

- [ ] **Step 2: Rewrite `factor_impl`'s predicates and δ_c engagement**

In `src/solvers/psiopt.cpp`, inside `factor_impl` (starts :1639):

**(a)** Immediately after the existing `Inertia` lambda (:1642-1644), add:

```cpp
    // Full Ipopt inertia-correction condition (Algorithm IC, Wächter & Biegler
    // 2006): accept only inertia exactly (kkt_dim - m, m, 0). Singular() is the
    // rank-deficiency part; Inertia() != 0 covers both excess (the only case the
    // pre-2026-07 ladder corrected) and missing negative eigenvalues.
    auto Singular = [&]() {
        return (this->kkt_sol_.neigs() + this->kkt_sol_.peigs() - this->kkt_dim_) != 0;
    };
```

**(b)** After the `Compute` lambda (:1679), add the function-scope δ_c machinery (the
`PerturbC` lambda MOVES here from inside the proximal branch — delete it there):

```cpp
    auto PerturbC = [&](double p) {
        this->nlp_->perturb_kkt_c_diags(p, this->kkt_sol_.get_matrix());
    };
    // On-demand dual regularization (delta_c). dual_shift is the available
    // magnitude for this call (0.0 while a nested l1 restoration phase owns the
    // constraint-row diagonals -- the caller suppresses it, see
    // inertia_regularization.h's nested-suppression rationale). The proximal
    // branch applies it up-front as part of the base matrix; the classic branch
    // applies it here, at most once per call, the first time a factorization
    // reports rank deficiency. It lands in the matrix and takes effect at the
    // next Refactor(), so a singular base costs one ladder rung -- a small
    // delta_w rides along with delta_c, matching Ipopt, which raises both on
    // singularity.
    bool dc_applied = false;
    auto EngageDualReg = [&]() {
        if (!dc_applied && dual_shift != 0.0) {
            PerturbC(-dual_shift);
            dc_applied = true;
        }
    };
```

**(c)** In the proximal branch: where `PerturbC(-dual_shift);` is applied at the base
(:1701-1702), set the flag:

```cpp
        Perturb(base_prox);
        if (dual_shift != 0.0) {
            PerturbC(-dual_shift);
            dc_applied = true;
        }
```

and replace its exit test (:1711-1718, including the local `bool singular` line, which
is deleted — `Singular()` replaces it):

```cpp
        // A singular or wrong-inertia base factorization enters the ladder.
        if (IncEigs == 0 && !Singular())
            return 0;
```

**(d)** In the classic branch (:1719-1730), replace `if (IncEigs <= 0) return 0;` with:

```cpp
        if (Singular())
            EngageDualReg();
        if (IncEigs == 0 && !Singular())
            return 0;
```

**(e)** In the shared ladder (:1733-1753), replace `if (IncEigs <= 0) return i + 1;`
with:

```cpp
        if (Singular())
            EngageDualReg();
        if (IncEigs == 0 && !Singular())
            return i + 1;
```

**(f)** At the exhaustion tail (:1754-1760), extend the warning with the inertia
triple and set the new out-param — replace the block with:

```cpp
    if (settings_.print_level_ < 3)
        fmt::print(fmt::fg(fmt::color::yellow),
                   "Warning: Inertia correction exhausted ({} perturbation attempts, "
                   "inertia p/n/z = {}/{}/{}, expected {}/{}/0)\n",
                   settings_.max_refac_, this->kkt_sol_.peigs(), this->kkt_sol_.neigs(),
                   this->kkt_dim_ - this->kkt_sol_.peigs() - this->kkt_sol_.neigs(),
                   this->kkt_dim_ - (this->equal_cons_ + this->inequal_cons_),
                   this->equal_cons_ + this->inequal_cons_);
    exhausted = true;
    return settings_.max_refac_;
```

**(g)** Change the signature (definition :1639-1641) to append `bool &exhausted`
after `dual_shift` and drop the two `= 0.0` defaults (single call site; a missed
caller must be a compile error):

```cpp
int tycho::solvers::PSIOPT::factor_impl(bool docompute, bool Zfac, double ipurt, double incpurt0,
                                        double incpurt, double &finalpert, double &cumpert,
                                        double base_prox, double dual_shift, bool &exhausted) {
```

- [ ] **Step 3: Update the declaration and the call site**

**(a)** `include/tycho/detail/solvers/psiopt.h:1016-1018` — new declaration:

```cpp
    int factor_impl(bool docompute, bool ZFac, double ipurt, double incpurt0, double incpurt,
                    double &finalpert, double &cumpert, double base_prox, double dual_shift,
                    bool &exhausted);
```

and rewrite the last sentence of its doc comment (:1012-1015, the part describing
`base_prox`/`dual_shift`) to:

```cpp
    // `base_prox` is the proximal-regularization base shift (ρ_k on the Hessian
    // diagonal), read only when inertia_mode_ == proximal_regularization.
    // `dual_shift` is the δ_c magnitude AVAILABLE to this call for both modes:
    // the proximal branch applies it up-front as part of the base matrix; the
    // classic branch applies it on demand, at most once, when a factorization
    // reports rank deficiency (0.0 = suppressed, e.g. during nested l1
    // restoration). `exhausted` is set (never cleared) when the ladder runs out
    // of attempts with inertia still wrong -- the return value alone cannot
    // distinguish that from success on the final attempt.
```

**(b)** `src/solvers/psiopt.cpp:2279-2293` — replace the block from
`double base_prox = 0.0;` through the stale policy comment (ending `...could break
existing convergence behavior.`) with:

```cpp
        // δ_c availability is computed for BOTH inertia modes now: the classic
        // ladder engages it on demand when a factorization reports rank
        // deficiency (see factor_impl). Suppressed while a nested l1
        // restoration phase is active -- the elastic pivots own the
        // constraint-row diagonals (inertia_regularization.h).
        const bool dc_suppressed = this->restoration_ && this->restoration_->is_active() &&
                                   this->restoration_->is_nested();
        double base_prox = 0.0;
        double dual_shift = dc_suppressed ? 0.0 : tycho::solvers::dual_regularization(mu);
        if (settings_.inertia_mode_ == InertiaModes::proximal_regularization)
            base_prox = rho_k;

        bool kkt_exhausted = false;
        Citer.h_facs_ = this->factor_impl(false, Zfac, Hpert0, Incr, Incr2, nhpert, nhpert_cum,
                                          base_prox, dual_shift, kkt_exhausted);
        // kkt_exhausted routing (forced step rejection -> recovery chain ->
        // SINGULAR_KKT abort) lands in the next commit.
        (void)kkt_exhausted;
```

Leave the proximal display/decay block (:2305-2319) untouched — it stays gated on the
proximal mode and correctly records `dual_shift` there.

- [ ] **Step 4: Probe red/green (no full build yet)**

Write `/tmp/ic_probe.cpp` (location is free; keep it out of the repo):

```cpp
// IC probe: classic mode must now converge on (1) the Maratos corpus and
// (2) the duplicated-equality rank-deficient NLP.
#include <tycho/solvers.h>
#include <tycho/vector_functions.h>
#include <Eigen/Core>
#include <cstdio>

using tycho::vf::Arguments;
using tycho::vf::GenericFunction;
using tycho::solvers::OptimizationProblem;

static int report(const char *name, OptimizationProblem &prob, double expect_obj) {
    prob.optimizer_->set_print_level(3);
    auto flag = prob.optimize();
    const auto &r = prob.optimizer_->result();
    bool ok = static_cast<int>(flag) == 0 && std::abs(r.obj_val_ - expect_obj) < 1e-4 &&
              r.iter_num_ <= 60;
    std::printf("%s: flag=%d obj=%.8f iters=%d  -> %s\n", name, static_cast<int>(flag),
                r.obj_val_, static_cast<int>(r.iter_num_), ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int rc = 0;
    {
        OptimizationProblem prob;
        prob.set_vars((Eigen::VectorXd(2) << 0.0, 1.0).finished());
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob.add_objective(GenericFunction<-1, 1>(2.0 * (x0 * x0 + x1 * x1 - 1.0) - x0),
                               (Eigen::VectorXi(2) << 0, 1).finished());
        }
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob.add_equal_con(GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 1.0),
                               (Eigen::VectorXi(2) << 0, 1).finished());
        }
        rc |= report("maratos  ", prob, -1.0);
    }
    {
        // min x0^2 + x1^2 s.t. x0 + x1 = 1 (twice) -> rank(J)=1 < m=2, optimum 0.5.
        OptimizationProblem prob;
        prob.set_vars((Eigen::VectorXd(2) << 0.0, 0.0).finished());
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob.add_objective(GenericFunction<-1, 1>(x0 * x0 + x1 * x1),
                               (Eigen::VectorXi(2) << 0, 1).finished());
        }
        for (int k = 0; k < 2; ++k) {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob.add_equal_con(GenericFunction<-1, -1>(x0 + x1 - 1.0),
                               (Eigen::VectorXi(2) << 0, 1).finished());
        }
        rc |= report("dup-eqcon", prob, 0.5);
    }
    return rc;
}
```

Build and run (from the repo root; this recompiles the 7 solver TUs against your edit
and links them with the prebuilt archives — the probe TU carries no `AccelerateImpl`
instantiations, so this is a faithful A/B of the solver logic):

```bash
mkdir -p /tmp/ic_objs
python3 - <<'EOF'
import json, shlex, subprocess, os
cc = json.load(open('build/compile_commands.json'))
tus = ['psiopt.cpp', 'psiopt_globalization.cpp', 'psiopt_print.cpp',
       'non_linear_program.cpp', 'optimization_problem.cpp', 'solver_init.cpp',
       'ipopt_backend_stub.cpp']
for tu in tus:
    e = [x for x in cc if x['file'].endswith('solvers/' + tu)][0]
    args = shlex.split(e['command'])
    out, skip = [], False
    for a in args[1:]:
        if skip: skip = False; continue
        if a in ('-o', '-c'): skip = True; continue
        out.append(a)
    subprocess.run([args[0]] + out + ['-c', e['file'],
                    '-o', f'/tmp/ic_objs/{tu}.o'], check=True)
    print(tu, 'ok', flush=True)
e = [x for x in cc if x['file'].endswith('solvers/psiopt.cpp')][0]
args = shlex.split(e['command'])
out, skip = [], False
for a in args[1:]:
    if skip: skip = False; continue
    if a in ('-o', '-c'): skip = True; continue
    out.append(a)
subprocess.run([args[0]] + out + ['-c', '/tmp/ic_probe.cpp',
                '-o', '/tmp/ic_probe.o'], check=True)
print('probe ok')
EOF
/opt/homebrew/opt/llvm/bin/clang++ /tmp/ic_probe.o /tmp/ic_objs/*.o \
  build/src/liboptimalcontrol.a build/src/libastro.a \
  build/src/libintegrators_instantiations.a build/src/libvf_instantiations.a \
  build/src/libutils.a -framework Accelerate \
  /opt/homebrew/opt/llvm/lib/libomp.dylib -fopenmp=libomp -o /tmp/ic_probe
/tmp/ic_probe; echo "PROBE_EXIT=$?"
```

Expected: both lines `OK`, `PROBE_EXIT=0`. (Pre-change reference, verified 2026-07-25:
maratos flag=2/iters=500; so any FAIL means your edit is wrong — do not proceed.)

- [ ] **Step 5: Full build**

```bash
conda activate tycho && cd /Users/granthec/Source/tycho/build && ninja -j4 all
```

Background it; wait for exit 0. `psiopt.h` changed, so this is the Task-1 full
rebuild (~20-40 min). ONE build only.

- [ ] **Step 6: Run the targeted suites**

```bash
cd /Users/granthec/Source/tycho/build && ctest -R "DivergencePersistence|InertiaRegularization" --output-on-failure
```

Expected: ALL pass — including `MaratosCorpusConvergesAtDefaults` (the primary red
test), the upgraded `ClassicConvergesOnRankDeficientKkt`, `GenuineDivergenceStillAborts`,
and every proximal-mode test.

- [ ] **Step 7: Format and commit**

```bash
cd /Users/granthec/Source/tycho/build && ninja clang-format
cd /Users/granthec/Source/tycho && git add -A src/solvers include/tycho/detail/solvers tests/cpp/solvers
git commit -m "fix(psiopt): full Ipopt IC inertia condition on the classic ladder

Classic factor_impl accepted any factorization without excess negative
eigenvalues -- including singular ones (rank deficiency only warned) and
neigs < m. On Accelerate, which reports inertia honestly (no Pardiso-style
static pivot perturbation), an exactly singular KKT was accepted, solved to
a zero step, and stalled to max_iters (DivergencePersistence Maratos corpus,
500 iters). Accept only inertia exactly (kkt_dim - m, m, 0); on observed
rank deficiency engage the #103 dual regularization (delta_c, Ipopt
jacobian_regularization_value/_exponent) on demand, at most once per call,
suppressed during nested l1 restoration. The shared ladder exit is
strengthened for both inertia modes; delta_w escalation, warm-start, and
display accounting are byte-identical. factor_impl gains a bool &exhausted
out-param (routing lands in the next commit)."
```

### Task 2: `SINGULAR_KKT` + exhaustion routing through the recovery chain

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt_fwd.h:26-36` (enum + severity comment)
- Modify: `src/solvers/psiopt.cpp` (~:2455-2620 region: forced rejection, abort
  terminal)
- Modify: `src/solvers/psiopt_print.cpp:204-216` (exit-code print)
- Modify: `include/tycho/detail/solvers/jet.h:187-203` (flag-tracking switch)
- Modify: `src/bindings/solvers/psiopt_bind.cpp:539-544` (enum binding)
- Modify: `tychopy/_stubs/` (regenerated snapshot)
- Test: `tests/cpp/solvers/test_divergence_persistence.cpp` (new exhaustion test)

**Interfaces:**
- Consumes: `bool kkt_exhausted` local (Task 1), `should_dispatch_recovery(GoodStep,
  Citer)` (recovery_chain.h:176 — `good_step && !citer.accepted_`),
  `RecoveryChain::Action`, `kRecoveryDepthUnresolved`.
- Produces: `tycho::ConvergenceFlags::SINGULAR_KKT = 4` (public, Python-visible).

- [ ] **Step 1: Write the failing exhaustion test**

Append to `tests/cpp/solvers/test_divergence_persistence.cpp` (after
`MaratosCorpusConvergesAtDefaults`):

```cpp
// Exhaustion is Ipopt-faithful: with the perturbation ladder disabled outright
// (max_refac_ = 0), every wrong-inertia factorization exhausts immediately. The
// forced step rejection must consult the recovery chain and -- with none
// configured at defaults -- abort the phase as SINGULAR_KKT promptly, instead
// of crawling to max_iters on a singular system.
TEST(DivergencePersistence, ExhaustedInertiaCorrectionAbortsAsSingularKkt) {
    using tycho::vf::Arguments;
    using tycho::vf::GenericFunction;

    OptimizationProblem prob;
    prob.set_vars((Eigen::VectorXd(2) << 0.0, 1.0).finished());
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_objective(GenericFunction<-1, 1>(2.0 * (x0 * x0 + x1 * x1 - 1.0) - x0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    {
        auto args = Arguments<2>();
        auto x0 = args.coeff<0>();
        auto x1 = args.coeff<1>();
        prob.add_equal_con(GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 1.0),
                           (Eigen::VectorXi(2) << 0, 1).finished());
    }
    prob.optimizer_->set_print_level(0);
    prob.optimizer_->settings().max_refac_ = 0;

    auto flag = prob.optimize();

    EXPECT_EQ(flag, tycho::ConvergenceFlags::SINGULAR_KKT);
    EXPECT_LE(prob.optimizer_->result().iter_num_, 10);
}
```

(It cannot compile until Step 2 adds the enum value — that is the expected "red".)

- [ ] **Step 2: Add the enum value and its consumers**

**(a)** `include/tycho/detail/solvers/psiopt_fwd.h:26-36`:

```cpp
enum class ConvergenceFlags {
    CONVERGED = 0,
    ACCEPTABLE = 1,
    NOTCONVERGED = 2,
    DIVERGING = 3,
    SINGULAR_KKT = 4,
};

// Severity ordering:
// CONVERGED < ACCEPTABLE < NOTCONVERGED < DIVERGING < SINGULAR_KKT
```

(The `operator<=>` below it is value-based and needs no change.)

**(b)** `src/solvers/psiopt_print.cpp` — in the exit-code chain (:206-215), add before
the closing brace of the chain:

```cpp
        } else if (ExitCode == ConvergenceFlags::SINGULAR_KKT) {
            fmt::print(fmt::fg(fmt::color::dark_red), "\nKKT System Persistently Singular\n");
        }
```

**(c)** `include/tycho/detail/solvers/jet.h` — in the `track` lambda's switch
(:190-203), add after the `DIVERGING` case:

```cpp
            case tycho::ConvergenceFlags::SINGULAR_KKT:
                NumDiv++; // counted with divergence in the jet progress line
                break;
```

**(d)** `src/bindings/solvers/psiopt_bind.cpp` — in the `nb::enum_<ConvergenceFlags>`
chain (:539-544), add after the `DIVERGING` value line:

```cpp
        .value("SINGULAR_KKT", ConvergenceFlags::SINGULAR_KKT)
```

- [ ] **Step 3: Route exhaustion at the call site**

All in `src/solvers/psiopt.cpp`, `alg_impl`:

**(a)** In Task 1's call-site block, delete the two placeholder lines
(`// kkt_exhausted routing ... next commit.` and `(void)kkt_exhausted;`).

**(b)** Immediately after the `GoodStep` line-search block (the
`if (GoodStep) { alpha = mechanism_->compute_step(...); } else { Citer.h_facs_ = -1; }`
construct ending ~:2458), insert:

```cpp
        // Inertia-correction exhaustion (Ipopt-faithful fail-the-step): a step
        // solved on a factorization that never reached correct inertia must not
        // be accepted on merit. Force the rejection so the recovery chain gets
        // its say (feasibility switch when configured); if nothing resolves it,
        // the phase aborts as SINGULAR_KKT below. The non-finite-direction case
        // (!GoodStep) skips the dispatch entirely, so it aborts directly.
        if (kkt_exhausted)
            Citer.accepted_ = false;
        bool singular_abort = kkt_exhausted && !GoodStep;
```

**(c)** In the `RecoveryChain::Action::kAcceptAsIs` case (:2503-2527), after the
`try_recenter_elastics` block, add:

```cpp
                if (kkt_exhausted && resolved_depth == kRecoveryDepthUnresolved) {
                    // Exhausted ladder and no recovery link resolved the forced
                    // rejection: discard the step and abort the phase (the Ipopt
                    // analogue is Error_In_Step_Computation). A nested re-center
                    // above stamps kRecoveryDepthRestoration and is a
                    // resolution, so it does not abort.
                    alpha = 0.0;
                    singular_abort = true;
                }
```

(`kRetry`, `kSwitchToFeasibility`, and `kSoftFeasibilityStep` are resolutions — no
change in those cases.)

**(d)** Locate the `XSL += alpha * DXSL;` step commit that follows the recovery
dispatch (it is after `iters.push_back(Citer);` and the `return_best_`/late-callback
blocks — grep for it within `alg_impl`). Immediately AFTER that commit statement,
insert:

```cpp
        if (singular_abort) {
            // alpha was zeroed above, so the commit was a no-op; the iterate and
            // its table row are already recorded. Mirror the divergence-abort
            // exit shape: stamp the flag and leave the iteration loop.
            ExitCode = ConvergenceFlags::SINGULAR_KKT;
            this->result_.converge_flag_ = ExitCode;
            break;
        }
```

Before inserting, read the surrounding ~30 lines: if the loop tail between the commit
and the loop's closing brace performs additional per-iteration state updates (console
print of the row uses `iters.back()` and is fine either side), place the break AFTER
the console-print call so the aborting iteration is visible in the table, and confirm
`ExitCode` is the variable the existing convergence-break path assigns
(`this->result_.converge_flag_ = ExitCode; break;` at ~:2185 is the pattern to match).

- [ ] **Step 4: Probe the routing**

Reuse Task 1's probe pipeline (recompile the 7 TUs + relink — same commands), but
first append a third case to `/tmp/ic_probe.cpp` `main` (before `return rc;`):

```cpp
    {
        // Exhaustion terminal: ladder disabled, Maratos KKT is wrong-inertia at
        // iterate 1 -> expect SINGULAR_KKT (flag 4) within a few iterations.
        OptimizationProblem prob;
        prob.set_vars((Eigen::VectorXd(2) << 0.0, 1.0).finished());
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob.add_objective(GenericFunction<-1, 1>(2.0 * (x0 * x0 + x1 * x1 - 1.0) - x0),
                               (Eigen::VectorXi(2) << 0, 1).finished());
        }
        {
            auto args = Arguments<2>();
            auto x0 = args.coeff<0>();
            auto x1 = args.coeff<1>();
            prob.add_equal_con(GenericFunction<-1, -1>(x0 * x0 + x1 * x1 - 1.0),
                               (Eigen::VectorXi(2) << 0, 1).finished());
        }
        prob.optimizer_->set_print_level(3);
        prob.optimizer_->settings().max_refac_ = 0;
        auto flag = prob.optimize();
        const auto &r = prob.optimizer_->result();
        bool ok = static_cast<int>(flag) == 4 && r.iter_num_ <= 10;
        std::printf("exhaust  : flag=%d iters=%d  -> %s\n", static_cast<int>(flag),
                    static_cast<int>(r.iter_num_), ok ? "OK" : "FAIL");
        rc |= ok ? 0 : 1;
    }
```

Expected: all three lines `OK`, `PROBE_EXIT=0`. The first two cases also re-verify
Task 1 wasn't disturbed.

- [ ] **Step 5: Full build, regenerate stubs, run suites**

```bash
conda activate tycho && cd /Users/granthec/Source/tycho/build && ninja -j4 all
```

Background; wait for exit 0 (psiopt_fwd.h/jet.h changed — the Task-2 full rebuild).
Then:

```bash
cmake --build /Users/granthec/Source/tycho/build --target tychopy_stubs_snapshot
cd /Users/granthec/Source/tycho/build && ctest -R "DivergencePersistence|InertiaRegularization" --output-on-failure
conda run -n tycho python -c "import tychopy; print(tychopy.solvers.ConvergenceFlags.SINGULAR_KKT)"
```

Expected: all tests pass (including the new exhaustion test), the stub diff shows the
new enum value, and the Python import prints the value.

- [ ] **Step 6: Format and commit**

```bash
cd /Users/granthec/Source/tycho/build && ninja clang-format
cd /Users/granthec/Source/tycho && git add -A src include/tycho/detail/solvers tests/cpp/solvers tychopy/_stubs
git commit -m "fix(psiopt): route inertia-correction exhaustion through the recovery chain as SINGULAR_KKT

An exhausted ladder previously proceeded with the bad factorization
(documented warn-and-proceed), which on a persistently singular KKT crawls
to max_iters on zero steps. Ipopt-faithful policy instead: force the
line-search verdict to rejected so recovery_->on_step_rejected gets its say
(feasibility switch when configured); an unresolved rejection discards the
step and aborts the phase with the new ConvergenceFlags::SINGULAR_KKT
(Python-visible; stubs regenerated). Non-finite directions on an exhausted
factorization abort directly (the dispatch gate excludes them)."
```

### Task 3: Gates, delivery, and the follow-up issue

**Files:**
- No source changes. Runs the pre-merge verification sequence, pushes, opens the PR,
  files the proximal-Maratos issue.

- [ ] **Step 1: Full ctest**

```bash
cd /Users/granthec/Source/tycho/build && ctest --output-on-failure 2>&1 | tail -20
```

Expected: exactly ONE failure — `cpp_example_optimal_docking_builder` (pre-existing,
out of scope, spec'd). Any other failure is a regression from this branch: STOP and
fix before proceeding (the task-2 commit is the likely culprit; bisect with the probe).

- [ ] **Step 2: Python examples + brachistochrone**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
cd /Users/granthec/Source/tycho/build && ./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp | tail -5
```

Expected: `34 passed, 0 failed, 0 skipped`; brachistochrone prints "Optimal Solution
Found" with objective ≈ 1.8013.

- [ ] **Step 3: Benchmarks**

```bash
cd /Users/granthec/Source/tycho && bench/bench_track.sh record && bench/bench_track.sh compare
```

If `compare` reports no baseline for the parent, record the comparison as
"no pre-existing baseline on this machine" in the PR body rather than skipping
silently. The changed code paths only fire on wrong-inertia factorizations, so any
benchmark delta beyond noise needs investigation.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin fix/psiopt-inertia-correction
gh pr create --title "fix(psiopt): full Ipopt IC inertia condition + SINGULAR_KKT exhaustion routing" --body-file <body written per below>
```

PR body must include: the Maratos root cause (link the spec and the #88 RESULTS doc);
red→green evidence (Maratos 500-iter stall → converged; upgraded rank-deficient test;
exhaustion test); the one MKL-reachable behavior change (`neigs < m` now corrected —
Linux CI corpus is the gate); the Python-visible enum addition; and the explicit
**PSIOPT-internals human review required** flag. Do NOT merge.

- [ ] **Step 5: File the proximal-Maratos issue**

```bash
gh issue create --title "proximal_regularization diverges on exactly-singular Hessians (Maratos corpus)" \
  --body "On main c9e8ddd, InertiaModes::proximal_regularization DIVERGES on the DivergencePersistence Maratos problem (flag=3, obj ~7.8e17, 4 iterations) while classic (after fix/psiopt-inertia-correction) converges. With Lagrangian Hessian exactly zero, the rho=1e-10 base shift yields a correct-inertia but catastrophically ill-conditioned system; the ~g/rho step blows up before the divergence-persistence window trips. Correct inertia is not a sufficient step-quality gate for the proximal base attempt. Found during the 2026-07-25 inertia-correction design probes (docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md). Candidate directions: floor the base attempt's effective curvature (rho vs a gradient-scaled floor), or run the ladder when the base step norm exceeds a trust bound even with correct inertia."
```

- [ ] **Step 6: Update the design doc status**

Edit `docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md` line 4:
`**Status:** implemented on fix/psiopt-inertia-correction, PR open, awaiting human review`,
then:

```bash
git add docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md
git commit -m "docs(plans): mark inertia-correction design implemented (PR open)" && git push
```

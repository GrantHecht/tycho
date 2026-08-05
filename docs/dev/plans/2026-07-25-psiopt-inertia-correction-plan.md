# PSIOPT Inertia Correction (IPOPT Algorithm IC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Rev 2 (2026-07-27):** re-anchored to the merged tree (main through #114 `e2314a3e`);
degeneracy latch added to Task 1; Task 2 redesigned to the loop-tail exit-flag idiom
introduced by #109/#110. All line anchors below verified on that tree; premises
(classic stalls flag=2/500, proximal diverges flag=3/4) probe-re-verified 2026-07-27.

**Goal:** Make PSIOPT's classic inertia-correction ladder enforce the full IPOPT
condition — inertia exactly `(kkt_dim − m, m, 0)` — engaging the existing #103 dual
regularization on demand (with an IPOPT-style degeneracy latch), and abort the phase as
`SINGULAR_KKT` (after consulting the recovery chain) when correction exhausts.

**Architecture:** All solver logic lives in `PSIOPT::factor_impl` and its one call site
in `alg_impl` (src/solvers/psiopt.cpp). δ_c machinery is reused verbatim from #103.
Exhaustion is signaled by a new `bool &exhausted` out-param, routed by forcing the
line-search verdict to rejected (the recovery chain gets its say), with an unresolved
rejection terminating via the loop-tail exit-flag idiom (`singular_abort`, mirroring
`exit_at_acceptable`/`exit_stage_stalled`) as a new `ConvergenceFlags::SINGULAR_KKT`.

**Tech Stack:** C++20, Eigen, gtest (incl. the existing PSIOPT friend-test pattern),
nanobind (one enum value), the standalone-probe pipeline for fast red/green.

**Spec:** `docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md` (rev 3).

## Global Constraints

- Branch: `fix/psiopt-inertia-correction` (main merged through `e2314a3e`). All work on
  this Mac (16 GB Apple Silicon) — the defect only reproduces on macOS/Accelerate.
- `conda activate tycho` before any build or Python step.
- Builds: `cd /Users/granthec/Source/tycho/build && ninja -j4 all`. **ONE build at a
  time, ever** — two concurrent builds OOM this machine. Launch long builds from the
  controller session in the background and wait for completion.
- `psiopt.h`, `psiopt_fwd.h`, and `jet.h` are in the PCH include chain: touching them
  costs a full rebuild (~20-40 min at `-j4`). Task 1 and Task 2 each contain exactly
  one such rebuild; do not add extra ones.
- **Probe before building:** every code change is first verified through the probe
  pipeline (§Task 1 Step 5; ~1 min). Only after the probe is green do you pay for the
  real build.
- **Never trust a `ctest` run against binaries older than your change.** Rebuild first.
- clangd diagnostics in this repo are unusable (`-fopenmp=libomp` parse failure).
  Ignore editor diagnostics; trust the compiler.
- Do not modify: δ_w escalation constants/flow, `finalpert`/`Hpert0` warm-start,
  `cumpert` accounting, the proximal display/decay block, anything in `notices/`.
- Format before each commit: `cd build && ninja clang-format`. Prefixes: `fix:` /
  `test:` / `docs:`.
- PSIOPT internals + Python-visible enum ⇒ the PR requires Grant's explicit review and
  the Linux CI corpus run. Nothing merges from this plan.
- Expected macOS end state: Maratos test green, `ClassicConvergesOnRankDeficientKkt`
  green, latch test green, exhaustion test green, `GenuineDivergenceStillAborts` and
  all `InertiaRegularizationSolve` tests green; full ctest red ONLY on
  `cpp_example_optimal_docking_builder` (pre-existing, out of scope).

---

### Task 1: Full IC condition + on-demand δ_c + degeneracy latch in `factor_impl`

**Files:**
- Modify: `src/solvers/psiopt.cpp:1062-1183` (`factor_impl` body)
- Modify: `src/solvers/psiopt.cpp:1823-1838` (call site: hoist `dual_shift`, thread
  `exhausted`)
- Modify: `src/solvers/psiopt.cpp:1227` region (per-phase latch reset)
- Modify: `include/tycho/detail/solvers/psiopt.h:987-989` (declaration + doc comment),
  member block (add `dc_latched_`), :54-61 + :763-770 (friend-test declarations)
- Test: `tests/cpp/solvers/test_inertia_regularization.cpp` (upgrade test at :165, new
  latch test)

**Interfaces:**
- Consumes: `NonLinearProgram::perturb_kkt_c_diags(double, Eigen::SparseMatrix<double,
  Eigen::RowMajor>&)`, `tycho::solvers::dual_regularization(double mu)`,
  `build_inertia_duplicated_equality_nlp()` (renamed by #113) and
  `build_inertia_wellcond_nlp()`-equivalent (verify the well-conditioned builder's
  current name in the test file before use).
- Produces: `int factor_impl(bool docompute, bool ZFac, double ipurt, double incpurt0,
  double incpurt, double &finalpert, double &cumpert, double base_prox,
  double dual_shift, bool &exhausted)`; PSIOPT member `bool dc_latched_`; the
  `kkt_exhausted` local Task 2 consumes.

- [ ] **Step 1: Upgrade the classic rank-deficient test**

In `tests/cpp/solvers/test_inertia_regularization.cpp`, replace the whole test
`ClassicOnRankDeficientKktDocumented` (comment block through closing brace, at :165)
with: (as-built note: this name is pre-this-step and no longer exists in the tree —
Step 1 itself renames it to `ClassicConvergesOnRankDeficientKkt`; grepping the
current file for the old name will not find it.)

```cpp
// (b) The SAME rank-deficient problem under classic. The full Ipopt IC condition
// engages the on-demand dual regularization when the factorization reports rank
// deficiency, so classic converges here too. (On MKL the static pivot
// perturbation may mask the deficiency instead; either road must reach the
// unique optimum.)
TEST(InertiaRegularizationSolve, ClassicConvergesOnRankDeficientKkt) {
    auto prob = build_inertia_duplicated_equality_nlp();
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

- [ ] **Step 2: Write the failing latch test (friend-based)**

**(a)** In `include/tycho/detail/solvers/psiopt.h`, add to the forward-declaration
block (:54-61):

```cpp
class InertiaRegularizationSolve_ClassicDegeneracyLatchTracksSingularity_Test;
```

and to the friend block (:763-770):

```cpp
    friend class ::InertiaRegularizationSolve_ClassicDegeneracyLatchTracksSingularity_Test;
```

**(b)** Append to `test_inertia_regularization.cpp` (after the upgraded test; use the
well-conditioned builder under its current #113 name — verify with
`grep -n "wellcond" tests/cpp/solvers/test_inertia_regularization.cpp`):

```cpp
// The degeneracy latch (Ipopt hess/jac-degenerate adaptation, sticky per phase):
// set once delta_c is engaged for a singular factorization, so later iterations
// pre-apply it at the base attempt instead of re-discovering the singularity;
// never set on problems whose factorizations stay full-rank.
TEST(InertiaRegularizationSolve, ClassicDegeneracyLatchTracksSingularity) {
    auto degen = build_inertia_duplicated_equality_nlp();
    degen->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    degen->optimizer_->set_max_iters(100);
    (void)degen->optimize();
    EXPECT_TRUE(degen->optimizer_->dc_latched_)
        << "delta_c engaged on a rank-deficient problem must set the latch";

    auto healthy = build_inertia_wellcond_nlp();
    healthy->optimizer_->settings().inertia_mode_ = InertiaModes::classic;
    (void)healthy->optimize();
    EXPECT_FALSE(healthy->optimizer_->dc_latched_)
        << "a full-rank problem must never engage delta_c or the latch";
}
```

(Adjust the two builder names to the file's actual #113-renamed helpers; the test name
must match the friend declaration exactly.)

- [ ] **Step 3: Rewrite `factor_impl`**

In `src/solvers/psiopt.cpp` (function at :1062):

**(a)** After the `Inertia` lambda (:1064-1066), add:

```cpp
    // Full Ipopt inertia-correction condition (Algorithm IC, Wächter & Biegler
    // 2006): accept only inertia exactly (kkt_dim - m, m, 0). Singular() is the
    // rank-deficiency part; Inertia() != 0 covers both excess (the only case the
    // pre-2026-07 ladder corrected) and missing negative eigenvalues.
    auto Singular = [&]() {
        return (this->kkt_sol_.neigs() + this->kkt_sol_.peigs() - this->kkt_dim_) != 0;
    };
```

**(b)** After the `Compute` lambda (:1102), add (the `PerturbC` lambda MOVES here from
the proximal branch :1120-1122 — delete it there):

```cpp
    auto PerturbC = [&](double p) {
        this->nlp_->perturb_kkt_c_diags(p, this->kkt_sol_.get_matrix());
    };
    // On-demand dual regularization (delta_c). dual_shift is the magnitude
    // available to this call (0.0 while a nested l1 restoration phase owns the
    // constraint-row diagonals -- the caller suppresses it). The proximal branch
    // applies it up-front as part of the base matrix; the classic branch applies
    // it here, at most once per call, the first time a factorization reports
    // rank deficiency -- it lands in the matrix and takes effect at the next
    // Refactor(), so a singular base costs one ladder rung (a small delta_w
    // rides along with delta_c, matching Ipopt, which raises both on
    // singularity). dc_latched_ is the Ipopt hess/jac-degenerate adaptation:
    // sticky per phase, it makes later calls pre-apply delta_c at the base
    // attempt instead of re-paying the singular factorization every iteration.
    bool dc_applied = false;
    auto EngageDualReg = [&]() {
        if (!dc_applied && dual_shift != 0.0) {
            PerturbC(-dual_shift);
            dc_applied = true;
            this->dc_latched_ = true;
        }
    };
```

**(c)** Proximal branch: set the flag where the base δ_c is applied (:1123-1125):

```cpp
        Perturb(base_prox);
        if (dual_shift != 0.0) {
            PerturbC(-dual_shift);
            dc_applied = true;
        }
```

and replace its exit test (:1134-1141, deleting the local `bool singular` line):

```cpp
        // A singular or wrong-inertia base factorization enters the ladder.
        if (IncEigs == 0 && !Singular())
            return 0;
```

**(d)** Classic branch (:1142-1153) becomes:

```cpp
    } else if (Zfac || docompute) {
        if (this->dc_latched_)
            EngageDualReg();
        if (!docompute)
            Refactor();
        else
            Compute();
        CheckInfo();
        RankDef();
        IncEigs = Inertia();
        finalpert = 0.0;
        if (Singular())
            EngageDualReg();
        if (IncEigs == 0 && !Singular())
            return 0;
    }
```

**(e)** Shared ladder (:1156-1176): replace `if (IncEigs <= 0) return i + 1;` with:

```cpp
        if (Singular())
            EngageDualReg();
        if (IncEigs == 0 && !Singular())
            return i + 1;
```

**(f)** Exhaustion tail (:1177-1182) becomes:

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

**(g)** Signature (definition :1062-1064): append `bool &exhausted` after
`dual_shift`, drop the parameter defaults:

```cpp
int tycho::solvers::PSIOPT::factor_impl(bool docompute, bool Zfac, double ipurt, double incpurt0,
                                        double incpurt, double &finalpert, double &cumpert,
                                        double base_prox, double dual_shift, bool &exhausted) {
```

- [ ] **Step 4: Declaration, member, per-phase reset, call site**

**(a)** `psiopt.h:987-989` — declaration matching (g) (no defaults); rewrite the
`base_prox`/`dual_shift` sentence of the doc comment above it to:

```cpp
    // `base_prox` is the proximal-regularization base shift (ρ_k on the Hessian
    // diagonal), read only when inertia_mode_ == proximal_regularization.
    // `dual_shift` is the δ_c magnitude AVAILABLE to this call for both modes:
    // the proximal branch applies it up-front; the classic branch applies it on
    // demand when a factorization reports rank deficiency, or up-front once
    // dc_latched_ is set (0.0 = suppressed, e.g. during nested l1 restoration).
    // `exhausted` is set (never cleared) when the ladder runs out of attempts
    // with inertia still wrong -- the return value alone cannot distinguish
    // that from success on the final attempt.
```

**(b)** `psiopt.h`, near the other per-solve mutable state members: add

```cpp
    // Degeneracy latch (Ipopt hess_degenerate_/jac_degenerate_ adaptation,
    // simplified to sticky-per-phase): set by factor_impl when the on-demand
    // dual regularization first engages; later classic base attempts pre-apply
    // δ_c instead of re-discovering the singularity. Reset at each alg_impl
    // phase init. See inertia_regularization.h for the δ_c reference.
    bool dc_latched_ = false;
```

**(c)** `psiopt.cpp:1227` — next to `this->result_.last_kkt_info_ = Eigen::Success;`
(per-phase init), add:

```cpp
    // Fresh phase: re-probe rank rather than inheriting the previous phase's
    // degeneracy diagnosis.
    this->dc_latched_ = false;
```

**(d)** Call site :1823-1838 — replace from `double base_prox = 0.0;` through the
stale proceed-anyway comment (`// Note: if factor_impl exhausted ... existing
convergence behavior.`) with:

```cpp
        // δ_c availability is computed for BOTH inertia modes: the classic
        // ladder engages it on demand when a factorization reports rank
        // deficiency (see factor_impl). Suppressed while a nested l1
        // restoration phase owns the constraint-row diagonals
        // (inertia_regularization.h).
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

Check whether the surrounding code already computes an equivalent nested-active bool
in scope (post-#110 the region changed); if so, reuse it instead of `dc_suppressed`
and say so in the commit message. Leave the proximal display/decay block untouched.

- [ ] **Step 5: Probe red/green (no full build)**

Write `/tmp/ic_probe.cpp` — the three-case probe (Maratos classic must converge;
duplicated-constraint classic must converge; both `flag==0`, `iters<=60`; keep a
proximal informational line if desired):

```cpp
#include <tycho/solvers.h>
#include <tycho/vector_functions.h>
#include <Eigen/Core>
#include <cmath>
#include <cstdio>

using tycho::vf::Arguments;
using tycho::vf::GenericFunction;
using tycho::solvers::OptimizationProblem;

static void add_maratos(OptimizationProblem &prob) {
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
}

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
        add_maratos(prob);
        rc |= report("maratos  ", prob, -1.0);
    }
    {
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

Build and run (repo root; compiles every solver TU — except the real-Ipopt adapter —
against your edit, falling back to psiopt.cpp's flags for TUs missing from the stale
compile_commands.json, then links against the prebuilt non-solver archives):

```bash
mkdir -p /tmp/ic_objs
python3 - <<'EOF'
import json, shlex, subprocess, os, glob
cc = json.load(open('build/compile_commands.json'))
def flags_for(suffix, fallback):
    hits = [x for x in cc if x['file'].endswith(suffix)]
    e = hits[0] if hits else fallback
    args = shlex.split(e['command'])
    out, skip = [], False
    for a in args[1:]:
        if skip: skip = False; continue
        if a in ('-o', '-c'): skip = True; continue
        out.append(a)
    return args[0], out
fallback = [x for x in cc if x['file'].endswith('solvers/psiopt.cpp')][0]
for f in sorted(glob.glob('src/solvers/*.cpp')):
    tu = os.path.basename(f)
    if tu == 'ipopt_tnlp_adapter.cpp':
        continue  # real-Ipopt TU; the stub provides ipopt_backend::solve here
    cxx, out = flags_for('solvers/' + tu, fallback)
    subprocess.run([cxx] + out + ['-c', f, '-o', f'/tmp/ic_objs/{tu}.o'], check=True)
    print(tu, 'ok', flush=True)
cxx, out = flags_for('solvers/psiopt.cpp', fallback)
subprocess.run([cxx] + out + ['-c', '/tmp/ic_probe.cpp', '-o', '/tmp/ic_probe.o'], check=True)
print('probe ok')
EOF
/opt/homebrew/opt/llvm/bin/clang++ /tmp/ic_probe.o /tmp/ic_objs/*.o \
  build/src/liboptimalcontrol.a build/src/libastro.a \
  build/src/libintegrators_instantiations.a build/src/libvf_instantiations.a \
  build/src/libutils.a -framework Accelerate \
  /opt/homebrew/opt/llvm/lib/libomp.dylib -fopenmp=libomp -o /tmp/ic_probe
/tmp/ic_probe; echo "PROBE_EXIT=$?"
```

Expected: both lines `OK`, `PROBE_EXIT=0`. (Pre-change reference, re-verified
2026-07-27 on the merged tree: maratos flag=2/iters=500. Any FAIL means the edit is
wrong — do not proceed to the build.)

- [ ] **Step 6: Full build**

```bash
conda activate tycho && cd /Users/granthec/Source/tycho/build && ninja -j4 all
```

Background it; wait for exit 0. This is Task 1's one full rebuild (psiopt.h changed).

- [ ] **Step 7: Targeted suites**

```bash
cd /Users/granthec/Source/tycho/build && ctest -R "DivergencePersistence|InertiaRegularization" --output-on-failure
```

Expected: ALL pass — `MaratosCorpusConvergesAtDefaults`,
`ClassicConvergesOnRankDeficientKkt`, `ClassicDegeneracyLatchTracksSingularity`,
`GenuineDivergenceStillAborts`, and every proximal-mode test.

- [ ] **Step 8: Format and commit**

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
rank deficiency engage the #103 dual regularization on demand, at most once
per call, suppressed during nested l1 restoration, with an Ipopt
hess/jac-degenerate-style latch (sticky per phase) so persistently
rank-deficient problems pre-apply delta_c at the base attempt instead of
re-paying the singular factorization every iteration. The shared ladder exit
is strengthened for both inertia modes; delta_w escalation, warm-start, and
display accounting are byte-identical. factor_impl gains a bool &exhausted
out-param (routing lands in the next commit)."
```

### Task 2: `SINGULAR_KKT` + exhaustion routing (loop-tail exit-flag idiom)

**Files:**
- Modify: `include/tycho/detail/solvers/psiopt_fwd.h:26-36` (enum + severity comment)
- Modify: `src/solvers/psiopt.cpp` (declaration region ~:2000; forced rejection
  before :2054; `kAcceptAsIs` case :2062; ExitCode override after :2269-2270; exit
  disjunction :2281-2283)
- Modify: `src/solvers/psiopt_print.cpp:204-216` (exit-code print)
- Modify: `include/tycho/detail/solvers/jet.h:187-203` (flag-tracking switch)
- Modify: `src/bindings/solvers/psiopt_bind.cpp:601-605` (enum binding)
- Modify: `tychopy/_stubs/` (regenerated snapshot)
- Test: `tests/cpp/solvers/test_divergence_persistence.cpp` (new exhaustion test)

**Interfaces:**
- Consumes: `kkt_exhausted` (Task 1), `should_dispatch_recovery` (recovery_chain.h:176,
  `good_step && !citer.accepted_`), `RecoveryChain::Action`,
  `kRecoveryDepthUnresolved`, the loop-tail idiom precedents `exit_at_acceptable`
  (declared :2000) and `exit_stage_stalled` (:1665).
- Produces: `tycho::ConvergenceFlags::SINGULAR_KKT = 4` (public, Python-visible).

- [ ] **Step 1: Write the failing exhaustion test**

Append to `tests/cpp/solvers/test_divergence_persistence.cpp` (after
`MaratosCorpusConvergesAtDefaults`; reuse the file's `using` declarations):

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

(Red = does not compile until Step 2 adds the enum value. Note `max_refac_` is a plain
public `int` member of Settings with no validating setter — direct assignment is the
intended route.)

- [ ] **Step 2: Add the enum value and its consumers**

**(a)** `psiopt_fwd.h:26-33`:

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

(The value-based `operator<=>` below needs no change.)

**(b)** `psiopt_print.cpp` exit-code chain (:206-215) — add before the chain's close:

```cpp
        } else if (ExitCode == ConvergenceFlags::SINGULAR_KKT) {
            fmt::print(fmt::fg(fmt::color::dark_red), "\nKKT System Persistently Singular\n");
        }
```

**(c)** `jet.h` `track` switch (:190-203) — after the `DIVERGING` case:

```cpp
            case tycho::ConvergenceFlags::SINGULAR_KKT:
                NumDiv++; // counted with divergence in the jet progress line
                break;
```

**(d)** `psiopt_bind.cpp` enum chain (ends :605 `.value("DIVERGING", ...);`) — extend:

```cpp
        .value("DIVERGING", ConvergenceFlags::DIVERGING)
        .value("SINGULAR_KKT", ConvergenceFlags::SINGULAR_KKT);
```

- [ ] **Step 3: Route exhaustion (exit-flag idiom)**

All in `src/solvers/psiopt.cpp`, `alg_impl`:

**(a)** Delete Task 1's two placeholder lines at the call site
(`// kkt_exhausted routing ...` and `(void)kkt_exhausted;`).

**(b)** Next to `bool exit_at_acceptable = false;` (:2000), declare:

```cpp
        // Set by the exhausted-inertia-correction dispatch below: the KKT
        // factorization never reached correct inertia, the forced rejection
        // went through the recovery chain, and nothing resolved it. Terminates
        // the phase as SINGULAR_KKT at the loop tail (same idiom as
        // exit_at_acceptable / exit_stage_stalled).
        bool singular_abort = false;
```

**(c)** Immediately after the `if (GoodStep) { alpha = mechanism_->compute_step(...); }
else { Citer.h_facs_ = -1; }` construct (ends ~:2013), insert:

```cpp
        // Inertia-correction exhaustion (Ipopt-faithful fail-the-step): a step
        // solved on a factorization that never reached correct inertia must not
        // be accepted on merit. Force the rejection so the recovery chain gets
        // its say (feasibility switch when configured); if nothing resolves it,
        // singular_abort terminates the phase below. The non-finite case
        // (!GoodStep) already exits as DIVERGING at the loop tail -- the
        // non-finite verdict dominates and needs no special-casing here.
        if (kkt_exhausted)
            Citer.accepted_ = false;
```

**(d)** In `RecoveryChain::Action::kAcceptAsIs` (:2062), after the
`try_recenter_elastics` block:

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

(`kRetry`, `kSwitchToFeasibility`, `kSoftFeasibilityStep` are resolutions — no change.)

**(e)** After the `exit_at_acceptable` upgrade block (:2269-2270,
`if (exit_at_acceptable && ExitCode == ConvergenceFlags::NOTCONVERGED) ...`), insert:

```cpp
        // SINGULAR_KKT is decisive: an inertia-correction failure is a
        // step-computation error (Ipopt Error_In_Step_Computation), reported as
        // such even at an otherwise-acceptable iterate.
        if (singular_abort)
            ExitCode = ConvergenceFlags::SINGULAR_KKT;
```

**(f)** Extend the exit disjunction (:2281-2283) with the new flag:

```cpp
        if (ExitCode == ConvergenceFlags::CONVERGED || ExitCode == ConvergenceFlags::ACCEPTABLE ||
            ExitCode == ConvergenceFlags::DIVERGING || ExitCode == ConvergenceFlags::SINGULAR_KKT ||
            exit_stage_stalled || i == (settings_.max_iters_ - 1)) {
```

The terminating iteration never reaches the `XSL += alpha * DXSL;` commit (:2295), so
`alpha = 0.0` in (d) is belt-and-suspenders, not load-bearing.

- [ ] **Step 4: Probe the routing**

Append a third case to `/tmp/ic_probe.cpp` `main` (before `return rc;`), then rebuild
and run with the same Step-5 pipeline from Task 1:

```cpp
    {
        // Exhaustion terminal: ladder disabled, Maratos KKT is wrong-inertia at
        // iterate 1 -> expect SINGULAR_KKT (flag 4) within a few iterations.
        OptimizationProblem prob;
        add_maratos(prob);
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

Expected: all three lines `OK`, `PROBE_EXIT=0` (first two re-verify Task 1).

- [ ] **Step 5: Full build, stubs, suites, Python smoke**

```bash
conda activate tycho && cd /Users/granthec/Source/tycho/build && ninja -j4 all
```

Background; wait (Task 2's one full rebuild — psiopt_fwd.h/jet.h changed). Then:

```bash
cmake --build /Users/granthec/Source/tycho/build --target tychopy_stubs_snapshot
cd /Users/granthec/Source/tycho/build && ctest -R "DivergencePersistence|InertiaRegularization" --output-on-failure
conda run -n tycho python -c "import tychopy; print(tychopy.solvers.ConvergenceFlags.SINGULAR_KKT)"
```

Expected: all tests pass; the stub diff shows the enum value; Python prints it.

- [ ] **Step 6: Format and commit**

```bash
cd /Users/granthec/Source/tycho/build && ninja clang-format
cd /Users/granthec/Source/tycho && git add -A src include/tycho/detail/solvers tests/cpp/solvers tychopy/_stubs
git commit -m "fix(psiopt): route inertia-correction exhaustion through the recovery chain as SINGULAR_KKT

An exhausted ladder previously proceeded with the bad factorization
(documented warn-and-proceed), which on a persistently singular KKT crawls
to max_iters on zero steps. Ipopt-faithful policy instead: force the
line-search verdict to rejected so recovery_->on_step_rejected gets its say
(feasibility switch when configured); an unresolved rejection terminates the
phase via the loop-tail exit-flag idiom (singular_abort, alongside
exit_at_acceptable / exit_stage_stalled) with the new
ConvergenceFlags::SINGULAR_KKT (Python-visible; stubs regenerated).
Non-finite directions keep their existing DIVERGING exit."
```

### Task 3: Gates, delivery, and the follow-up issues

**Files:** none (verification, push, PR, issues).

- [ ] **Step 1: Full ctest**

```bash
cd /Users/granthec/Source/tycho/build && ctest --output-on-failure 2>&1 | tail -20
```

Expected: exactly ONE failure — `cpp_example_optimal_docking_builder` (pre-existing,
out of scope, spec'd). Anything else is a regression from this branch: STOP and fix
(bisect with the probe) before proceeding.

- [ ] **Step 2: Python examples + brachistochrone**

```bash
conda run -n tycho bash -c "MPLBACKEND=Agg python scripts/run_examples.py"
cd /Users/granthec/Source/tycho/build && ./examples/cpp_examples/static/brachistochrone/brachistochrone_cpp | tail -5
```

Expected: `34 passed, 0 failed, 0 skipped` (count may have grown with merged examples —
all must pass, none skipped); "Optimal Solution Found", objective ≈ 1.8013.

- [ ] **Step 3: Benchmarks**

```bash
cd /Users/granthec/Source/tycho && bench/bench_track.sh record && bench/bench_track.sh compare
```

If no baseline exists for the parent, record that verbatim in the PR body rather than
skipping silently. The changed paths only fire on wrong-inertia factorizations, so any
delta beyond noise needs investigation.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin fix/psiopt-inertia-correction
gh pr create --title "fix(psiopt): full Ipopt IC inertia condition + SINGULAR_KKT exhaustion routing" --body-file <body>
```

Body must include: root cause (link spec rev 3 + the #88 RESULTS doc); red→green
evidence (Maratos 500-iter stall → converged; rank-deficient test; latch test;
exhaustion test); the one MKL-reachable behavior change (`neigs < m` now corrected —
Linux CI corpus gates); the Python-visible enum addition; the parity statement (which
IPOPT mechanisms are matched, which deviations remain and why); and the explicit
**PSIOPT-internals human review required** flag. Do NOT merge.

- [ ] **Step 5: File the two follow-up issues**

```bash
gh issue create --title "proximal_regularization diverges on exactly-singular Hessians (Maratos corpus)" \
  --body "On main c9e8ddd (re-verified after the #108-#114 merge), InertiaModes::proximal_regularization DIVERGES on the DivergencePersistence Maratos problem (flag=3, obj ~7.8e17, 4 iterations) while classic (after fix/psiopt-inertia-correction) converges. With the Lagrangian Hessian exactly zero, the rho=1e-10 base shift yields a correct-inertia but catastrophically ill-conditioned system; the ~g/rho step blows up before the divergence-persistence window trips. Correct inertia is not a sufficient step-quality gate for the proximal base attempt. Found during the 2026-07-25 inertia-correction design probes (docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md). Candidate directions: floor the base attempt's effective curvature, or enter the ladder when the base step norm exceeds a trust bound even with correct inertia."

gh issue create --title "campaign: measure Ipopt's delta_w escalation schedule as a sweep arm" \
  --body "The inertia-correction fix (fix/psiopt-inertia-correction) adopts Ipopt Algorithm IC's condition and delta_c mechanism but deliberately keeps PSIOPT's native delta_w ladder constants (Hpert0/delta_h_/incr_h_/decr_h_), because the globalization campaign corpus was measured against them. If Ipopt's schedule (delta_w0=1e-4, x100 first-ever, x8 in-episode, /3 warm-start decay, cap 1e40) is to be adopted, it should be measured as a #108 sweep-driver arm against the corpus, not folded into a correctness fix. See docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md (design decision 5)."
```

- [ ] **Step 6: Mark the design implemented**

Edit the spec's `**Status:**` line to
`implemented on fix/psiopt-inertia-correction, PR open, awaiting human review`, then:

```bash
git add docs/dev/plans/2026-07-25-psiopt-inertia-correction-design.md
git commit -m "docs(plans): mark inertia-correction design implemented (PR open)" && git push
```

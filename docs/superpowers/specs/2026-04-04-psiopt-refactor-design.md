# PSIOPT Readability Refactor — Design Spec

**Date:** 2026-04-04
**Status:** Draft
**Scope:** Refactor PSIOPT for readability and encapsulation without changing the algorithm or sacrificing performance.

---

## Goals

1. Convert PSIOPT from a public struct (~60 read-write fields) to a proper class with encapsulated state.
2. Significantly improve readability by grouping related fields, deduplicating code, and removing dead code.
3. Preserve identical algorithmic behavior — all existing tests and examples must pass unchanged.
4. Preserve the Python API — no user-visible syntax changes (property access via `def_prop_rw`).
5. No performance regression.

## Non-Goals

- Changing the optimization algorithm itself.
- Redesigning the Python-facing API (beyond removing dead/internal bindings).
- Adding new features or solver capabilities.

---

## File Organization

### Current (2 source + 1 binding)

- `include/tycho/detail/solvers/psiopt.h` (762 lines)
- `src/solvers/psiopt.cpp` (1268 lines)
- `src/bindings/solvers/psiopt_bind.cpp` (253 lines)

### Proposed (3 source + 1 forward decl + 1 binding)

| File | Purpose |
|---|---|
| `include/tycho/detail/solvers/psiopt_fwd.h` | Forward declarations + `ConvergenceFlags` enum (lightweight include for code that doesn't need the full PSIOPT definition) |
| `include/tycho/detail/solvers/psiopt.h` | PSIOPT class declaration with nested `Settings`, `SolveResult`, `KktVector`, enums. Declarations only — non-trivial method bodies live in .cpp files. |
| `src/solvers/psiopt.cpp` | Algorithm core: `run_phase_sequence`, `alg_impl`, `init_impl`, `ls_impl` (dispatcher + 3 variants + helpers), `factor_impl`, entry points, setters, barrier math, NLP eval wrappers |
| `src/solvers/psiopt_print.cpp` | All printing/formatting methods (still methods on PSIOPT, just compiled in a separate TU) |
| `src/bindings/solvers/psiopt_bind.cpp` | Updated bindings: `def_rw` -> `def_prop_rw` for config, `def_prop_ro` for results, dead bindings removed |

The `psiopt_print.cpp` split is a build-organization choice. The printing methods remain on PSIOPT — no new classes needed.

---

## Nested Types

### `PSIOPT::Settings`

Groups the ~40 configuration fields into a plain aggregate struct with logical sections. No methods, no validation — just data with defaults.

```cpp
struct PSIOPT::Settings {
    // --- Iteration limits ---
    int max_iters = 500;
    int max_ls_iters = 2;
    int max_acc_iters = 50;
    int max_refac = 15;
    int max_soc = 1;
    int max_feas_rest = 2;

    // --- Convergence tolerances ---
    double kkt_tol = 1.0e-6;
    double econ_tol = 1.0e-6;
    double icon_tol = 1.0e-6;
    double bar_tol = 1.0e-6;

    // --- Acceptable tolerances ---
    double acc_kkt_tol = 1.0e-2;
    double acc_econ_tol = 1.0e-3;
    double acc_icon_tol = 1.0e-3;
    double acc_bar_tol = 1.0e-3;

    // --- Unacceptable tolerances ---
    double unacc_kkt_tol = 10;
    double unacc_econ_tol = 2;
    double unacc_icon_tol = 2;
    double unacc_bar_tol = 2;

    // --- Divergence tolerances ---
    double div_kkt_tol = 1.0e15;
    double div_econ_tol = 1.0e15;
    double div_icon_tol = 1.0e15;
    double div_bar_tol = 1.0e15;

    // --- Algorithm modes ---
    AlgorithmModes soe_mode = AlgorithmModes::SOE;
    BarrierModes opt_bar_mode = BarrierModes::LOQO;
    BarrierModes soe_bar_mode = BarrierModes::LOQO;
    LineSearchModes opt_ls_mode = LineSearchModes::AUGLANG;
    LineSearchModes soe_ls_mode = LineSearchModes::NOLS;
    PDStepStrategies pd_step_strategy = PDStepStrategies::PrimSlackEq_Iq;

    // --- Barrier parameters ---
    double init_mu = 0.001;
    double max_mu = 100.0;
    double min_mu = 1.0e-12;

    // --- Step parameters ---
    double bound_fraction = 0.99;
    double bound_push = 1.0e-3;
    double neg_slack_reset = 1.0e-12;
    double soe_bound_relax = 1.0e-8;
    double min_ls_step = 0.01;
    double alpha_red = 2.0;

    // --- Hessian perturbation ---
    double delta_h = 1.0e-5;
    double incr_h = 8.0;
    double decr_h = 0.333333;

    // --- QP solver ---
    int qp_threads = TYCHO_DEFAULT_QP_THREADS;  // clamped in PSIOPT ctor
    QPAlgModes qp_alg = QPAlgModes::Classic;
    QPOrderingModes qp_ord = QPOrderingModes::METIS;
    QPPivotModes qp_pivot_strategy = QPPivotModes::TwoByTwo;
    int qp_matching = 1;
    int qp_scaling = 0;
    int qp_pivot_perturb = 8;
    int qp_ref_steps = 0;
    int qp_par_solve = 0;
    bool qp_print = false;
#ifdef USE_ACCELERATE_SPARSE
    double accel_pivot_tolerance = 0.01;
    double accel_zero_tolerance = 1e-4 * std::numeric_limits<double>::epsilon();
#endif

    // --- Objective ---
    double obj_scale = 1.0;
    double max_cpu_time = 1200;

    // --- Output/behavior ---
    int print_level = 0;
    bool wide_console = false;
    bool cnr_mode = false;
    bool fast_factor_alg = true;
    bool return_best = false;
    BestCriteriaModes best_criteria = BestCriteriaModes::ECONS;
};
```

All validation stays on PSIOPT's setter methods (e.g., `PSIOPT::set_kkt_tol()` validates then writes to `settings_.kkt_tol`). Internal code reads config as `settings_.kkt_tol`.

### `PSIOPT::SolveResult`

Groups all solve output fields. Written by the algorithm, read by callers.

```cpp
struct PSIOPT::SolveResult {
    // --- Solution ---
    Eigen::VectorXd primals;

    // --- Solve outcome ---
    int iter_num = 0;
    double obj_val = 0;
    ConvergenceFlags converge_flag = ConvergenceFlags::NOTCONVERGED;

    // --- Multipliers and constraints ---
    Eigen::VectorXd eq_lmults;
    Eigen::VectorXd iq_lmults;
    Eigen::VectorXd eq_cons;
    Eigen::VectorXd iq_cons;

    // --- Timing (seconds) ---
    double total_time = 0;
    double pre_time = 0;
    double func_time = 0;
    double kkt_time = 0;
    double print_time = 0;
    double misc_time = 0;
    double solver_init_time = 0;

    // --- Factorization stats ---
    int factor_mem = 0;
    int factor_flops = 0;

    void zero_timing() {
        total_time = 0;  pre_time = 0;  func_time = 0;
        kkt_time = 0;  print_time = 0;  misc_time = 0;
        solver_init_time = 0;  iter_num = 0;
    }
};
```

Python bindings expose result fields as **read-only** properties (`def_prop_ro`). Python property names keep the `last_` prefix for backward compatibility (e.g., `def_prop_ro("last_total_time", ...)`).

Entry points populate `result_` fully (including primals) then return `result_.primals` to preserve the existing `Eigen::VectorXd` return type.

### `PSIOPT::KktVector`

Lightweight non-owning view over the compound `[primals | slacks | eq_lmults | iq_lmults]` vector layout. Replaces the 14 `get_*` helper methods.

```cpp
class PSIOPT::KktVector {
public:
    KktVector(Eigen::VectorXd& data, int primal_vars, int slack_vars,
              int equal_cons, int inequal_cons);

    // --- Primal/slack segments ---
    auto primals();
    auto slacks();
    auto primals_slacks();

    // --- Multiplier segments ---
    auto eq_lmults();
    auto iq_lmults();
    auto lmults();

    // --- Gradient/constraint segments (same layout, different semantics) ---
    auto prim_grad();
    auto dual_grad();
    auto prim_dual_grad();
    auto eq_cons();
    auto iq_cons();
    auto all_cons();

    // --- Full vector access ---
    Eigen::VectorXd& data();
    const Eigen::VectorXd& data() const;

private:
    Eigen::VectorXd& data_;
    int pv_, sv_, ec_, ic_;
};
```

Zero overhead — accessors return Eigen segment expressions (inline, same as current helpers). Created at the top of `alg_impl`, `init_impl`, and `ls_*` methods.

Transforms algorithm code from:
```cpp
this->complementarity(this->get_slacks(XSL), this->get_iq_lmults(XSL), avgcomp, mincomp, maxcomp);
```
to:
```cpp
complementarity(xsl.slacks(), xsl.iq_lmults(), avgcomp, mincomp, maxcomp);
```

---

## Entry Point Deduplication

The 5 public entry points share ~30 lines of identical scaffolding (timing, printing, init, misc-time calculation). They differ only in which sequence of algorithm phases they run.

A single private `run_phase_sequence` method handles all scaffolding:

```cpp
struct PhaseStep {
    AlgorithmModes alg_mode;
    BarrierModes bar_mode;
    LineSearchModes ls_mode;
    const char* label;         // e.g., "Optimization Algorithm"
    bool conditional = false;  // only run if previous phase didn't converge
};

Eigen::VectorXd run_phase_sequence(const Eigen::VectorXd& x,
                                   std::initializer_list<PhaseStep> steps);
```

The 5 entry points become thin wrappers:

| Method | Phase sequence |
|---|---|
| `optimize` | OPT |
| `solve` | SOE |
| `solve_optimize` | SOE -> OPT |
| `optimize_solve` | OPT -> SOE (conditional) |
| `solve_optimize_solve` | SOE -> OPT -> SOE (conditional) |

`run_phase_sequence` scaffolding:
1. `result_.zero_timing()`, print stats, `ensure_solver_initialized()`, start timer
2. `analyze_kkt_matrix()`, `init_impl()`
3. For each step: skip if `conditional` and already converged. Otherwise call `alg_impl()` with the step's modes. Re-init between phases (extract primals, re-run `init_impl`).
4. Stop timer, compute misc time, print summary, populate `result_`, return primals.

---

## Line Search Deduplication

`ls_impl` dispatches to three private methods (`ls_lang`, `ls_l1`, `ls_auglang`), each self-contained but calling shared helpers for the duplicated work:

```cpp
double ls_impl(LineSearchModes lsmode, ...) {
    switch (lsmode) {
    case LineSearchModes::LANG:    return ls_lang(...);
    case LineSearchModes::L1:      return ls_l1(...);
    case LineSearchModes::AUGLANG: return ls_auglang(...);
    case LineSearchModes::NOLS:    citer.ls_iters_ = 0; return 1.0;
    }
}
```

Shared helpers extract the duplicated work:

- `eval_trial_point(...)` — compute trial step `XSL2 = XSL + alpha * DXSL`, evaluate NLP at trial point, apply slack resets, compute barrier objective
- `compute_penalties(...)` — compute L1, L2, Linf penalty norms from constraint residuals (returns a `PenaltyTerms` struct)
- `secondary_accept(...)` — check secondary acceptance criteria shared by L1/AUGLANG

Each variant reads cleanly on its own (~20-30 lines), with variant-specific logic inline (e.g., AUGLANG's tolerance-based L1 filtering, different `sc` bias constants).

---

## Dead Code Removal

| Item | Action |
|---|---|
| `diagnostic_` field + empty `if (diagnostic_) {}` block | Remove |
| `print_matrixinfo()` (empty method) | Remove |
| `tst` variable (assigned, never used) | Remove |
| `store_sp_mat_` / `spmat` / `get_sp_mat()` / `get_sp_mat2()` | Remove |
| `ex_obj_val_` (set to `-1.0e20`, never read) | Remove |
| `FIACCO` and `BARDISABLED` barrier modes (declared but unimplemented) | Remove from enum |
| `wide` variable in `print_last_iterate` (unused) | Remove |

---

## Python Binding Changes

### Removed bindings

- `store_sp_mat`, `get_sp_mat`, `get_sp_mat2` — debugging internals
- `diagnostic` — does nothing

### Changed to read-only (`def_rw` -> `def_prop_ro`)

- `last_total_time`, `last_pre_time`, `last_func_time`, `last_kkt_time`, `last_misc_time`, `last_print_time`, `last_solver_init_time`
- `last_iter_num`, `last_obj_val`
- `converge_flag`

### Changed to property-backed (`def_rw` -> `def_prop_rw`)

All remaining config fields route through `settings_` via getter/setter lambdas. Python syntax unchanged (e.g., `optimizer.kkt_tol = 1e-8` still works).

### Preserved as-is

- All validated setter methods (`set_kkt_tol`, `set_tols`, etc.)
- All enum bindings
- `optimize`, `solve`, `solve_optimize` method signatures and return types
- `get_convergence_flag()` method

---

## Header vs. Source File Placement

### Moved to `psiopt.cpp`

- String-to-enum converters (`strto_OrderingMode`, etc.)
- All validated setters with throw logic (`set_max_iters`, `set_kkt_tol`, `set_bound_fraction`, etc.)
- Bulk setters (`set_tols`, `set_acc_tols`, `set_div_tols`, `set_hpert_params`, etc.)
- `set_qp_params()` (40 lines with platform `#ifdef` logic)
- Barrier math helpers (`apply_reset_slacks`, `barrier_objective`, `barrier_gradient`, `barrier_hessian`, `loqo_mu`, `mpc_mu`, `complementarity`, `max_step_to_boundary`)
- NLP eval wrappers (`eval_kkt`, `eval_kkt_no`, `eval_aug`, `eval_soe`, `eval_rhs`)
- `analyze_kkt_matrix()`

### Kept in header (inline)

- `settings()` / `result()` — trivial accessors
- `KktVector` accessors — return Eigen expressions, must be inline for zero overhead
- Callback setters — true one-liners
- `SolveResult::zero_timing()` — small utility
- `print_header()` — static one-liner

---

## PSIOPT Class Shape

```cpp
class PSIOPT {
public:
    // Nested enums (unchanged)
    // Settings, SolveResult, KktVector (see above)

    // Construction
    PSIOPT();
    explicit PSIOPT(std::shared_ptr<NonLinearProgram> np);

    // NLP setup
    void set_nlp(std::shared_ptr<NonLinearProgram> np);
    void release();

    // Public entry points (unchanged signatures)
    Eigen::VectorXd optimize(const Eigen::VectorXd& x);
    Eigen::VectorXd solve(const Eigen::VectorXd& x);
    Eigen::VectorXd solve_optimize(const Eigen::VectorXd& x);
    Eigen::VectorXd optimize_solve(const Eigen::VectorXd& x);
    Eigen::VectorXd solve_optimize_solve(const Eigen::VectorXd& x);

    // Configuration access
    Settings& settings();
    const Settings& settings() const;

    // Validated setters (all existing setters preserved)
    void set_max_iters(int v);
    // ... etc.

    // Results (read-only)
    const SolveResult& result() const;

    // Callbacks
    void set_early_callback(const EarlyCallBackType& f);
    void disable_early_callback();
    void set_late_callback(const LateCallBackType& f);
    void disable_late_callback();

    // Static string-to-enum converters
    static QPOrderingModes strto_OrderingMode(const std::string& str);
    // ... etc.

private:
    Settings settings_;
    SolveResult result_;

    std::shared_ptr<NonLinearProgram> nlp_;
    int primal_vars_ = 0;
    int slack_vars_ = 0;
    int equal_cons_ = 0;
    int inequal_cons_ = 0;
    int kkt_dim_ = 0;

    // KKT linear solver (platform-specific)
    // ... kkt_sol_;
    bool qp_analyzed_ = false;

    // Callbacks
    EarlyCallBackType early_callback_;
    LateCallBackType late_callback_;
    bool early_callback_enabled_ = false;
    bool late_callback_enabled_ = false;

    // Entry point scaffolding
    struct PhaseStep { /* ... */ };
    Eigen::VectorXd run_phase_sequence(const Eigen::VectorXd& x,
                                       std::initializer_list<PhaseStep> steps);

    // Core algorithm
    Eigen::VectorXd alg_impl(...);
    Eigen::VectorXd init_impl(...);

    // Line search
    double ls_impl(...);
    double ls_lang(...);
    double ls_l1(...);
    double ls_auglang(...);
    double eval_trial_point(...);
    PenaltyTerms compute_penalties(...);
    bool secondary_accept(...);

    // KKT factorization
    int factor_impl(...);
    bool analyze_kkt_matrix();
    void set_qp_params();
    void ensure_solver_initialized();

    // Barrier math helpers
    void apply_reset_slacks(...) const;
    double max_step_to_boundary(...) const;
    void complementarity(...) const;
    double barrier_objective(...) const;
    void barrier_gradient(...) const;
    void barrier_hessian(...);
    double loqo_mu(...) const;
    double mpc_mu(...) const;
    void max_primal_dual_step(...);

    // NLP evaluation dispatch
    void eval_nlp(...);
    void eval_kkt(...);
    void eval_kkt_no(...);
    void eval_aug(...);
    void eval_soe(...);
    void eval_rhs(...);
    void fill_iter_info(...);
    ConvergenceFlags converge_check(...);

    // Printing (defined in psiopt_print.cpp)
    static void print_psiopt();
    static void print_header();
    void print_stats();
    void print_settings();
    void print_last_iterate(...);
    void print_beginning(...) const;
    void print_finished(...) const;
    void print_exit_stats(...);
    void print_timing_summary();
    static fmt::text_style calculate_color(...);
};
```

---

## Verification Strategy

No new tests needed — this is a pure refactoring. The existing test suite serves as the regression gate:

1. **C++ unit tests** — `ctest --output-on-failure` — all must pass
2. **Python examples** — all 38 must pass
3. **C++ brachistochrone** — must converge, obj ~1.8013 s
4. **Benchmarks** — `bench/bench_track.sh compare` — no regressions

---

## Constraints

- **PSIOPT optimizer internals** are flagged for human review (per CLAUDE.md). All changes in this refactoring are structural, not algorithmic, but the PR should be reviewed carefully.
- **Python API** remains backward-compatible: all existing property names and method signatures are preserved. The only behavioral change is that result fields become read-only (no legitimate code writes to them).
- No new dependencies introduced.

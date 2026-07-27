"""Single source of truth for corpus problem module names.

Both the harness (``scripts/run_corpus.py``) and the smoke test
(``tychopy/test/test_corpus_smoke.py``) import ``ALL_PROBLEMS`` from here.
A module under ``tests/corpus/problems/`` is part of the corpus if and only
if its (bare, no-package-prefix) name appears in this list.

Order is tier-grouped: degenerate, then hard, then literature. Early
throwaway stub problems (``stub_converges``/``stub_fails``) that used to
head this list to exercise the harness before any real corpus problem
existed have since been removed now that the degenerate/hard/literature
tiers below cover that role.
"""

ALL_PROBLEMS: list[str] = [
    # --- degenerate tier: structurally ill-posed problems (redundant or
    # conflicting constraints, zero objectives, near-infeasibility) ---
    "deg_dup_equality",
    "deg_conflicting_equality",
    "deg_zero_objective",
    "deg_redundant_defects",
    "deg_near_infeasible",
    # --- in-domain hard tier: realistic in-tree examples perturbed into a
    # strained regime (bad scaling, cold starts, degraded initial guesses,
    # tight bounds, stiffness) ---
    "hard_vanderpol",
    "hard_brach_coldstart",
    "hard_brach_illscaled",
    "hard_zermelo_wrongbasin",
    "hard_mountaincar_badguess",
    "hard_lowthrust_badguess",
    "hard_cartpole_tightbounds",
    "hard_hypersens_stiff",
    # --- literature tier: classic NLP counterexamples for interior-point /
    # SQP methods, verified against their cited source ---
    # lit_cycling (Chamberlain-Powell-Lemarechal-Pedersen 1982 watchdog-paper
    # cycling example) is deliberately NOT registered: the paper is
    # paywalled and no accessible reproduction of its actual motivating
    # example (as opposed to just its topic) was found. See
    # tests/corpus/README.md, "Literature tier" section, for the record of
    # what was checked. Corpus target is 15-25; 17 without it is fine.
    "lit_wb2000",
    "lit_maratos",
    "lit_hs13",
    "lit_powell_badscaled",
]

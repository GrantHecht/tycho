"""Single source of truth for corpus problem module names.

Both the harness (``scripts/run_corpus.py``) and the smoke test
(``tychopy/test/test_corpus_smoke.py``) import ``ALL_PROBLEMS`` from here.
A module under ``tests/corpus/problems/`` is part of the corpus if and only
if its (bare, no-package-prefix) name appears in this list.

Order is tier-grouped: degenerate, then hard, then literature. Later G0
tasks append to this list as their modules land; nothing here is deleted
except the Task 1 throwaway stubs, which Task 5 removes once real corpus
problems exist.
"""

ALL_PROBLEMS: list[str] = [
    # --- Task 1 throwaway stubs (deleted in Task 5) ---
    "stub_converges",
    "stub_fails",
    # --- Task 2: degenerate tier ---
    "deg_dup_equality",
    "deg_conflicting_equality",
    "deg_zero_objective",
    "deg_redundant_defects",
    "deg_near_infeasible",
    # --- Task 3: in-domain hard tier ---
    "hard_vanderpol",
    "hard_brach_coldstart",
    "hard_brach_illscaled",
    "hard_zermelo_wrongbasin",
    "hard_mountaincar_badguess",
    "hard_lowthrust_badguess",
    "hard_cartpole_tightbounds",
    "hard_hypersens_stiff",
]

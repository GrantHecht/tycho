"""Single source of truth for corpus problem module names.

Both the harness (``scripts/run_corpus.py``) and the smoke test
(``tychopy/test/test_corpus_smoke.py``) import ``ALL_PROBLEMS`` from here.
A module under ``tests/corpus/problems/`` is part of the corpus if and only
if its (bare, no-package-prefix) name appears in this list.

Order is tier-grouped: degenerate, then hard, then literature. The Task 1
throwaway stubs (``stub_converges``/``stub_fails``) that used to head this
list were removed in Task 5 once the real degenerate/hard/literature tiers
below existed to exercise the harness instead.
"""

ALL_PROBLEMS: list[str] = [
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
    # --- Task 4: literature tier ---
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

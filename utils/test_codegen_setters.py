"""Unit tests for the constraint + auto-setter machinery in CodeGen.py.

Run with:
    conda run -n tycho python -m pytest utils/test_codegen_setters.py -v
"""

import pytest
import sympy as sp
from CodeGen import TychoHeaderGen


def _build(name, Eq, Xs, scalar_params):
    """Construct a header gen and return the generated class-header text."""
    gen = TychoHeaderGen(name, Eq, sp.Matrix(Xs), scalar_params)
    return gen.gen_class_header()


def _toy_inputs_and_eq():
    """Toy 2-in/1-out function whose only parameter is ``k``.

    Returns (Xs, Eq, k_symbol).
    """
    x0, x1 = sp.symbols("x0 x1")
    k = sp.symbols("k")
    Eq = sp.Matrix([k * x0 + x1])
    return [x0, x1], Eq, k


# ── constraints ────────────────────────────────────────────────────────────


def test_ctor_emits_validation_when_constraint_supplied():
    Xs, Eq, k = _toy_inputs_and_eq()
    text = _build("Toy", Eq, Xs, [(k, "gain", "k > 0.0")])

    assert "if (!(k > 0.0))" in text
    assert "throw std::invalid_argument(" in text
    assert '"Toy: k must satisfy k > 0.0"' in text


def test_ctor_omits_validation_when_no_constraint():
    Xs, Eq, k = _toy_inputs_and_eq()
    text = _build("Toy", Eq, Xs, [(k, "gain")])

    assert "invalid_argument" not in text
    assert "throw" not in text


def test_two_element_tuple_backward_compatible():
    """Old-form ``(Symbol, Description)`` entries must still work."""
    Xs, Eq, k = _toy_inputs_and_eq()
    gen = TychoHeaderGen("Toy", Eq, sp.Matrix(Xs), [(k, "gain")])
    assert gen.ScalarParams[0][2] is None  # constraint normalized to None


def test_compound_constraint_preserved_verbatim():
    Xs, Eq, k = _toy_inputs_and_eq()
    text = _build("Toy", Eq, Xs, [(k, "ratio", "k > 0.0 && k < 1.0")])

    assert "if (!(k > 0.0 && k < 1.0))" in text
    assert '"Toy: k must satisfy k > 0.0 && k < 1.0"' in text


# ── setters ─────────────────────────────────────────────────────────────────


def test_setter_emitted_for_every_scalar_param():
    Xs, Eq, k = _toy_inputs_and_eq()
    text = _build("Toy", Eq, Xs, [(k, "gain", "k > 0.0")])

    assert "void set_k(double k) {" in text


def test_setter_revalidates_via_same_constraint():
    Xs, Eq, k = _toy_inputs_and_eq()
    text = _build("Toy", Eq, Xs, [(k, "gain", "k > 0.0")])

    # The validation block must appear at least twice (ctor + setter).
    assert text.count("if (!(k > 0.0))") == 2
    assert text.count('"Toy: k must satisfy k > 0.0"') == 2


def test_setter_omits_validation_without_constraint():
    Xs, Eq, k = _toy_inputs_and_eq()
    text = _build("Toy", Eq, Xs, [(k, "gain")])

    assert "void set_k(double k) {" in text
    assert "invalid_argument" not in text


# ── precomputed-member setters ──────────────────────────────────────────────


def _mee_like_inputs_and_eq():
    """Construct a toy function whose Jacobian yields a parameter-only CSE.

    Uses a ``sqrt(mu)`` subexpression so that _identify_precomputed lifts
    it into a ``pcN_`` member and we can verify the setter recomputes it.
    """
    x = sp.symbols("x")
    mu = sp.symbols("mu")
    # f(x) = sqrt(mu) * x + mu * x**2 — ensures sqrt(mu) shows up in the
    # Jacobian/Hessian often enough for CSE to lift it.
    Eq = sp.Matrix([sp.sqrt(mu) * x + mu * x**2])
    return [x], Eq, mu


def test_setter_recomputes_dependent_pcN():
    Xs, Eq, mu = _mee_like_inputs_and_eq()
    gen = TychoHeaderGen("Toy", Eq, sp.Matrix(Xs), [(mu, "gm", "mu > 0.0")])
    text = gen.gen_class_header()

    # Precondition: at least one precomputed member was identified.
    assert len(gen._precomputed) >= 1, "test setup needs a pcN_ member"

    # The setter body must contain the recomputation for every pcN_
    # that depends (directly or transitively) on mu_.
    setter_start = text.index("void set_mu(double mu) {")
    setter_end = text.index("}", setter_start)
    setter_body = text[setter_start:setter_end]

    for member_name, _ in gen._precomputed:
        assert f"{member_name} =" in setter_body, (
            f"setter body missing recomputation of {member_name}"
        )


def test_vector_param_setter_walks_element_dependencies():
    """VectorParams setters must recompute pcN_ that depend on *element*
    symbols, not the container name. Regression guard for the
    _affected_precomputed compound-param bug: prior to the fix the setter
    looked up ``sp.Symbol("v")`` against expressions containing ``v0/v1/v2``,
    never matched, and silently skipped every recomputation.
    """
    x = sp.symbols("x")
    v0, v1, v2 = sp.symbols("v0 v1 v2")
    # sqrt(v0^2 + v1^2 + v2^2) is a parameter-only subexpression; CSE
    # should lift it into a pcN_ and the setter for v must recompute it.
    vnorm = sp.sqrt(v0**2 + v1**2 + v2**2)
    Eq = sp.Matrix([vnorm * x + (v0**2 + v1**2 + v2**2) * x**2])

    gen = TychoHeaderGen(
        "VecToy",
        Eq,
        sp.Matrix([x]),
        ScalarParams=[],
        VectorParams=[(sp.Matrix([v0, v1, v2]), "v", "thrust direction")],
    )
    text = gen.gen_class_header()

    assert len(gen._precomputed) >= 1, "test setup needs a pcN_ member"
    assert "void set_v(" in text

    setter_start = text.index("void set_v(")
    setter_end = text.index("}", setter_start)
    setter_body = text[setter_start:setter_end]
    for member_name, _ in gen._precomputed:
        assert f"{member_name} =" in setter_body, (
            f"set_v body missing recomputation of {member_name}; "
            f"_affected_precomputed failed to walk element symbols"
        )


def test_ctor_and_setter_precompute_blocks_are_identical():
    """The ctor and set_mu must emit byte-identical pcN_ assignment lines."""
    Xs, Eq, mu = _mee_like_inputs_and_eq()
    gen = TychoHeaderGen("Toy", Eq, sp.Matrix(Xs), [(mu, "gm", "mu > 0.0")])
    text = gen.gen_class_header()

    # Extract pcN_ lines from the ctor body.
    lines = text.splitlines()
    ctor_start = next(i for i, l in enumerate(lines) if "Toy(double mu) {" in l)
    setter_start = next(
        i for i, l in enumerate(lines) if "void set_mu(double mu) {" in l
    )

    def _pc_lines(start):
        out = []
        for l in lines[start:]:
            if l.strip().startswith("}"):
                break
            if l.strip().startswith("pc"):
                out.append(l.strip())
        return out

    ctor_pcs = _pc_lines(ctor_start)
    setter_pcs = _pc_lines(setter_start)
    assert ctor_pcs == setter_pcs, (
        f"ctor and setter precompute blocks diverged:\n"
        f"ctor:   {ctor_pcs}\nsetter: {setter_pcs}"
    )


# ── NSDMI NaN defaults ──────────────────────────────────────────────────────


def test_scalar_param_has_nan_nsdmi_default():
    Xs, Eq, k = _toy_inputs_and_eq()
    text = _build("Toy", Eq, Xs, [(k, "gain", "k > 0.0")])

    assert "double k_ = std::numeric_limits<double>::quiet_NaN();" in text


def test_precomputed_member_has_nan_nsdmi_default():
    Xs, Eq, mu = _mee_like_inputs_and_eq()
    gen = TychoHeaderGen("Toy", Eq, sp.Matrix(Xs), [(mu, "gm", "mu > 0.0")])
    text = gen.gen_class_header()

    # Every pcN_ declared as a member must have NaN NSDMI.
    for member_name, _ in gen._precomputed:
        needle = f"double {member_name} = std::numeric_limits<double>::quiet_NaN();"
        assert needle in text, f"missing NaN default for {member_name}"


# ── _hpN_ collision guard ───────────────────────────────────────────────────


def test_hpn_name_collision_in_inputs_raises():
    """A user input symbol named ``_hp0_`` must be rejected, not shadowed."""
    x = sp.Symbol("_hp0_")
    Eq = sp.Matrix([x**2 + 1])
    gen = TychoHeaderGen("Bad", Eq, sp.Matrix([x]), [])

    with pytest.raises(ValueError, match="_hpN_ reserved namespace"):
        gen.gen_compute_impl()


def test_hpn_name_collision_in_scalar_param_raises():
    """A scalar param named ``_hp5_`` must be rejected."""
    x = sp.Symbol("x0")
    bad = sp.Symbol("_hp5_")
    Eq = sp.Matrix([bad * x])
    gen = TychoHeaderGen("Bad", Eq, sp.Matrix([x]), [(bad, "bad")])

    with pytest.raises(ValueError, match="_hpN_ reserved namespace"):
        gen.gen_compute_impl()


# ── atomic write + pow() assertion ──────────────────────────────────────────


def test_make_header_writes_atomically(tmp_path):
    """make_header must leave no ``.tmp`` sibling after success."""
    Xs, Eq, k = _toy_inputs_and_eq()
    gen = TychoHeaderGen("Toy", Eq, sp.Matrix(Xs), [(k, "gain", "k > 0.0")])
    gen.make_header(output_dir=str(tmp_path))

    assert (tmp_path / "toy.h").exists()
    assert not (tmp_path / "toy.h.tmp").exists()
    text = (tmp_path / "toy.h").read_text()
    assert "void set_k(double k)" in text

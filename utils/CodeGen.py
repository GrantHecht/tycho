import os
import re

import numpy as np
import sympy as sp
from sympy.codegen.cfunctions import Sqrt

# Indent unit used in generated code
_IND = "    "


def find_pow_expressions(s):
    expressions = []
    start_index = None
    parentheses_count = 0

    for i, c in enumerate(s):
        if s[i : i + 4] == "pow(":
            if start_index is None:
                start_index = i
            parentheses_count += 1

        elif c == "(" and start_index is not None:
            parentheses_count += 1

        elif c == ")" and start_index is not None:
            parentheses_count -= 1
            if parentheses_count == 1:
                parentheses_count -= 1
                expressions.append(s[start_index : i + 1])
                start_index = None
    return expressions


def find_innermost_pow_expressions(s):
    expressions = []
    start_index = None
    parentheses_count = 0
    pow_count = 0

    for i, c in enumerate(s):
        if s[i : i + 4] == "pow(":
            if start_index is None:
                start_index = i
            parentheses_count += 1
            pow_count += 1
        elif c == "(" and start_index is not None:
            parentheses_count += 1
        elif c == ")" and start_index is not None:
            parentheses_count -= 1
            if parentheses_count == 0:
                expressions.append(s[start_index : i + 1])
                start_index = None
                pow_count = 0
            elif pow_count > 1 and parentheses_count == pow_count - 1:
                expressions.append(s[start_index : i + 1])
                start_index = None
                pow_count -= 1

    return expressions


class TychoHeaderGen:
    def __init__(
        self,
        Name,
        F,
        Xs,
        ScalarParams=[],  # [(Symbol, Description),]
        VectorParams=[],  # [(Vec, Cppname, Description),]
        MatrixParams=[],  # [(Mat, Cppname, Description),]
        docstr="A doc string",
        namespace="tycho::astro",
        include_path="tycho/detail/vf/core/vector_function.h",
        gen_build_method=False,
    ):

        self.Name = Name
        self.namespace = namespace
        self.include_path = include_path
        self.gen_build_method = gen_build_method
        self.ninputs = len(Xs)
        self.noutputs = len(F)

        self.Func = sp.Matrix(list(F))
        self.Inputs = sp.Matrix(list(Xs))
        self.ScalarParams = ScalarParams
        self.VectorParams = VectorParams
        self.MatrixParams = MatrixParams

        self.docstr = docstr

        print("Generating Jacobian")
        self.Jac = self.Func.jacobian(self.Inputs)
        print("Finished Jacobian")
        self.LMults = sp.Matrix(sp.symbols("LM:" + str(self.noutputs)))
        print("Generating Gradient")
        self.Grad = self.Jac.transpose() * self.LMults
        print("Finished Gradient")
        print("Generating Hessian")
        self.Hess = self.Grad.jacobian(self.Inputs)
        print("Finished Hessian")

    # ── pow simplification ──────────────────────────────────────────────
    # After CSE, all sub-expressions are named x0..xN, so the simple
    # variable-name regex ([A-Za-z_]\w*) reliably matches them.

    def powsimp(self, expr):
        powcount = expr.count("pow")
        if powcount == 0:
            return expr

        # Convert integer pows to multiplies: pow(x, 2) → x*x
        for i in range(2, 16):
            regex = re.compile(rf"pow\((?P<var>[A-Za-z_]\w*)\s*,\s*{i}\s*\)")
            repl = r"\g<var>"
            for j in range(1, i):
                repl += r"*\g<var>"
            expr = re.sub(regex, repl, expr)

        # Convert pow(x, 0.5) → sqrt(x), pow(x, 1.5) → sqrt(x)*x, etc.
        for i in range(0, 7):
            regex = re.compile(rf"pow\((?P<var>[A-Za-z_]\w*)\s*,\s*{i}\.5\s*\)")
            repl = r"(sqrt(\g<var>)"
            for j in range(1, i + 1):
                repl += r"*\g<var>"
            repl += r")"
            expr = re.sub(regex, repl, expr)

        # Convert pow(x, -N) → Scalar(1.0)/(x*x*...)
        for i in range(1, 16):
            regex = re.compile(rf"pow\((?P<var>[A-Za-z_]\w*)\s*,\s*-{i}\s*\)")
            repl = r"(Scalar(1.0)/(\g<var>"
            for j in range(1, i):
                repl += r"*\g<var>"
            repl += r"))"
            expr = re.sub(regex, repl, expr)

        # Convert pow(x, -0.5) → Scalar(1.0)/sqrt(x), pow(x, -1.5) → ...
        for i in range(0, 7):
            regex = re.compile(rf"pow\((?P<var>[A-Za-z_]\w*)\s*,\s*-{i}\.5\s*\)")
            repl = r"(Scalar(1.0)/(sqrt(\g<var>)"
            for j in range(1, i + 1):
                repl += r"*\g<var>"
            repl += r"))"
            expr = re.sub(regex, repl, expr)

        # Fractional forms: pow(x, 3.0/2.0) style
        for i in range(1, 10, 2):
            regex = re.compile(
                rf"pow\((?P<var>[A-Za-z_]\w*)\s*,\s*{i}\.0/2\.0\s*\)"
            )
            repl = r"(sqrt(\g<var>)"
            for j in range(0, int((i - 1) / 2)):
                repl += r"*\g<var>"
            repl += r")"
            expr = re.sub(regex, repl, expr)

        for i in range(1, 10, 2):
            regex = re.compile(
                rf"pow\((?P<var>[A-Za-z_]\w*)\s*,\s*-{i}\.0/2\.0\s*\)"
            )
            repl = r"(Scalar(1.0)/(sqrt(\g<var>)"
            for j in range(0, int((i - 1) / 2)):
                repl += r"*\g<var>"
            repl += r"))"
            expr = re.sub(regex, repl, expr)

        # Last-resort for compound-expression pow with single occurrence
        powcount = expr.count("pow")
        commacount = expr.count(",")
        if powcount == 1 and commacount == 1:
            for i in range(0, 7):
                regex = re.compile(rf"pow\((?P<var>\s*([^,]+)),\s*-{i}\.5\s*\)")
                repl = r"(Scalar(1.0)/(sqrt(Scalar(\g<var>))"
                for j in range(1, i + 1):
                    repl += r"*Scalar(\g<var>)"
                repl += r"))"
                expr = re.sub(regex, repl, expr)

        return expr

    # ── fixup: scalar-wrap parameters, apply powsimp ────────────────────

    def fixup(self, expr):
        wrapscalar = False

        for Param, Descr in self.ScalarParams:
            Name = str(Param)
            if Name in expr:
                wrapscalar = True

        for Vec, Name, Descr in self.VectorParams:
            for i, elem in enumerate(Vec):
                symname = str(elem)
                if symname in expr:
                    replname = "({0:}[{1:}])".format(Name, i)
                    regex = re.compile(rf"\b{symname}\b")
                    expr = re.sub(symname, replname, expr)
                    wrapscalar = True

        for Mat, Name, Descr in self.MatrixParams:
            rows = len(Mat)
            cols = len(Mat[0])
            for i in range(0, rows):
                for j in range(0, cols):
                    symname = str(Mat[i, j])
                    if symname in expr:
                        replname = "({0:}({1:},{2:}))".format(Name, i, j)
                        regex = re.compile(rf"\b{symname}\b")
                        expr = re.sub(regex, replname, expr)
                        wrapscalar = True

        if wrapscalar:
            expr = "Scalar(" + expr + ")"

        # Apply pow simplification (works on CSE variable names)
        expr = self.powsimp(expr)

        return expr

    # ── class header generation ─────────────────────────────────────────

    def gen_class_header(self):
        I = _IND
        dm = "DenseDerivativeMode::Analytic"

        lines = []
        lines.append(f"struct {self.Name}")
        lines.append(f"{I}: VectorFunction<{self.Name}, {self.ninputs}, "
                      f"{self.noutputs}, {dm}, {dm}> {{")
        lines.append(f"{I}using Base = VectorFunction<{self.Name}, "
                      f"{self.ninputs}, {self.noutputs}, {dm}, {dm}>;")
        lines.append(f"{I}VF_TYPE_ALIASES(Base);")
        lines.append("")

        # Members
        for P, Descr in self.ScalarParams:
            lines.append(f"{I}double {P}; // {Descr}")
        for Vec, Name, Descr in self.VectorParams:
            vsize = len(Vec)
            lines.append(f"{I}Eigen::Matrix<double, {vsize}, 1> {Name}; // {Descr}")
        for Mat, Name, Descr in self.MatrixParams:
            rows, cols = len(Mat), len(Mat[0])
            lines.append(f"{I}Eigen::Matrix<double, {rows}, {cols}> {Name}; // {Descr}")

        lines.append(f"{I}static constexpr bool is_vectorizable = true;")
        lines.append("")

        # Constructor
        params = []
        for P, Descr in self.ScalarParams:
            params.append(f"double {P}")
        for Vec, Name, Descr in self.VectorParams:
            vsize = len(Vec)
            params.append(f"const Eigen::Matrix<double, {vsize}, 1>& {Name}")
        for Mat, Name, Descr in self.MatrixParams:
            rows, cols = len(Mat), len(Mat[0])
            params.append(f"const Eigen::Matrix<double, {rows}, {cols}>& {Name}")

        paramstr = ", ".join(params)
        assigns = []
        for P, _ in self.ScalarParams:
            assigns.append(f"{P}({P})")
        for Vec, Name, _ in self.VectorParams:
            assigns.append(f"{Name}({Name})")
        for Mat, Name, _ in self.MatrixParams:
            assigns.append(f"{Name}({Name})")
        initlist = ", ".join(assigns)

        lines.append(f"{I}{self.Name}() {{")
        lines.append(f"{I}{I}this->set_io_rows({self.ninputs}, {self.noutputs});")
        lines.append(f"{I}}}")
        if initlist:
            lines.append(f"{I}{self.Name}({paramstr})")
            lines.append(f"{I}{I}: {initlist} {{")
        else:
            lines.append(f"{I}{self.Name}({paramstr}) {{")
        lines.append(f"{I}{I}this->set_io_rows({self.ninputs}, {self.noutputs});")
        lines.append(f"{I}}}")
        lines.append("")

        return "\n".join(lines) + "\n"

    # ── compute_impl ────────────────────────────────────────────────────

    def gen_compute_impl(self):
        I = _IND
        lines = []
        lines.append(f"{I}template <class InType, class OutType>")
        lines.append(f"{I}inline void compute_impl(CVecRef<InType> x_,")
        lines.append(f"{I}{I}{I}{I}{I}{I} CVecRef<OutType> fx_) const {{")
        lines.append(f"{I}{I}typedef typename InType::Scalar Scalar;")
        lines.append(f"{I}{I}using std::cos; using std::sin; using std::sqrt;")
        lines.append(f"{I}{I}VecRef<OutType> _fx_ = fx_.const_cast_derived();")
        lines.append("")

        body_lines = self._gen_input_names()
        cses, func = sp.cse(self.Func)
        body_lines += self._gen_cselist(cses)
        body_lines.append("")
        for i, F in enumerate(func[0]):
            body_lines.append(self._assignment(f"_fx_[{i}]", sp.ccode(F), False))
        lines += [f"{I}{I}{l}" for l in body_lines]

        lines.append(f"{I}}}")
        lines.append("")
        return "\n".join(lines) + "\n"

    # ── compute_jacobian_impl ───────────────────────────────────────────

    def gen_compute_jacobian_impl(self):
        I = _IND
        lines = []
        lines.append(f"{I}template <class InType, class OutType, class JacType>")
        lines.append(f"{I}inline void compute_jacobian_impl(CVecRef<InType> x_,")
        lines.append(f"{I}{I}{I}{I}{I}{I} CVecRef<OutType> fx_,")
        lines.append(f"{I}{I}{I}{I}{I}{I} CMatRef<JacType> jx_) const {{")
        lines.append(f"{I}{I}typedef typename InType::Scalar Scalar;")
        lines.append(f"{I}{I}using std::cos; using std::sin; using std::sqrt;")
        lines.append(f"{I}{I}VecRef<OutType> _fx_ = fx_.const_cast_derived();")
        lines.append(f"{I}{I}MatRef<JacType> _jx_ = jx_.const_cast_derived();")
        lines.append("")

        body_lines = self._gen_input_names()
        cses, funcs = sp.cse([self.Func, self.Jac])
        body_lines += self._gen_cselist(cses)
        body_lines.append("")

        for i, F in enumerate(funcs[0]):
            body_lines.append(self._assignment(f"_fx_[{i}]", sp.ccode(F), False))
        body_lines.append("")

        for col in range(self.ninputs):
            for row in range(self.noutputs):
                expr = funcs[1][row, col]
                body_lines.append(self._assignment(f"_jx_({row},{col})", sp.ccode(expr), False))

        lines += [f"{I}{I}{l}" for l in body_lines]
        lines.append(f"{I}}}")
        lines.append("")
        return "\n".join(lines) + "\n"

    # ── compute_jacobian_adjointgradient_adjointhessian_impl ────────────

    def gen_compute_all(self):
        I = _IND
        lines = []
        lines.append(f"{I}template <class InType, class OutType, class JacType,")
        lines.append(f"{I}{I}{I}{I} class AdjGradType, class AdjHessType,")
        lines.append(f"{I}{I}{I}{I} class AdjVarType>")
        lines.append(f"{I}inline void compute_jacobian_adjointgradient_adjointhessian_impl(")
        lines.append(f"{I}{I}CVecRef<InType> x_,")
        lines.append(f"{I}{I}CVecRef<OutType> fx_,")
        lines.append(f"{I}{I}CMatRef<JacType> jx_,")
        lines.append(f"{I}{I}CVecRef<AdjGradType> adjgrad_,")
        lines.append(f"{I}{I}CMatRef<AdjHessType> adjhess_,")
        lines.append(f"{I}{I}CVecRef<AdjVarType> adjvars) const {{")
        lines.append(f"{I}{I}typedef typename InType::Scalar Scalar;")
        lines.append(f"{I}{I}using std::cos; using std::sin; using std::sqrt;")
        lines.append(f"{I}{I}VecRef<OutType> _fx_ = fx_.const_cast_derived();")
        lines.append(f"{I}{I}MatRef<JacType> _jx_ = jx_.const_cast_derived();")
        lines.append(f"{I}{I}VecRef<AdjGradType> _gx_ = adjgrad_.const_cast_derived();")
        lines.append(f"{I}{I}MatRef<AdjHessType> _hx_ = adjhess_.const_cast_derived();")
        lines.append("")

        body_lines = self._gen_input_names()
        body_lines += self._gen_mult_names()
        cses, funcs = sp.cse([self.Func, self.Jac, self.Grad, self.Hess])
        body_lines += self._gen_cselist(cses)
        body_lines.append("")

        for i, F in enumerate(funcs[0]):
            body_lines.append(self._assignment(f"_fx_[{i}]", sp.ccode(F), False))
        body_lines.append("")
        for col in range(self.ninputs):
            for row in range(self.noutputs):
                body_lines.append(self._assignment(
                    f"_jx_({row},{col})", sp.ccode(funcs[1][row, col]), False))
        body_lines.append("")
        for row in range(self.ninputs):
            body_lines.append(self._assignment(f"_gx_[{row}]", sp.ccode(funcs[2][row]), False))
        body_lines.append("")
        for col in range(self.ninputs):
            for row in range(self.ninputs):
                body_lines.append(self._assignment(
                    f"_hx_({row},{col})", sp.ccode(funcs[3][row, col]), False))

        lines += [f"{I}{I}{l}" for l in body_lines]
        lines.append(f"{I}}}")
        lines.append("")
        return "\n".join(lines) + "\n"

    # ── helpers ─────────────────────────────────────────────────────────

    def _assignment(self, lhs, rhs, is_scalar=False):
        """Return a single assignment line (no leading newline/indent)."""
        rhs = self.fixup(str(rhs))
        prefix = "Scalar " if is_scalar else ""
        return f"{prefix}{lhs} = {rhs};"

    def _gen_input_names(self):
        return [self._assignment(str(X), f"x_[{i}]", True)
                for i, X in enumerate(self.Inputs)]

    def _gen_mult_names(self):
        return [self._assignment(str(L), f"adjvars[{i}]", True)
                for i, L in enumerate(self.LMults)]

    def _gen_cselist(self, cses):
        return [self._assignment(str(cse[0]), sp.ccode(cse[1]), True)
                for cse in cses]

    # ── output ──────────────────────────────────────────────────────────

    def print_header(self):
        print(self._build_text())

    def make_header(self, output_dir=None):
        text = self._build_text()
        fname = self.Name + ".h"
        if output_dir:
            fname = os.path.join(output_dir, fname)
        with open(fname, "w") as f:
            f.write(text)

    def _build_text(self):
        include = '#pragma once\n#include "{}"\n\nnamespace {} {{\n\n'.format(
            self.include_path, self.namespace)
        a = self.gen_class_header()
        b = self.gen_compute_impl()
        c = self.gen_compute_jacobian_impl()
        d = self.gen_compute_all()
        e = "};\n\n} // namespace " + self.namespace + "\n"
        return include + a + b + c + d + e

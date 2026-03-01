#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class definitions of VectorExpression::Build() binding methods.
// Included from VectorFunction.h under TYCHO_PYTHON_BINDINGS.

namespace Tycho {

template <class Derived, class ExprImpl, class... Ts>
void VectorExpression<Derived, ExprImpl, Ts...>::Build(nb::module_ &m, const char *name) {
    using Base = typename VectorExpression<Derived, ExprImpl, Ts...>::Base;
    auto obj = nb::class_<Derived>(m, name).def(nb::init<Ts...>());
    Base::DenseBaseBuild(obj);
}

template <class Derived, class ExprImpl>
void VectorExpression<Derived, ExprImpl>::Build(nb::module_ &m, const char *name) {
    using Base = typename VectorExpression<Derived, ExprImpl>::Base;
    auto obj = nb::class_<Derived>(m, name).def(nb::init<>());
    Base::DenseBaseBuild(obj);
}

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS

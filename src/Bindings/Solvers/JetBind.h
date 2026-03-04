#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

#include "FunctionRegistry.h"
#include "Solvers/Jet.h"

namespace Tycho {

// Partial specialisation: when Arg = nb::args, dereference via args_proxy so
// the genfunc receives nb::detail::args_proxy (the unpacked argument sequence).
namespace detail {
template <class T>
struct JetInvoker<T, std::function<std::shared_ptr<T>(nb::detail::args_proxy)>, nb::args> {
    static inline std::shared_ptr<T>
    invoke(const std::function<std::shared_ptr<T>(nb::detail::args_proxy)> &f, const nb::args &a) {
        return f(*a);
    }
};
} // namespace detail

template <> struct TychoBind<Jet> {
    static void Build(nb::module_ &m);
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS

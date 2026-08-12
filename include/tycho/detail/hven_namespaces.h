// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// Name bridge to hven, the NLP solver library tycho builds on.
//
// hven publishes its interior-point engine in `hven::solvers`, the runtime
// support layer that engine needs (thread pool, arena allocator, sizing
// helpers, timers, ...) in `hven::utils`, and the Eigen aliases and the
// convergence-flag enum both rest on directly in `hven`. All three lived in
// tycho as `tycho::solvers` / `tycho::utils` / `tycho` before the engine moved
// out, and tycho's own code — public headers, bindings, tests, benchmarks and
// examples — names those entities either qualified as `tycho::…` or
// unqualified from inside `namespace tycho`. What follows keeps every one of
// those spellings resolving to exactly what it always did; only the
// definition's home repository changed.
//
// All three stay real tycho namespaces with content of their own:
// `tycho::solvers` owns OptimizationProblem (the VectorFunction-coupled
// convenience layer) and the NLP-backend seam, `tycho::utils` owns the
// tycho-only utilities (exception_what.h, lambda_jump_table.h, eigen_stl.h),
// and `tycho` owns everything else in the library. A tycho declaration always
// outranks a name imported into the same namespace, so nothing tycho declares
// is displaced, and none of these is an alias.

#pragma once

#include <hven/detail/drivers/psiopt_fwd.h>
#include <hven/detail/interior/typedefs/eigen_types.h>

// Declared so the two directives below stand on their own: a using-directive
// needs its target namespace to exist, and the two headers above do not
// declare either of them.
namespace hven::solvers {}
namespace hven::utils {}

namespace tycho::solvers {
using namespace ::hven::solvers;
}

namespace tycho::utils {
using namespace ::hven::utils;
}

// `hven`'s own members are imported ONE BY ONE rather than with a matching
// using-directive. A directive would import hven's nested NAMESPACE names too,
// and `tycho` already declares `solvers`, `utils` and `vf` of its own. Inside
// tycho that is harmless — the inner declaration wins — but a consumer that
// writes `using namespace tycho;` at namespace scope, which every C++ example
// and most binding translation units do, would pull tycho's `solvers` and
// hven's to the SAME lookup scope, and `solvers::Jet` would stop compiling.
// Naming the entities individually imports exactly the aliases and the enum,
// and no namespace names at all.
namespace tycho {

using ::hven::ConvergenceFlags;

using ::hven::ConstEigenRef;
using ::hven::DefaultSuperScalar;
using ::hven::DomainMatrix;
using ::hven::EigenRef;
using ::hven::IOint;
using ::hven::MaxMatrix;
using ::hven::MaxVector;
using ::hven::SuperScalarType;
using ::hven::Vector;
using ::hven::Vector1;
using ::hven::Vector10;
using ::hven::Vector11;
using ::hven::Vector12;
using ::hven::Vector13;
using ::hven::Vector14;
using ::hven::Vector2;
using ::hven::Vector3;
using ::hven::Vector4;
using ::hven::Vector5;
using ::hven::Vector6;
using ::hven::Vector7;
using ::hven::Vector8;
using ::hven::Vector9;
using ::hven::VectorX;

} // namespace tycho

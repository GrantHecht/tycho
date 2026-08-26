// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "engines_bind.h"
#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers/engines.h"
#include <hven/drivers/sqp_types.h>

#include <functional>
#include <map>
#include <string>

#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

using namespace tycho;
using namespace tycho::solvers;
using hven::solvers::QpMode;
using hven::solvers::SqpOptions;
using hven::solvers::SsnHintRule;
using hven::solvers::SsnInfeasibilityRule;
using hven::solvers::SsnSigmaRule;
using hven::solvers::StartLevel;

namespace {

// Helper macro for binding SqpOptions fields as read-write properties on
// SqpSolver. SqpOptions carries no validated setters of its own (plain
// struct, per sqp_types.h) -- every field here is a direct read/write.
#define BIND_SQP_RW(obj, pyname, field, ...)                                                       \
    obj.def_prop_rw(                                                                               \
        pyname, [](const SqpSolver &self) { return self.options().field; },                        \
        [](SqpSolver &self, decltype(self.options().field) v) {                                    \
            self.options().field = v;                                                              \
        } __VA_OPT__(, ) __VA_ARGS__)

using SqpKwSetter = std::function<void(SqpSolver &, nb::handle)>;

// One kwargs-ctor table entry per plain-value SqpOptions field -- the same
// fields BIND_SQP_RW exposes as properties (R-1: the callable-typed
// make_strategy and the nested QpOptions are NOT kwargs-exposed).
template <class T> SqpKwSetter sqp_kw_setter(T SqpOptions::*field) {
    return [field](SqpSolver &self, nb::handle v) { self.options().*field = nb::cast<T>(v); };
}

const std::map<std::string, SqpKwSetter> &sqp_kwarg_table() {
    static const std::map<std::string, SqpKwSetter> table = {
        {"kkt_tol", sqp_kw_setter(&SqpOptions::kkt_tol)},
        {"feas_tol", sqp_kw_setter(&SqpOptions::feas_tol)},
        {"max_iter", sqp_kw_setter(&SqpOptions::max_iter)},
        {"tr_init", sqp_kw_setter(&SqpOptions::tr_init)},
        {"tr_max", sqp_kw_setter(&SqpOptions::tr_max)},
        {"tr_min", sqp_kw_setter(&SqpOptions::tr_min)},
        {"enable_soc", sqp_kw_setter(&SqpOptions::enable_soc)},
        {"adaptive_mu", sqp_kw_setter(&SqpOptions::adaptive_mu)},
        {"start_level", sqp_kw_setter(&SqpOptions::start_level)},
        {"warm_full_step", sqp_kw_setter(&SqpOptions::warm_full_step)},
        {"budget_mode", sqp_kw_setter(&SqpOptions::budget_mode)},
        {"elastic_ladder_early_exit", sqp_kw_setter(&SqpOptions::elastic_ladder_early_exit)},
        {"crash_basis", sqp_kw_setter(&SqpOptions::crash_basis)},
        {"qp_mode", sqp_kw_setter(&SqpOptions::qp_mode)},
        {"ssn_prox_carry", sqp_kw_setter(&SqpOptions::ssn_prox_carry)},
        {"ssn_certify_from_face", sqp_kw_setter(&SqpOptions::ssn_certify_from_face)},
        {"ssn_sigma_rule", sqp_kw_setter(&SqpOptions::ssn_sigma_rule)},
        {"ssn_hint_rule", sqp_kw_setter(&SqpOptions::ssn_hint_rule)},
        {"ssn_infeasibility_rule", sqp_kw_setter(&SqpOptions::ssn_infeasibility_rule)},
    };
    return table;
}

} // namespace

void TychoBind<SqpSolver>::build(nb::module_ &m) {
    auto obj = nb::class_<SqpSolver>(m, "SqpSolver");
    obj.def(nb::init<>());

    obj.def(
        "__init__",
        [](SqpSolver *self, nb::kwargs kwargs) {
            new (self) SqpSolver();
            const auto &table = sqp_kwarg_table();
            for (auto item : kwargs) {
                auto name = nb::cast<std::string>(item.first);
                auto it = table.find(name);
                if (it == table.end()) {
                    throw nb::type_error(
                        fmt::format("SqpSolver: unrecognized keyword argument '{}'", name).c_str());
                }
                it->second(*self, item.second);
            }
        },
        R"doc(Construct an SqpSolver, optionally overriding SqpOptions fields by name.

Every plain-value SqpOptions field is accepted as a keyword argument
(kkt_tol, feas_tol, max_iter, tr_init, tr_max, tr_min, enable_soc,
adaptive_mu, start_level, warm_full_step, budget_mode,
elastic_ladder_early_exit, crash_basis, qp_mode, ssn_prox_carry,
ssn_certify_from_face, ssn_sigma_rule, ssn_hint_rule,
ssn_infeasibility_rule). The QP sub-options (``qp``) and the
globalization-strategy factory (``make_strategy``) are not kwargs-
exposed; use the property/attribute defaults for those.

Raises
------
TypeError
    If an unrecognized keyword argument is given, naming it.
)doc");

    BIND_SQP_RW(obj, "kkt_tol", kkt_tol);
    BIND_SQP_RW(obj, "feas_tol", feas_tol);
    BIND_SQP_RW(obj, "max_iter", max_iter);
    BIND_SQP_RW(obj, "tr_init", tr_init);
    BIND_SQP_RW(obj, "tr_max", tr_max);
    BIND_SQP_RW(obj, "tr_min", tr_min);
    BIND_SQP_RW(obj, "enable_soc", enable_soc);
    BIND_SQP_RW(obj, "adaptive_mu", adaptive_mu);
    BIND_SQP_RW(obj, "start_level", start_level);
    BIND_SQP_RW(obj, "warm_full_step", warm_full_step);
    BIND_SQP_RW(obj, "budget_mode", budget_mode);
    BIND_SQP_RW(obj, "elastic_ladder_early_exit", elastic_ladder_early_exit);
    BIND_SQP_RW(obj, "crash_basis", crash_basis);
    BIND_SQP_RW(obj, "qp_mode", qp_mode);
    BIND_SQP_RW(obj, "ssn_prox_carry", ssn_prox_carry);
    BIND_SQP_RW(obj, "ssn_certify_from_face", ssn_certify_from_face);
    BIND_SQP_RW(obj, "ssn_sigma_rule", ssn_sigma_rule);
    BIND_SQP_RW(obj, "ssn_hint_rule", ssn_hint_rule);
    BIND_SQP_RW(obj, "ssn_infeasibility_rule", ssn_infeasibility_rule);

    nb::enum_<StartLevel>(m, "StartLevel",
                          "How much of a previous solve's state a caller intends to feed into "
                          "the next one -- kCold ignores it, kSeeded trusts values but not "
                          "provenance, kWarm additionally trusts the globalization state, kHot "
                          "additionally reuses a factorization.")
        .value("kCold", StartLevel::kCold)
        .value("kSeeded", StartLevel::kSeeded)
        .value("kWarm", StartLevel::kWarm)
        .value("kHot", StartLevel::kHot);

    nb::enum_<QpMode>(m, "QpMode", "Which QP subproblem solver the SQP driver dispatches to.")
        .value("kWalk", QpMode::kWalk)
        .value("kSsn", QpMode::kSsn);

    nb::enum_<SsnSigmaRule>(m, "SsnSigmaRule",
                            "How the SSN proximal/Levenberg-Marquardt shift sigma is sized.")
        .value("kLadder", SsnSigmaRule::kLadder)
        .value("kResidualArmed", SsnSigmaRule::kResidualArmed)
        .value("kResidualAlways", SsnSigmaRule::kResidualAlways);

    nb::enum_<SsnHintRule>(m, "SsnHintRule", "What protects the SSN hinted first step.")
        .value("kIterationZeroFree", SsnHintRule::kIterationZeroFree)
        .value("kWatchdog", SsnHintRule::kWatchdog);

    nb::enum_<SsnInfeasibilityRule>(m, "SsnInfeasibilityRule",
                                    "What turns an SSN infeasibility suspicion into an exit.")
        .value("kSymptoms", SsnInfeasibilityRule::kSymptoms)
        .value("kFarkasGated", SsnInfeasibilityRule::kFarkasGated);

    // -----------------------------------------------------------------
    // IpoptSolver -- Ipopt as a peer engine handle.
    // -----------------------------------------------------------------
    nb::class_<IpoptSolver>(
        m, "IpoptSolver",
        "Ipopt as a peer engine handle. Constructible only when the backend is compiled in "
        "(ENABLE_IPOPT); otherwise raises RuntimeError.")
        .def(nb::init<>())
        .def_prop_rw(
            "options", [](const IpoptSolver &self) { return self.options(); },
            [](IpoptSolver &self, const std::map<std::string, std::string> &opts) {
                self.options() = opts;
            },
            "String key/value options forwarded verbatim to Ipopt. Reading this attribute "
            "returns a copy; assign a whole dict to change it.");
}

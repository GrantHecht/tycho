// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "solve_types_bind.h"
#include "tycho/detail/hven_namespaces.h"
#include "tycho/detail/solvers/solve_types.h"
#include <hven/warmstart/warm_start_data.h>

#include <cstddef>
#include <cstring>
#include <functional>
#include <string_view>
#include <tuple>

#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

using namespace tycho;
using namespace tycho::solvers;
using hven::solvers::DeclarationKey;
using hven::solvers::WarmExtension;
using hven::solvers::WarmStartData;

namespace {

// bytes <-> std::vector<std::byte>, used for WarmExtension::payload_ and for
// WarmStartData's own byte-serialized pickling below.
nb::bytes bytes_from_byte_vector(const std::vector<std::byte> &v) {
    return nb::bytes(reinterpret_cast<const char *>(v.data()), v.size());
}

std::vector<std::byte> byte_vector_from_bytes(const nb::bytes &b) {
    std::vector<std::byte> v(b.size());
    if (!v.empty()) {
        std::memcpy(v.data(), b.data(), b.size());
    }
    return v;
}

// A boost::hash_combine-shaped mixer, used to build __hash__ for the value
// types below from the same fields their __eq__ compares (or, for
// WarmStartData, the cheap documented subset -- see its class docstring).
std::size_t hash_combine(std::size_t seed, std::size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::size_t hash_warm_extension(const WarmExtension &self) {
    std::size_t h = std::hash<std::string>{}(self.tag_);
    std::string_view payload_view(reinterpret_cast<const char *>(self.payload_.data()),
                                  self.payload_.size());
    return hash_combine(h, std::hash<std::string_view>{}(payload_view));
}

std::size_t hash_warm_start_data(const WarmStartData &self) {
    std::size_t h = static_cast<std::size_t>(self.structure_key_.digest());
    h = hash_combine(h, static_cast<std::size_t>(self.primal_.size()));
    h = hash_combine(h, static_cast<std::size_t>(self.eq_lmults_.size()));
    h = hash_combine(h, static_cast<std::size_t>(self.iq_lmults_.size()));
    h = hash_combine(h, static_cast<std::size_t>(self.bound_lmults_.size()));
    h = hash_combine(h, static_cast<std::size_t>(self.extensions_.size()));
    return h;
}

} // namespace

void TychoBind<SolveResult>::build(nb::module_ &m) {
    nb::enum_<Mode>(m, "Mode",
                    "Which objective a solve() call pursued: drive to optimality, or only to "
                    "feasibility.")
        .value("Optimal", Mode::Optimal)
        .value("Feasible", Mode::Feasible);

    // -------------------------------------------------------------------
    // DeclarationKey -- the warm-start currency's declaration-identity stamp.
    // -------------------------------------------------------------------
    nb::class_<DeclarationKey>(m, "DeclarationKey",
                               "The declared problem's identity stamp a WarmStartData was taken "
                               "under. Engine-independent and treatment-independent by "
                               "construction -- see hven's structure_identity.h for the exact "
                               "coverage.")
        .def(nb::init<>())
        .def_rw("declaration_digest", &DeclarationKey::declaration_digest_)
        .def_rw("bound_digest", &DeclarationKey::bound_digest_)
        .def("digest", &DeclarationKey::digest,
             "The two conjuncts folded into one value, for diagnostics. Comparing folded "
             "digests is weaker than comparing keys -- prefer ==.")
        .def(
            "__eq__", [](const DeclarationKey &a, const DeclarationKey &b) { return a == b; },
            nb::is_operator())
        .def("__hash__", [](const DeclarationKey &self) { return self.digest(); })
        .def("__getstate__",
             [](const DeclarationKey &self) {
                 return std::make_tuple(self.declaration_digest_, self.bound_digest_);
             })
        .def("__setstate__",
             [](DeclarationKey &self, std::tuple<std::uint64_t, std::uint64_t> state) {
                 new (&self) DeclarationKey{std::get<0>(state), std::get<1>(state)};
             });

    // -------------------------------------------------------------------
    // WarmExtension -- one opaque engine extension (tag + payload bytes).
    // -------------------------------------------------------------------
    nb::class_<WarmExtension>(
        m, "WarmExtension",
        "One opaque engine extension carried by a WarmStartData: a tag naming the producer "
        "and meaning, and payload bytes only that producer interprets.")
        .def(
            "__init__",
            [](WarmExtension *self, std::string tag, nb::bytes payload) {
                new (self) WarmExtension{std::move(tag), byte_vector_from_bytes(payload)};
            },
            nb::arg("tag"), nb::arg("payload"))
        .def_rw("tag", &WarmExtension::tag_)
        .def_prop_rw(
            "payload",
            [](const WarmExtension &self) { return bytes_from_byte_vector(self.payload_); },
            [](WarmExtension &self, nb::bytes payload) {
                self.payload_ = byte_vector_from_bytes(payload);
            })
        .def(
            "__eq__", [](const WarmExtension &a, const WarmExtension &b) { return a == b; },
            nb::is_operator())
        .def("__hash__", &hash_warm_extension)
        .def("__getstate__",
             [](const WarmExtension &self) {
                 return std::make_tuple(self.tag_, bytes_from_byte_vector(self.payload_));
             })
        .def("__setstate__", [](WarmExtension &self, std::tuple<std::string, nb::bytes> state) {
            new (&self)
                WarmExtension{std::get<0>(state), byte_vector_from_bytes(std::get<1>(state))};
        });

    // -------------------------------------------------------------------
    // WarmStartData -- the warm-start currency.
    //
    // Pickled via hven's own serialize()/deserialize() (bit-exact, versioned
    // byte form) rather than a field-by-field tuple: the byte form is the
    // currency's own versioned representation, so it survives a field being
    // added on the solver-library side.
    // -------------------------------------------------------------------
    nb::class_<WarmStartData>(
        m, "WarmStartData",
        "The engine-neutral warm-start currency: a declared-space primal/dual core, the "
        "declaration-identity stamp it was taken under, and opaque engine extensions. Value-"
        "semantic and comparable; nothing here interprets an extension's bytes.")
        .def(nb::init<>())
        .def_rw("primal", &WarmStartData::primal_)
        .def_rw("eq_lmults", &WarmStartData::eq_lmults_)
        .def_rw("iq_lmults", &WarmStartData::iq_lmults_)
        .def_rw("bound_lmults", &WarmStartData::bound_lmults_)
        .def_rw("structure_key", &WarmStartData::structure_key_)
        .def_rw("extensions", &WarmStartData::extensions_)
        .def(
            "__eq__", [](const WarmStartData &a, const WarmStartData &b) { return a == b; },
            nb::is_operator())
        .def("__hash__", &hash_warm_start_data,
             "Hashes a cheap, stable subset consistent with == -- the declaration-identity "
             "stamp's digest plus the four block sizes -- not the full primal/dual/extension "
             "content. Two equal WarmStartData values always hash equal; two unequal values "
             "sharing that subset (e.g. differing only in payload values) hash equal too, "
             "which is a legal (if collision-prone) hash under Python's contract.")
        .def("__getstate__",
             [](const WarmStartData &self) {
                 auto bytes = hven::solvers::serialize(self);
                 return nb::bytes(reinterpret_cast<const char *>(bytes.data()), bytes.size());
             })
        .def("__setstate__", [](WarmStartData &self, const nb::bytes &state) {
            std::vector<std::byte> buf(state.size());
            if (!buf.empty()) {
                std::memcpy(buf.data(), state.data(), state.size());
            }
            new (&self) WarmStartData(hven::solvers::deserialize(buf));
        });

    // -------------------------------------------------------------------
    // StageResult -- one solver stage's outcome.
    // -------------------------------------------------------------------
    nb::class_<StageResult>(
        m, "StageResult",
        "One solver stage's outcome: which stage it was, which engine ran it, and the "
        "numbers that describe how it finished.")
        .def(nb::init<>())
        .def_ro("role", &StageResult::role_, "\"presolve\" | \"main\" | \"polish\".")
        .def_ro("engine_name", &StageResult::engine_name_,
                "Engine class name, e.g. \"InteriorPointSolver\".")
        .def_ro("mode", &StageResult::mode_,
                "Which objective this stage pursued: a presolve stage always runs "
                "Mode.Feasible, a polish stage always Mode.Optimal, a main stage whichever "
                "mode the call asked for. This is what tells a later solve whether the "
                "stage's multipliers price the objective it is about to minimize.")
        .def_ro("flag", &StageResult::flag_)
        .def_ro("iterations", &StageResult::iterations_)
        .def_ro("objective", &StageResult::objective_, "Caller's scale.")
        .def_ro("kkt_residual", &StageResult::kkt_residual_)
        .def_ro("eq_violation", &StageResult::eq_violation_, "Max-norm.")
        .def_ro("iq_violation", &StageResult::iq_violation_, "Max-norm.")
        .def_ro("wall_time_s", &StageResult::wall_time_s_)
        .def_prop_ro(
            "engine_details", [](const StageResult &self) { return self.engine_details_; },
            "Engine-specific numeric annex, as a plain dict.")
        .def_prop_ro(
            "engine_notes", [](const StageResult &self) { return self.engine_notes_; },
            "Engine-specific string annex, as a plain dict.")
        .def("__getstate__",
             [](const StageResult &self) {
                 return std::make_tuple(self.role_, self.engine_name_, self.mode_, self.flag_,
                                        self.iterations_, self.objective_, self.kkt_residual_,
                                        self.eq_violation_, self.iq_violation_, self.wall_time_s_,
                                        self.engine_details_, self.engine_notes_);
             })
        .def("__setstate__",
             [](StageResult &self,
                std::tuple<std::string, std::string, Mode, tycho::ConvergenceFlags, int, double,
                           double, double, double, double, std::map<std::string, double>,
                           std::map<std::string, std::string>>
                    state) {
                 new (&self) StageResult{};
                 self.role_ = std::get<0>(state);
                 self.engine_name_ = std::get<1>(state);
                 self.mode_ = std::get<2>(state);
                 self.flag_ = std::get<3>(state);
                 self.iterations_ = std::get<4>(state);
                 self.objective_ = std::get<5>(state);
                 self.kkt_residual_ = std::get<6>(state);
                 self.eq_violation_ = std::get<7>(state);
                 self.iq_violation_ = std::get<8>(state);
                 self.wall_time_s_ = std::get<9>(state);
                 self.engine_details_ = std::get<10>(state);
                 self.engine_notes_ = std::get<11>(state);
             });

    // -------------------------------------------------------------------
    // PhaseResult -- one OCP phase's slice of a solve.
    // -------------------------------------------------------------------
    nb::class_<PhaseResult>(
        m, "PhaseResult",
        "One OCP phase's slice of a solve, keyed the same way as the OCP itself (index == 0 "
        "for a single Phase, no OCP). Every field is a snapshot taken at solve time.")
        .def(nb::init<>())
        .def_ro("index", &PhaseResult::index_)
        .def_ro("var_start", &PhaseResult::var_start_)
        .def_ro("var_count", &PhaseResult::var_count_)
        .def_ro("eq_start", &PhaseResult::eq_start_)
        .def_ro("eq_count", &PhaseResult::eq_count_)
        .def_ro("iq_start", &PhaseResult::iq_start_)
        .def_ro("iq_count", &PhaseResult::iq_count_)
        .def_prop_ro("eq_lmults", [](const PhaseResult &self) { return self.eq_lmults_; })
        .def_prop_ro("iq_lmults", [](const PhaseResult &self) { return self.iq_lmults_; })
        .def_prop_ro(
            "bound_lmults", [](const PhaseResult &self) { return self.bound_lmults_; },
            "Declared-space signed z = zL - zU slice.")
        .def("__getstate__",
             [](const PhaseResult &self) {
                 return std::make_tuple(self.index_, self.var_start_, self.var_count_,
                                        self.eq_start_, self.eq_count_, self.iq_start_,
                                        self.iq_count_, self.eq_lmults_, self.iq_lmults_,
                                        self.bound_lmults_);
             })
        .def("__setstate__",
             [](PhaseResult &self, std::tuple<int, int, int, int, int, int, int, Eigen::VectorXd,
                                              Eigen::VectorXd, Eigen::VectorXd>
                                       state) {
                 new (&self) PhaseResult{};
                 self.index_ = std::get<0>(state);
                 self.var_start_ = std::get<1>(state);
                 self.var_count_ = std::get<2>(state);
                 self.eq_start_ = std::get<3>(state);
                 self.eq_count_ = std::get<4>(state);
                 self.iq_start_ = std::get<5>(state);
                 self.iq_count_ = std::get<6>(state);
                 self.eq_lmults_ = std::get<7>(state);
                 self.iq_lmults_ = std::get<8>(state);
                 self.bound_lmults_ = std::get<9>(state);
             });

    // -------------------------------------------------------------------
    // SolveResult -- what a solve() call hands back.
    // -------------------------------------------------------------------
    nb::class_<SolveResult>(
        m, "SolveResult",
        "What a solve() call hands back: the deciding convergence flag, every stage that "
        "ran, every OCP phase's slice (empty for a bare VF problem), and the declared-space "
        "warm-start currency taken from the final deciding stage.")
        .def(nb::init<>())
        .def_ro("flag", &SolveResult::flag_)
        .def_prop_ro(
            "stages", [](const SolveResult &self) { return self.stages_; },
            "Run order: presolve?, main, polish?. A fresh copy on every access.")
        .def_prop_ro(
            "phases", [](const SolveResult &self) { return self.phases_; },
            "Index-keyed like the OCP; empty for a bare VF problem. A fresh copy on every "
            "access.")
        .def_prop_ro(
            "warm", [](const SolveResult &self) { return self.warm_; },
            "Declared-space warm-start payload from the final deciding stage. A fresh copy "
            "on every access.")
        .def_prop_ro(
            "structure_key", [](const SolveResult &self) { return self.structure_key_; },
            "The declaration-identity stamp warm was taken under.")
        .def("converged", &SolveResult::converged,
             "True for CONVERGED or ACCEPTABLE -- ACCEPTABLE is convergence to the "
             "acceptable tolerance ladder, still a caller-usable answer.")
        .def("__bool__", &SolveResult::converged)
        .def("objective", &SolveResult::objective)
        .def("iterations", &SolveResult::iterations)
        .def("__getstate__",
             [](const SolveResult &self) {
                 return std::make_tuple(self.flag_, self.stages_, self.phases_, self.warm_,
                                        self.structure_key_);
             })
        .def("__setstate__", [](SolveResult &self,
                                std::tuple<tycho::ConvergenceFlags, std::vector<StageResult>,
                                           std::vector<PhaseResult>, WarmStartData, DeclarationKey>
                                    state) {
            new (&self) SolveResult{};
            self.flag_ = std::get<0>(state);
            self.stages_ = std::get<1>(state);
            self.phases_ = std::get<2>(state);
            self.warm_ = std::get<3>(state);
            self.structure_key_ = std::get<4>(state);
        });
}

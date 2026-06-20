// =============================================================================
// Tycho — Builder API: Variable Registry
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#pragma once

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <fmt/format.h>

namespace tycho {

/// @ingroup optimal_control
/// @brief Maps string variable names to XtUP-space indices for the builder API.
///
/// XtUP-space is the concatenated layout @c [X_0..X_{n-1}, t, U_0..U_{m-1}, P_0..P_{k-1}]
/// used as the input vector for ODE functions. After construction and
/// registration of names/groups, the registry is populated during setup and
/// then used read-only.
/// @note Thread-safe for concurrent reads after registration is complete; not
///       thread-safe for concurrent writes or mixed read/write access.
class VarRegistry {
  public:
    /// @brief Construct an empty registry with zero dimensions.
    VarRegistry() = default;

    /// @brief Construct a registry for the given ODE dimensions.
    /// @param xvars  Number of state variables (must be non-negative).
    /// @param uvars  Number of control variables (must be non-negative).
    /// @param pvars  Number of parameter variables (must be non-negative).
    /// @throws std::invalid_argument if any dimension is negative.
    VarRegistry(int xvars, int uvars, int pvars) : xvars_(xvars), uvars_(uvars), pvars_(pvars) {
        if (xvars < 0)
            throw std::invalid_argument(
                fmt::format("VarRegistry: xvars must be non-negative (got {})", xvars));
        if (uvars < 0)
            throw std::invalid_argument(
                fmt::format("VarRegistry: uvars must be non-negative (got {})", uvars));
        if (pvars < 0)
            throw std::invalid_argument(
                fmt::format("VarRegistry: pvars must be non-negative (got {})", pvars));
    }

    // ── Registration ────────────────────────────────────────────────────

    /// @brief Register a single variable name mapped to one XtUP-space index.
    ///
    /// @param name        Variable name (must be non-empty and not already registered).
    /// @param xtup_index  Zero-based index in @c [0, xtup_size()).
    /// @throws std::invalid_argument if @p name is empty, already registered,
    ///         or @p xtup_index is outside @c [0, xtup_size()).
    void add_name(const std::string &name, int xtup_index) {
        check_name_not_empty(name, "add_name");
        check_not_registered(name);
        check_index(xtup_index);
        Eigen::VectorXi idx(1);
        idx[0] = xtup_index;
        map_.emplace(name, std::move(idx));
    }

    /// @brief Register a group name mapped to a contiguous range in XtUP space.
    ///
    /// The group maps @p name to the index sequence
    /// @c [start, start+1, ..., start+count-1].
    ///
    /// @param name   Group name (must be non-empty and not already registered).
    /// @param start  Zero-based start index in @c [0, xtup_size()).
    /// @param count  Number of contiguous variables (must be positive).
    /// @throws std::invalid_argument if @p name is empty or already registered,
    ///         @p count is not positive, or the range exceeds @c xtup_size().
    void add_group(const std::string &name, int start, int count) {
        check_name_not_empty(name, "add_group");
        check_not_registered(name);
        if (count <= 0) {
            throw std::invalid_argument(
                fmt::format("VarRegistry::add_group: count must be positive (got {})", count));
        }
        check_index(start);
        check_index(start + count - 1);
        Eigen::VectorXi idx(count);
        for (int i = 0; i < count; ++i)
            idx[i] = start + i;
        map_.emplace(name, std::move(idx));
    }

    /// @brief Register a group name as the ordered union of previously registered names.
    ///
    /// Each element of @p members is resolved to its index vector and the
    /// results are concatenated in order.  All member names must already be
    /// registered.
    ///
    /// @param name     Group name (must be non-empty and not already registered).
    /// @param members  Previously registered names whose indices are concatenated.
    /// @throws std::invalid_argument if @p name is empty, already registered,
    ///         @p members is empty, or any member name is not registered.
    void add_group(const std::string &name, std::initializer_list<std::string> members) {
        check_name_not_empty(name, "add_group");
        check_not_registered(name);
        if (members.size() == 0) {
            throw std::invalid_argument(fmt::format(
                "VarRegistry::add_group: member list for '{}' must not be empty", name));
        }
        int total = 0;
        for (const auto &m : members)
            total += resolve(m).size();

        Eigen::VectorXi idx(total);
        int pos = 0;
        for (const auto &m : members) {
            auto sub = resolve(m);
            for (int i = 0; i < sub.size(); ++i)
                idx[pos++] = sub[i];
        }
        map_.emplace(name, std::move(idx));
    }

    // ── Resolution ──────────────────────────────────────────────────────

    /// @brief Resolve a single registered name to its XtUP index vector.
    /// @param name  The variable or group name to look up.
    /// @return The index vector associated with @p name.
    /// @throws std::invalid_argument if @p name is not registered.
    Eigen::VectorXi resolve(const std::string &name) const {
        auto it = map_.find(name);
        if (it == map_.end()) {
            throw std::invalid_argument(
                fmt::format("VarRegistry::resolve: unknown variable '{}'", name));
        }
        return it->second;
    }

    /// @brief Resolve multiple names, concatenating their index vectors in order.
    /// @param names  Brace-enclosed list of registered variable or group names.
    /// @return A single index vector formed by concatenating the index vectors
    ///         of all @p names in order.
    /// @throws std::invalid_argument if any name in @p names is not registered.
    Eigen::VectorXi resolve(std::initializer_list<std::string> names) const {
        return resolve_impl(names);
    }

    /// @brief Resolve a @c std::vector of names, concatenating their index vectors in order.
    /// @param names  Registered variable or group names to resolve.
    /// @return A single index vector formed by concatenating the index vectors
    ///         of all @p names in order.
    /// @throws std::invalid_argument if any name in @p names is not registered.
    Eigen::VectorXi resolve(const std::vector<std::string> &names) const {
        return resolve_impl(names);
    }

    // ── Convenience builders ────────────────────────────────────────────

    /// @brief Build an XtUP-sized input vector from name-value pairs.
    ///
    /// Each pair assigns a scalar value to the XtUP index associated with the
    /// given name (which must map to exactly one index).  Unassigned entries
    /// default to @c 0.0.
    ///
    /// @param scalar_vals  Name/value pairs; each name must map to a single index.
    /// @return A zero-initialised XtUP-sized vector with the specified entries set.
    /// @throws std::invalid_argument if a name is not registered, maps to more
    ///         than one index, or is assigned more than once.
    Eigen::VectorXd
    make_input(std::initializer_list<std::pair<std::string, double>> scalar_vals) const {
        Eigen::VectorXd vec = Eigen::VectorXd::Zero(xtup_size());
        std::vector<bool> assigned(xtup_size(), false);
        for (const auto &[name, val] : scalar_vals) {
            auto idx = resolve(name);
            if (idx.size() != 1) {
                throw std::invalid_argument(fmt::format(
                    "VarRegistry::make_input: '{}' maps to {} indices, expected 1 for scalar "
                    "assignment",
                    name, idx.size()));
            }
            if (assigned[idx[0]]) {
                throw std::invalid_argument(fmt::format(
                    "VarRegistry::make_input: duplicate assignment to index {} (variable '{}')",
                    idx[0], name));
            }
            vec[idx[0]] = val;
            assigned[idx[0]] = true;
        }
        return vec;
    }

    /// @brief Build an XtUP-sized variable-scaling vector from name-value pairs.
    ///
    /// Each pair assigns a scaling factor to every XtUP index associated with
    /// the given name (which may map to one or more indices, e.g. for a group).
    /// Unassigned entries default to @c 1.0.
    ///
    /// @param unit_vals  Name/scale pairs applied to all indices of each name.
    /// @return A ones-initialised XtUP-sized vector with the specified entries set.
    /// @throws std::invalid_argument if a name is not registered or any
    ///         resolved index is assigned more than once.
    Eigen::VectorXd
    make_units(std::initializer_list<std::pair<std::string, double>> unit_vals) const {
        Eigen::VectorXd vec = Eigen::VectorXd::Ones(xtup_size());
        std::vector<bool> assigned(xtup_size(), false);
        for (const auto &[name, val] : unit_vals) {
            auto idx = resolve(name);
            for (int i = 0; i < idx.size(); ++i) {
                if (assigned[idx[i]]) {
                    throw std::invalid_argument(fmt::format(
                        "VarRegistry::make_units: duplicate assignment to index {} (variable '{}')",
                        idx[i], name));
                }
                vec[idx[i]] = val;
                assigned[idx[i]] = true;
            }
        }
        return vec;
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// @brief Number of state variables. @return The state-variable count.
    int xvars() const { return xvars_; }
    /// @brief Number of control variables. @return The control-variable count.
    int uvars() const { return uvars_; }
    /// @brief Number of parameter variables. @return The parameter-variable count.
    int pvars() const { return pvars_; }
    /// @brief Total XtUP-space size. @return The packed input-vector length.
    int xtup_size() const { return xvars_ + 1 + uvars_ + pvars_; }
    /// @brief Whether no names are registered. @return True if the registry is empty.
    bool empty() const { return map_.empty(); }
    /// @brief Whether a name is registered.
    /// @param name  The variable name to look up.
    /// @return True if @p name is registered.
    bool contains(const std::string &name) const { return map_.count(name) > 0; }

    /// @brief Read-only access to the full map of registered name-to-index entries.
    ///
    /// Used internally by @ref ODE::phase() to populate the @c GenericODE's
    /// flat variable-name index map.  Also useful for inspecting registered
    /// names programmatically.
    ///
    /// @return Const reference to the underlying @c unordered_map keyed by
    ///         variable/group name.
    const std::unordered_map<std::string, Eigen::VectorXi> &entries() const { return map_; }

  private:
    int xvars_ = 0;
    int uvars_ = 0;
    int pvars_ = 0;
    std::unordered_map<std::string, Eigen::VectorXi> map_;

    template <typename Range> Eigen::VectorXi resolve_impl(const Range &names) const {
        int total = 0;
        for (const auto &n : names)
            total += resolve(n).size();

        Eigen::VectorXi idx(total);
        int pos = 0;
        for (const auto &n : names) {
            auto sub = resolve(n);
            for (int i = 0; i < sub.size(); ++i)
                idx[pos++] = sub[i];
        }
        return idx;
    }

    void check_name_not_empty(const std::string &name, const char *method) const {
        if (name.empty()) {
            throw std::invalid_argument(
                fmt::format("VarRegistry::{}: variable name must not be empty", method));
        }
    }

    void check_not_registered(const std::string &name) const {
        if (map_.count(name) > 0) {
            throw std::invalid_argument(
                fmt::format("VarRegistry: name '{}' already registered", name));
        }
    }

    void check_index(int idx) const {
        if (idx < 0 || idx >= xtup_size()) {
            throw std::invalid_argument(
                fmt::format("VarRegistry: index {} out of XtUP range [0, {})", idx, xtup_size()));
        }
    }
};

} // namespace tycho

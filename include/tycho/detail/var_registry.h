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

namespace Tycho {

/// Maps string variable names to XtUP-space indices.
///
/// After construction and registration of names/groups, the registry is
/// effectively immutable — all lookups are const. Thread-safe for reads.
class VarRegistry {
  public:
    VarRegistry() = default;

    VarRegistry(int xvars, int uvars, int pvars) : xvars_(xvars), uvars_(uvars), pvars_(pvars) {}

    // ── Registration ────────────────────────────────────────────────────

    /// Map a single name to one XtUP-space index.
    void add_name(const std::string &name, int xtup_index) {
        check_not_registered(name);
        check_index(xtup_index);
        Eigen::VectorXi idx(1);
        idx[0] = xtup_index;
        map_.emplace(name, std::move(idx));
    }

    /// Map a name to a contiguous range [start, start+count-1] in XtUP space.
    void add_group(const std::string &name, int start, int count) {
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

    /// Map a group name to the union of previously registered names.
    void add_group(const std::string &name, std::initializer_list<std::string> members) {
        check_not_registered(name);
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

    /// Resolve a single name to its index vector.
    Eigen::VectorXi resolve(const std::string &name) const {
        auto it = map_.find(name);
        if (it == map_.end()) {
            throw std::invalid_argument(
                fmt::format("VarRegistry::resolve: unknown variable '{}'", name));
        }
        return it->second;
    }

    /// Resolve multiple names, concatenating their indices.
    Eigen::VectorXi resolve(std::initializer_list<std::string> names) const {
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

    /// Resolve a vector of names.
    Eigen::VectorXi resolve(const std::vector<std::string> &names) const {
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

    // ── Convenience builders ────────────────────────────────────────────

    /// Build a state vector from name-value pairs.  Unset entries default to 0.
    Eigen::VectorXd
    make_input(std::initializer_list<std::pair<std::string, double>> scalar_vals) const {
        Eigen::VectorXd vec = Eigen::VectorXd::Zero(xtup_size());
        for (const auto &[name, val] : scalar_vals) {
            auto idx = resolve(name);
            if (idx.size() != 1) {
                throw std::invalid_argument(fmt::format(
                    "VarRegistry::make_input: '{}' maps to {} indices, expected 1 for scalar "
                    "assignment",
                    name, idx.size()));
            }
            vec[idx[0]] = val;
        }
        return vec;
    }

    /// Build a units vector from name-value pairs.  Unset entries default to 1.
    Eigen::VectorXd
    make_units(std::initializer_list<std::pair<std::string, double>> unit_vals) const {
        Eigen::VectorXd vec = Eigen::VectorXd::Ones(xtup_size());
        for (const auto &[name, val] : unit_vals) {
            auto idx = resolve(name);
            for (int i = 0; i < idx.size(); ++i)
                vec[idx[i]] = val;
        }
        return vec;
    }

    // ── Accessors ───────────────────────────────────────────────────────

    int xvars() const { return xvars_; }
    int uvars() const { return uvars_; }
    int pvars() const { return pvars_; }
    int xtup_size() const { return xvars_ + 1 + uvars_ + pvars_; }
    bool empty() const { return map_.empty(); }
    bool contains(const std::string &name) const { return map_.count(name) > 0; }

  private:
    int xvars_ = 0;
    int uvars_ = 0;
    int pvars_ = 0;
    std::unordered_map<std::string, Eigen::VectorXi> map_;

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

} // namespace Tycho

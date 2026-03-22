// =============================================================================
// New file in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt)
//
// Minimal flat map backed by std::vector<std::pair<Key, Value>>.
// Memcpy-relocatable (safe for TypeStorage SBO memcpy-move):
// only member is std::vector (stores {pointer, size, capacity},
// no self-referential pointers). Elements live on the heap.
// Unsorted, insertion-order preserved, linear-scan lookup.
// Intended for small collections (N < ~20) where cache locality
// beats tree/hash overhead.
// =============================================================================

#pragma once

#include <algorithm>
#include <stdexcept>

#include <utility>
#include <vector>

namespace Tycho {

template <typename Key, typename Value> class FlatMap {
    std::vector<std::pair<Key, Value>> data_;

    auto find_it(const Key &k) {
        return std::find_if(data_.begin(), data_.end(),
                            [&k](const std::pair<Key, Value> &p) { return p.first == k; });
    }
    auto find_it(const Key &k) const {
        return std::find_if(data_.begin(), data_.end(),
                            [&k](const std::pair<Key, Value> &p) { return p.first == k; });
    }

  public:
    using iterator = typename std::vector<std::pair<Key, Value>>::iterator;
    using const_iterator = typename std::vector<std::pair<Key, Value>>::const_iterator;
    using value_type = std::pair<Key, Value>;

    FlatMap() = default;

    const Value &at(const Key &k) const {
        auto it = find_it(k);
        if (it == data_.end())
            throw std::out_of_range("FlatMap::at: key not found: " + k);
        return it->second;
    }

    Value &at(const Key &k) {
        auto it = find_it(k);
        if (it == data_.end())
            throw std::out_of_range("FlatMap::at: key not found: " + k);
        return it->second;
    }

    bool contains(const Key &k) const { return find_it(k) != data_.end(); }

    Value &operator[](const Key &k) {
        auto it = find_it(k);
        if (it != data_.end())
            return it->second;
        data_.emplace_back(k, Value{});
        return data_.back().second;
    }

    void insert(const Key &k, const Value &v) {
        if (contains(k))
            throw std::invalid_argument("FlatMap::insert: duplicate key");
        data_.emplace_back(k, v);
    }

    void clear() { data_.clear(); }

    [[nodiscard]] bool empty() const { return data_.empty(); }
    [[nodiscard]] std::size_t size() const { return data_.size(); }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }
};

} // namespace Tycho

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

#include <fmt/format.h>
#include <utility>
#include <vector>

namespace tycho::utils {

/// @brief Insertion-ordered flat map backed by a `std::vector` of key-value pairs.
///
/// Lookup is O(N) linear scan. Intended for small collections (fewer than ~20 entries)
/// where cache locality outweighs tree or hash overhead. The underlying `std::vector` is
/// trivially relocatable, making `FlatMap` safe for use inside `TypeStorage` inline storage.
///
/// @tparam Key   Key type. Must be equality-comparable.
/// @tparam Value Mapped value type.
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
    /// @brief Mutable iterator over key-value pairs.
    using iterator = typename std::vector<std::pair<Key, Value>>::iterator;
    /// @brief Const iterator over key-value pairs.
    using const_iterator = typename std::vector<std::pair<Key, Value>>::const_iterator;
    /// @brief Key-value pair type stored in the map.
    using value_type = std::pair<Key, Value>;

    /// @brief Construct an empty map.
    FlatMap() = default;

    /// @brief Return a const reference to the value mapped to `k`.
    /// @throws std::out_of_range if `k` is not present.
    const Value &at(const Key &k) const {
        auto it = find_it(k);
        if (it == data_.end())
            throw std::out_of_range(fmt::format("FlatMap::at: key not found: {}", k));
        return it->second;
    }

    /// @brief Return a mutable reference to the value mapped to `k`.
    /// @throws std::out_of_range if `k` is not present.
    Value &at(const Key &k) {
        auto it = find_it(k);
        if (it == data_.end())
            throw std::out_of_range(fmt::format("FlatMap::at: key not found: {}", k));
        return it->second;
    }

    /// @brief Return true if the map contains an entry for key `k`.
    bool contains(const Key &k) const { return find_it(k) != data_.end(); }

    /// @brief Return a mutable reference to the value for `k`, inserting a
    ///        default-constructed value if `k` is not present.
    Value &operator[](const Key &k) {
        auto it = find_it(k);
        if (it != data_.end())
            return it->second;
        data_.emplace_back(k, Value{});
        return data_.back().second;
    }

    /// @brief Insert the key-value pair `(k, v)`.
    /// @throws std::invalid_argument if `k` already exists in the map.
    void insert(const Key &k, const Value &v) {
        if (contains(k))
            throw std::invalid_argument(fmt::format("FlatMap::insert: duplicate key: {}", k));
        data_.emplace_back(k, v);
    }

    /// @brief Remove all entries from the map.
    void clear() { data_.clear(); }

    /// @brief Return true if the map contains no entries.
    [[nodiscard]] bool empty() const { return data_.empty(); }
    /// @brief Return the number of entries in the map.
    [[nodiscard]] std::size_t size() const { return data_.size(); }

    /// @brief Reserve capacity for at least `n` entries.
    void reserve(std::size_t n) { data_.reserve(n); }

    /// @brief Return a mutable iterator to the first key-value pair.
    const_iterator begin() { return data_.begin(); }
    /// @brief Return a mutable past-the-end iterator.
    const_iterator end() { return data_.end(); }
    /// @brief Return a const iterator to the first key-value pair.
    const_iterator begin() const { return data_.begin(); }
    /// @brief Return a const past-the-end iterator.
    const_iterator end() const { return data_.end(); }
};

} // namespace tycho::utils

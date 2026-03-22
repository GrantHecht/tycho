# Replace std::map in ODESize with FlatMap to eliminate TypeStorage UB

**Date:** 2026-03-22
**Status:** Approved

## Problem

`TypeStorage<C, SBO_CAP>` uses `memcpy` to relocate inline objects during moves.
This requires all inline-stored types to be trivially relocatable (no self-referential
pointers). `ODESize<XV, UV, PV>` contains a `std::map<std::string, Eigen::VectorXi>`
member (`_XtUPidxs`) whose sentinel node has self-referential pointers on both
libstdc++ and libc++. This is undefined behavior that happens to work on
libstdc++/macOS but crashes on libc++ (Linux).

See `TypeStorageProblems.md` (Issue 2) for full details.

## Solution

Replace `std::map` with a new `FlatMap<Key, Value>` utility backed by
`std::vector<std::pair<Key, Value>>`. `std::vector` is trivially relocatable
(stores `{pointer, size, capacity}` with no self-referential pointers). Contained
elements live on the vector's heap backing store, never in TypeStorage's SBO buffer.

## Design

### FlatMap<Key, Value> (`src/Utils/FlatMap.h`)

Minimal flat map: unsorted, insertion-order preserved, linear-scan lookup.
Optimal for the expected N (<10 entries, setup-only access).

```cpp
template <typename Key, typename Value>
class FlatMap {
    std::vector<std::pair<Key, Value>> data_;
public:
    using iterator       = typename std::vector<std::pair<Key, Value>>::iterator;
    using const_iterator = typename std::vector<std::pair<Key, Value>>::const_iterator;

    const Value &at(const Key &k) const;       // throws std::out_of_range
    bool contains(const Key &k) const;
    Value &operator[](const Key &k);           // insert default-constructed if missing
    void insert(const Key &k, const Value &v); // throws std::invalid_argument on dup
    void clear();
    bool empty() const;
    std::size_t size() const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
};
```

### ODESizes.h changes

```cpp
// Before:
std::map<std::string, Eigen::VectorXi> _XtUPidxs;

// After:
FlatMap<std::string, Eigen::VectorXi> _XtUPidxs;
```

Method body updates:
- `.count(name)` -> `.contains(name)`
- `.at(name)`, `operator[]` — unchanged (FlatMap provides both)
- `set_idxs` / `get_idxs` — change parameter/return type from `std::map` to `FlatMap`

No custom copy/move/dtor needed — `FlatMap` has correct compiler-generated defaults.

Bug fix: `add_idx(const std::string&, int)` calls `this->add_idx(idxv)` instead of
`this->add_idx(name, idxv)`. Fix during this change.

### ODESizesBind.h changes

Replace direct method pointer bindings for `get_idxs`/`set_idxs` with lambda
wrappers that convert between `FlatMap` and Python `dict` (nanobind auto-converts
`std::map` but not custom types).

### Tests

- New: `tests/cpp/utils/test_flat_map.cpp` — unit tests for FlatMap
- Existing: C++ unit tests, Python examples — integration validation

## Files Changed

| File | Change |
|------|--------|
| `src/Utils/FlatMap.h` | New — FlatMap utility |
| `src/OptimalControl/ODESizes.h` | `std::map` -> `FlatMap`, method body updates |
| `src/Bindings/OptimalControl/ODESizesBind.h` | Lambda wrappers for dict conversion |
| `tests/cpp/utils/test_flat_map.cpp` | New — FlatMap unit tests |

## Files Unchanged

`ODEBaseClass.py`, `Integrator.h`, `ODEPhase.h`, `ODEPhaseBaseBind.cpp`, all other
callers. Method signatures for `add_idx` and `idx` are unchanged.

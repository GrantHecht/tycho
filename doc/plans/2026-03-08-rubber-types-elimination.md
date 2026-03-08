# rubber_types Elimination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fully remove `rubber_types::TypeErasure` from Tycho by generalising the SBO type-erasure
pattern from `GFStorage` into a reusable `TypeStorage<C>` utility and migrating all remaining
rubber_types-based types (`GenericConditional`, `GenericComparative`, `ConstraintInterface`,
`ObjectiveInterface`) to use it.

**Architecture:** New `src/Utils/TypeStorage.h` provides a 128-byte SBO container with value
semantics; each use site defines its own abstract `*Base` class and `*Model<T>` implementation;
`GFStorage<IR,OR>` is refactored to hold a `TypeStorage<GFConcept<IR,OR>>` member. All
rubber_types files are deleted at the end.

**Tech Stack:** C++20 (clang 22, already in use), Eigen, nanobind. Build: `ninja -j6 all` from
`build/`. Tests: `python run_examples.py` (38 scripts, all must pass) +
`./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp` (must print "Optimal
Solution Found").

**Design doc:** `doc/plans/2026-03-08-rubber-types-elimination-design.md`

---

## Environment

```bash
# Always activate the conda environment first
conda activate tycho

# Build (run from repo root)
cd build && ninja -j6 all

# Full test suite
cd /path/to/tycho && python run_examples.py

# C++ example
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

---

## Task 1: Write `src/Utils/TypeStorage.h`

**Files:**
- Create: `src/Utils/TypeStorage.h`
- Modify: `src/pch.h` (swap `TypeErasure.h` → `TypeStorage.h`)

**Step 1: Create `src/Utils/TypeStorage.h`**

This is the only SBO implementation in the codebase from here on. It replaces the
`shared_ptr<const Concept>` pattern in rubber_types with value semantics.

Convention: `C` must declare `virtual void clone_into(TypeStorage<C, SBO_CAP>&) const = 0`
among its pure virtuals. `Model<T>` implements it as `s.emplace<Model<T>>(data_)`.

`emplace<Model, T>(obj)` takes the Model type explicitly because each use site defines its
own Model (e.g., `ConditionalModel<IR, T>` vs `ConstraintModel<T>`).

```cpp
// src/Utils/TypeStorage.h
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
//
// General-purpose small-buffer-optimised type-erasure container with value
// semantics. Replaces rubber_types::TypeErasure across the Tycho codebase.
//
// Convention: the abstract base class C must declare:
//   virtual void clone_into(TypeStorage<C, SBO_CAP>&) const = 0;
// Each Model<T> implements this as: s.emplace<Model<T>>(data_);
//
// emplace<Model, T> takes the Model type explicitly — each use site defines
// its own Model (ConditionalModel<IR,T>, ConstraintModel<T>, etc.).

#pragma once

#include <concepts>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace Tycho {

// Forward declaration needed for TypeStorageBase concept.
template <typename C, std::size_t SBO_CAP = 128>
class TypeStorage;

// C++20 concept: constrains what can serve as the base-class parameter C.
// C must expose clone_into(TypeStorage<C, N>&) — the virtual that enables
// value-semantic deep-copy.
template <typename C, std::size_t N>
concept TypeStorageBase = requires(const C& c, TypeStorage<C, N>& s) {
    { c.clone_into(s) };
};

template <typename C, std::size_t SBO_CAP>
class TypeStorage {
    static constexpr std::size_t SBO_ALIGN = alignof(std::max_align_t);
    enum class Kind : uint8_t { Empty, Inline, Heap };

    Kind kind_ = Kind::Empty;
    union {
        alignas(SBO_ALIGN) std::byte buf_[SBO_CAP];
        C* ptr_;
    };

    C* concept_ptr() const noexcept {
        return kind_ == Kind::Inline
            ? std::launder(reinterpret_cast<C*>(const_cast<std::byte*>(buf_)))
            : ptr_;
    }

    void destroy() noexcept {
        if (kind_ == Kind::Inline)
            concept_ptr()->~C();
        else if (kind_ == Kind::Heap)
            delete ptr_;
        kind_ = Kind::Empty;
    }

public:
    TypeStorage() noexcept = default;

    // emplace: construct Model in-place (SBO) or on heap.
    // Model must inherit C and implement all of C's pure virtuals.
    // SBO path: placement-new into inline buffer (no heap allocation).
    // Heap path: operator new for large or over-aligned models.
    template <typename Model, typename T>
    void emplace(T obj) {
        static_assert(std::is_base_of_v<C, Model>,
                      "Model must inherit from the TypeStorage base class C");
        destroy();
        if constexpr (sizeof(Model) <= SBO_CAP && alignof(Model) <= SBO_ALIGN) {
            std::construct_at(reinterpret_cast<Model*>(buf_), std::move(obj));
            kind_ = Kind::Inline;
        } else {
            ptr_ = new Model(std::move(obj));
            kind_ = Kind::Heap;
        }
    }

    // Deep-copy via C::clone_into virtual.
    TypeStorage(const TypeStorage& o) {
        if (!o.empty())
            o.concept_ptr()->clone_into(*this);
    }

    // Move: pointer swap (heap) or memcpy (inline, trivially relocatable).
    // All Tycho built-in types are trivially relocatable (plain data + vptr,
    // no internal self-pointers).
    TypeStorage(TypeStorage&& o) noexcept {
        switch (o.kind_) {
        case Kind::Inline:
            std::memcpy(buf_, o.buf_, SBO_CAP);
            kind_ = Kind::Inline;
            o.kind_ = Kind::Empty;
            break;
        case Kind::Heap:
            ptr_ = o.ptr_;
            kind_ = Kind::Heap;
            o.ptr_ = nullptr;
            o.kind_ = Kind::Empty;
            break;
        case Kind::Empty:
            kind_ = Kind::Empty;
            break;
        }
    }

    TypeStorage& operator=(const TypeStorage& o) {
        if (this != &o) {
            destroy();
            if (!o.empty())
                o.concept_ptr()->clone_into(*this);
        }
        return *this;
    }

    TypeStorage& operator=(TypeStorage&& o) noexcept {
        if (this != &o) {
            destroy();
            new (this) TypeStorage(std::move(o));
        }
        return *this;
    }

    ~TypeStorage() { destroy(); }

    [[nodiscard]] bool empty()    const noexcept { return kind_ == Kind::Empty; }
    C&                get()       const noexcept { return *concept_ptr(); }
    C*                get_ptr()   const noexcept { return concept_ptr(); }
};

} // namespace Tycho
```

**Step 2: Swap the include in `src/pch.h`**

In `src/pch.h`, line 45, replace:
```cpp
#include "Utils/TypeErasure.h"
```
with:
```cpp
#include "Utils/TypeStorage.h"
```

**Step 3: Build to verify TypeStorage.h compiles**

```bash
conda activate tycho
cd build && ninja -j6 all 2>&1 | tail -20
```

Expected: build succeeds (no rubber_types removal yet — TypeErasure.h still included via
Tycho_Utils.h at this point; we are only checking TypeStorage.h compiles).

Note: if you see "TypeErasure.h not found", check that `src/Utils/Tycho_Utils.h` still has
its own `#include "TypeErasure.h"` (it does — that gets removed in Task 5).

**Step 4: Commit**

```bash
git add src/Utils/TypeStorage.h src/pch.h
git commit -m "feat: add TypeStorage<C> SBO utility — replaces rubber_types::TypeErasure"
```

---

## Task 2: Migrate `GenericConditional` and `GenericComparative`

**Files:**
- Create: `src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h`
- Modify: `src/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h`
- Modify: `src/VectorFunctions/VectorFunctionTypeErasure/GenericComparative.h`

`GenericConditional<IR>` and `GenericComparative<IR>` are both type-erased boolean predicates
(used as stop-functions in integrators). They share the same 3-virtual interface and can share
one `ConditionalBase<IR>` + `ConditionalModel<IR,T>` pair. After migration, `GenericComparative`
becomes a type alias.

**Step 1: Create `src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h`**

```cpp
// src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
//
// TypeStorage-based replacement for the rubber_types ConditionalSpec<IR> in
// GenericConditional.h / GenericComparative.h.

#pragma once

#include "pch.h"
#include "Utils/TypeStorage.h"  // comes via pch.h in practice; explicit for clarity

namespace Tycho {

// ---------------------------------------------------------------------------
// C++20 concept: minimum interface a concrete conditional type T must satisfy.
// Gives a clear diagnostic at GenericConditional(const T&) when T is wrong.
// ---------------------------------------------------------------------------
template <int IR, typename T>
concept ConditionalStorable = requires(const T& t,
    const Eigen::Ref<const Eigen::Matrix<double, IR, 1>>& x) {
    { t.name() }     -> std::same_as<std::string>;
    { t.IRows() }    -> std::same_as<int>;
    { t.compute(x) } -> std::same_as<bool>;
};

// ---------------------------------------------------------------------------
// ConditionalBase<IR>: flat abstract base for TypeStorage.
// Must declare clone_into to satisfy TypeStorageBase concept.
// ---------------------------------------------------------------------------
template <int IR>
struct ConditionalBase {
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;
    template <class S> using ConstVectorBaseRef = const Eigen::MatrixBase<S>&;

    virtual ~ConditionalBase() = default;
    virtual std::string name()  const = 0;
    virtual int         IRows() const = 0;
    virtual bool compute(ConstVectorBaseRef<InType> x) const = 0;
    virtual void clone_into(TypeStorage<ConditionalBase<IR>>&) const = 0;
};

// ---------------------------------------------------------------------------
// ConditionalModel<IR, T>: concrete implementation.
// ---------------------------------------------------------------------------
template <int IR, typename T>
struct ConditionalModel final : ConditionalBase<IR> {
    using Base    = ConditionalBase<IR>;
    using InType  = typename Base::InType;
    template <class S> using ConstVectorBaseRef = const Eigen::MatrixBase<S>&;

    T data_;
    explicit ConditionalModel(T t) : data_(std::move(t)) {}

    std::string name()  const override { return data_.name(); }
    int         IRows() const override { return data_.IRows(); }

    bool compute(ConstVectorBaseRef<InType> x) const override {
        InType xt(x.derived());
        return data_.compute(xt);
    }

    void clone_into(TypeStorage<ConditionalBase<IR>>& s) const override {
        s.emplace<ConditionalModel<IR, T>>(data_);
    }
};

} // namespace Tycho
```

**Step 2: Replace `GenericConditional.h`**

Replace the entire file body (after the copyright header) with:

```cpp
#pragma once
#include "ConditionalTypeErasure.h"
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"
#include "pch.h"

namespace Tycho {

template <int IR>
struct GenericConditional {
    static const bool IsConditional = true;
    static const int  IRC = IR;

    using Storage = TypeStorage<ConditionalBase<IR>>;
    Storage storage;

    GenericConditional() = default;

    template <ConditionalStorable<IR> T>
    GenericConditional(const T& t) {
        storage.emplace<ConditionalModel<IR, std::decay_t<T>>>(t);
    }

    // Copy/move: TypeStorage value semantics handle deep-copy automatically.

    std::string name()  const { return storage.get().name(); }
    int         IRows() const { return storage.get().IRows(); }

    template <class InT>
    bool compute(const Eigen::MatrixBase<InT>& x) const {
        typename ConditionalBase<IR>::InType xt(x.derived());
        return storage.get().compute(xt);
    }
};

} // namespace Tycho
```

**Step 3: Replace `GenericComparative.h`**

Both types share the same interface, so `GenericComparative` becomes a type alias:

```cpp
#pragma once
#include "GenericConditional.h"

namespace Tycho {

// GenericComparative and GenericConditional have the same interface.
template <int IR>
using GenericComparative = GenericConditional<IR>;

} // namespace Tycho
```

**Step 4: Build**

```bash
conda activate tycho
cd build && ninja -j6 all 2>&1 | tail -30
```

Expected: clean build. Common issue: if any call site uses
`obj.get_container()` or `reset_container()` (old rubber_types API) you will see a
compile error pointing to that line — fix by removing those calls (they were
rubber_types copy-sharing, now handled by TypeStorage copy constructors).

**Step 5: Run tests**

```bash
cd /path/to/tycho && python run_examples.py
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: 38/38 pass, "Optimal Solution Found".

**Step 6: Commit**

```bash
git add src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h \
        src/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h \
        src/VectorFunctions/VectorFunctionTypeErasure/GenericComparative.h
git commit -m "feat: migrate GenericConditional/Comparative from rubber_types to TypeStorage"
```

---

## Task 3: Migrate `ConstraintInterface` and `ObjectiveInterface`

**Files:**
- Modify: `src/VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h`
- Modify: `src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h`
  (update `pack_into_constraint_interface` and `pack_into_objective_interface` bodies)

This is the largest change. `ConstraintInterface` and `ObjectiveInterface` are the types
that PSIOPT calls at solve time (hot path). The virtual signatures that PSIOPT uses are
**unchanged** — only the container changes from rubber_types to TypeStorage.

Note: `SolverInterfaceSelector` (lines ~295–300 of `SolverInterfaceSpecs.h`) is dead
code — only defined, never used anywhere. Delete it.

**Step 1: Add `ConstraintStorable` concept and `ConstraintBase` + `ConstraintModel<T>`
to `SolverInterfaceSpecs.h`**

After the closing `};` of `SolverConstraintSpec` (around line 205) and before
`SolverObjectiveSpec`, insert:

```cpp
// ---------------------------------------------------------------------------
// C++20 concept: minimum interface for ConstraintInterface construction.
// ---------------------------------------------------------------------------
template <typename T>
concept ConstraintStorable = requires(const T& t) {
    { t.IRows() }      -> std::same_as<int>;
    { t.ORows() }      -> std::same_as<int>;
    { t.name() }       -> std::same_as<std::string>;
    { t.thread_safe() } -> std::same_as<bool>;
};

// ---------------------------------------------------------------------------
// ConstraintBase: flat abstract base for TypeStorage<ConstraintBase>.
// Inherits SolverConstraintSpec::Concept and SizableSpec::Concept
// non-virtually (same flat-vtable pattern as GFConcept).
// ---------------------------------------------------------------------------
struct ConstraintBase
    : SolverConstraintSpec::Concept,
      SizableSpec::Concept {
    virtual ~ConstraintBase() = default;
    virtual void clone_into(TypeStorage<ConstraintBase>&) const = 0;
};

// ---------------------------------------------------------------------------
// ConstraintModel<T>: implements all virtuals by delegating to data_.
// ---------------------------------------------------------------------------
template <typename T>
struct ConstraintModel final : ConstraintBase {
    T data_;
    explicit ConstraintModel(T t) : data_(std::move(t)) {}

    // ---- SolverConstraintSpec::Concept overrides ----
    void constraints(const Eigen::Ref<const Eigen::VectorXd>& X,
                     Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData& data) const override {
        data_.constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd>& X,
                                     const Eigen::Ref<const Eigen::VectorXd>& L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData& data) const override {
        data_.constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd>& X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks,
                              const SolverIndexingData& data) const override {
        data_.constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const override {
        data_.constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat,
                                                   KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const override {
        data_.constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows,
                     Eigen::Ref<Eigen::VectorXi> KKTcols,
                     int& freeloc, int conoffset,
                     bool dojac, bool dohess,
                     SolverIndexingData& data) override {
        data_.getKKTSpace(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int numKKTEles(bool dojac, bool dohess) const override {
        return data_.numKKTEles(dojac, dohess);
    }

    // ---- SizableSpec::Concept overrides ----
    std::string name()       const override { return data_.name(); }
    int         IRows()      const override { return data_.IRows(); }
    int         ORows()      const override { return data_.ORows(); }
    bool        thread_safe() const override { return data_.thread_safe(); }

    // ---- TypeStorage clone ----
    void clone_into(TypeStorage<ConstraintBase>& s) const override {
        s.emplace<ConstraintModel<T>>(data_);
    }
};
```

**Step 2: Replace the `ConstraintInterface` class in `SolverInterfaceSpecs.h`**

Replace the entire `struct ConstraintInterface : rubber_types::TypeErasure<...> { ... };`
block with:

```cpp
/*
 * ConstraintInterface: value-semantic wrapper around TypeStorage<ConstraintBase>.
 * Stores the concrete function type directly (one virtual dispatch per solver call).
 * GenericFunction constructor uses pack_into_constraint_interface to avoid double-erasure.
 * Copy/move: TypeStorage clone_into provides deep-copy value semantics.
 */
struct ConstraintInterface {
    TypeStorage<ConstraintBase> storage;

    ConstraintInterface() = default;

    template <class T,
        std::enable_if_t<!std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>,
                                             std::decay_t<T>>, bool> = true>
    ConstraintInterface(const T& t) {
        storage.emplace<ConstraintModel<std::decay_t<T>>>(t);
    }

    // GenericFunction path: uses pack_into to store T directly (no double-erasure).
    // See GFModelCommon::pack_into_constraint_interface in GFTypeErasure.h.
    template <int IR, int OR>
    ConstraintInterface(const GenericFunction<IR, OR>& t) {
        t.func.get().pack_into_constraint_interface(*this);
    }

    // Forward all solver calls to storage.get()
    void constraints(const Eigen::Ref<const Eigen::VectorXd>& X,
                     Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData& data) const {
        storage.get().constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd>& X,
                                     const Eigen::Ref<const Eigen::VectorXd>& L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData& data) const {
        storage.get().constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd>& X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks,
                              const SolverIndexingData& data) const {
        storage.get().constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes,
                                           KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {
        storage.get().constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat,
                                                           KKTLocations, KKTClashes,
                                                           KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {
        storage.get().constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows,
                     Eigen::Ref<Eigen::VectorXi> KKTcols,
                     int& freeloc, int conoffset,
                     bool dojac, bool dohess,
                     SolverIndexingData& data) {
        storage.get().getKKTSpace(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int numKKTEles(bool dojac, bool dohess) const {
        return storage.get().numKKTEles(dojac, dohess);
    }
    std::string name()       const { return storage.get().name(); }
    int         IRows()      const { return storage.get().IRows(); }
    int         ORows()      const { return storage.get().ORows(); }
    bool        thread_safe() const { return storage.get().thread_safe(); }
};
```

**Step 3: Add `ObjectiveStorable` concept, `ObjectiveBase`, and `ObjectiveModel<T>`**

After the closing `};` of `SolverObjectiveSpec` and before `SolverInterfaceSelector`, insert:

```cpp
// ---------------------------------------------------------------------------
// C++20 concept: minimum interface for ObjectiveInterface construction.
// ---------------------------------------------------------------------------
template <typename T>
concept ObjectiveStorable = ConstraintStorable<T> && requires(
    const T& t, double os,
    const Eigen::Ref<const Eigen::VectorXd>& X,
    double& val, Eigen::Ref<Eigen::VectorXd> GX,
    const SolverIndexingData& data) {
    { t.objective(os, X, val, data) };
};

// ---------------------------------------------------------------------------
// ObjectiveBase: flat abstract base for TypeStorage<ObjectiveBase>.
// ---------------------------------------------------------------------------
struct ObjectiveBase
    : SolverConstraintSpec::Concept,
      SolverObjectiveSpec::Concept,
      SizableSpec::Concept {
    virtual ~ObjectiveBase() = default;
    virtual void clone_into(TypeStorage<ObjectiveBase>&) const = 0;
};

// ---------------------------------------------------------------------------
// ObjectiveModel<T>: implements all constraint + objective + sizable virtuals.
// ---------------------------------------------------------------------------
template <typename T>
struct ObjectiveModel final : ObjectiveBase {
    T data_;
    explicit ObjectiveModel(T t) : data_(std::move(t)) {}

    // ---- SolverConstraintSpec::Concept overrides (same as ConstraintModel) ----
    void constraints(const Eigen::Ref<const Eigen::VectorXd>& X,
                     Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData& data) const override {
        data_.constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd>& X,
                                     const Eigen::Ref<const Eigen::VectorXd>& L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData& data) const override {
        data_.constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd>& X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks,
                              const SolverIndexingData& data) const override {
        data_.constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const override {
        data_.constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat,
                                                   KKTLocations, KKTClashes, KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const override {
        data_.constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows,
                     Eigen::Ref<Eigen::VectorXi> KKTcols,
                     int& freeloc, int conoffset,
                     bool dojac, bool dohess,
                     SolverIndexingData& data) override {
        data_.getKKTSpace(KKTrows, KKTcols, freeloc, conoffset, dojac, dohess, data);
    }
    int numKKTEles(bool dojac, bool dohess) const override {
        return data_.numKKTEles(dojac, dohess);
    }

    // ---- SolverObjectiveSpec::Concept overrides ----
    void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd>& X,
                   double& Val, const SolverIndexingData& data) const override {
        data_.objective(ObjScale, X, Val, data);
    }
    void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd>& X,
                            double& Val, Eigen::Ref<Eigen::VectorXd> GX,
                            const SolverIndexingData& data) const override {
        data_.objective_gradient(ObjScale, X, Val, GX, data);
    }
    void objective_gradient_hessian(double ObjScale,
                                    const Eigen::Ref<const Eigen::VectorXd>& X,
                                    double& Val, Eigen::Ref<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                                    Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                    Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex>& KKTLocks,
                                    const SolverIndexingData& data) const override {
        data_.objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat,
                                         KKTLocations, KKTClashes, KKTLocks, data);
    }

    // ---- SizableSpec::Concept overrides ----
    std::string name()       const override { return data_.name(); }
    int         IRows()      const override { return data_.IRows(); }
    int         ORows()      const override { return data_.ORows(); }
    bool        thread_safe() const override { return data_.thread_safe(); }

    // ---- TypeStorage clone ----
    void clone_into(TypeStorage<ObjectiveBase>& s) const override {
        s.emplace<ObjectiveModel<T>>(data_);
    }
};
```

**Step 4: Replace the `ObjectiveInterface` class and delete `SolverInterfaceSelector`**

Replace the entire `struct ObjectiveInterface : rubber_types::TypeErasure<...> { ... };` block with:

```cpp
/*
 * ObjectiveInterface: value-semantic wrapper around TypeStorage<ObjectiveBase>.
 * See ConstraintInterface for rationale.
 */
struct ObjectiveInterface {
    TypeStorage<ObjectiveBase> storage;

    ObjectiveInterface() = default;

    template <class T,
        std::enable_if_t<!std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>,
                                             std::decay_t<T>>, bool> = true>
    ObjectiveInterface(const T& t) {
        storage.emplace<ObjectiveModel<std::decay_t<T>>>(t);
    }

    template <int IR>
    ObjectiveInterface(const GenericFunction<IR, 1>& t) {
        t.func.get().pack_into_objective_interface(*this);
    }

    // Forward solver calls to storage.get()
    void constraints(const Eigen::Ref<const Eigen::VectorXd>& X,
                     Eigen::Ref<Eigen::VectorXd> FX,
                     const SolverIndexingData& data) const {
        storage.get().constraints(X, FX, data);
    }
    void constraints_adjointgradient(const Eigen::Ref<const Eigen::VectorXd>& X,
                                     const Eigen::Ref<const Eigen::VectorXd>& L,
                                     Eigen::Ref<Eigen::VectorXd> FX,
                                     Eigen::Ref<Eigen::VectorXd> AGX,
                                     const SolverIndexingData& data) const {
        storage.get().constraints_adjointgradient(X, L, FX, AGX, data);
    }
    void constraints_jacobian(const Eigen::Ref<const Eigen::VectorXd>& X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks,
                              const SolverIndexingData& data) const {
        storage.get().constraints_jacobian(X, FX, KKTmat, KKTLocations, KKTClashes,
                                           KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {
        storage.get().constraints_jacobian_adjointgradient(X, L, FX, AGX, KKTmat,
                                                           KKTLocations, KKTClashes,
                                                           KKTLocks, data);
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        const Eigen::Ref<const Eigen::VectorXd>& X,
        const Eigen::Ref<const Eigen::VectorXd>& L,
        Eigen::Ref<Eigen::VectorXd> FX, Eigen::Ref<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        Eigen::Ref<Eigen::VectorXi> KKTLocations,
        Eigen::Ref<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {
        storage.get().constraints_jacobian_adjointgradient_adjointhessian(
            X, L, FX, AGX, KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
    }
    void getKKTSpace(Eigen::Ref<Eigen::VectorXi> KKTrows,
                     Eigen::Ref<Eigen::VectorXi> KKTcols,
                     int& freeloc, int conoffset,
                     bool dojac, bool dohess,
                     SolverIndexingData& data) {
        storage.get().getKKTSpace(KKTrows, KKTcols, freeloc, conoffset,
                                  dojac, dohess, data);
    }
    int numKKTEles(bool dojac, bool dohess) const {
        return storage.get().numKKTEles(dojac, dohess);
    }
    void objective(double ObjScale, const Eigen::Ref<const Eigen::VectorXd>& X,
                   double& Val, const SolverIndexingData& data) const {
        storage.get().objective(ObjScale, X, Val, data);
    }
    void objective_gradient(double ObjScale, const Eigen::Ref<const Eigen::VectorXd>& X,
                            double& Val, Eigen::Ref<Eigen::VectorXd> GX,
                            const SolverIndexingData& data) const {
        storage.get().objective_gradient(ObjScale, X, Val, GX, data);
    }
    void objective_gradient_hessian(double ObjScale,
                                    const Eigen::Ref<const Eigen::VectorXd>& X,
                                    double& Val, Eigen::Ref<Eigen::VectorXd> GX,
                                    Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                                    Eigen::Ref<Eigen::VectorXi> KKTLocations,
                                    Eigen::Ref<Eigen::VectorXi> KKTClashes,
                                    std::vector<std::mutex>& KKTLocks,
                                    const SolverIndexingData& data) const {
        storage.get().objective_gradient_hessian(ObjScale, X, Val, GX, KKTmat,
                                                 KKTLocations, KKTClashes, KKTLocks, data);
    }
    std::string name()       const { return storage.get().name(); }
    int         IRows()      const { return storage.get().IRows(); }
    int         ORows()      const { return storage.get().ORows(); }
    bool        thread_safe() const { return storage.get().thread_safe(); }
};
```

Also delete the `SolverInterfaceSelector` block entirely — it is defined but never used anywhere.

**Step 5: Remove `#include "DeepCopySpecs.h"` from `SolverInterfaceSpecs.h`**

Delete line 22: `#include "DeepCopySpecs.h"`

**Step 6: Update `pack_into_constraint_interface` in `GFTypeErasure.h`**

In `GFModelCommon` (around line 287–289), the body currently reads:
```cpp
void pack_into_constraint_interface(ConstraintInterface& ci) const override {
    ci = ConstraintInterface(this->data_);
}
```

Replace with direct emplace (avoids the extra copy through ConstraintInterface constructor):
```cpp
void pack_into_constraint_interface(ConstraintInterface& ci) const override {
    ci.storage.emplace<ConstraintModel<T>>(this->data_);
}
```

**Step 7: Update `pack_into_objective_interface` in `GFTypeErasure.h`**

In `GFModel<IR, 1, T>` (around line 334–336), the body currently reads:
```cpp
void pack_into_objective_interface(ObjectiveInterface& oi) const override {
    oi = ObjectiveInterface(this->data_);
}
```

Replace with:
```cpp
void pack_into_objective_interface(ObjectiveInterface& oi) const override {
    oi.storage.emplace<ObjectiveModel<T>>(this->data_);
}
```

Note: `GFTypeErasure.h` forward-declares `ConstraintInterface` and `ObjectiveInterface` (line 37–38).
After this change it also needs `ConstraintModel` and `ObjectiveModel` to be visible.
Since `GFTypeErasure.h` includes `SolverInterfaceSpecs.h` (line 23), and those new types are
defined there, no additional includes are needed.

**Step 8: Build**

```bash
conda activate tycho
cd build && ninja -j6 all 2>&1 | tail -40
```

Expected: clean build. Watch for:
- `ConstraintBase is not complete` — forward declaration ordering issue; check that
  `ConstraintBase` is defined before it's used as the TypeStorage parameter
- `no matching function for call to ConstraintInterface(...)` — a call site using the old
  rubber_types API; trace and update

**Step 9: Run tests**

```bash
cd /path/to/tycho && python run_examples.py
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: 38/38 pass, "Optimal Solution Found".

**Step 10: Commit**

```bash
git add src/VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h \
        src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h
git commit -m "feat: migrate ConstraintInterface/ObjectiveInterface from rubber_types to TypeStorage"
```

---

## Task 4: Refactor `GFStorage` to use `TypeStorage` internally

**Files:**
- Modify: `src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h`

`GFStorage<IR,OR>` currently reimplements the same SBO mechanics as TypeStorage.
This task replaces that implementation with a `TypeStorage<GFConcept<IR,OR>>` member.
The public API (`emplace`, `get`, `get_ptr`, `empty`, copy/move) is unchanged.

`GFConcept<IR,OR>` must satisfy `TypeStorageBase`: it needs to declare
`virtual void clone_into(TypeStorage<GFConcept<IR,OR>>&) const = 0`. Right now it declares
`virtual void clone_into(GFStorage<IR,OR>&) const = 0`. We need to add the TypeStorage
overload, since TypeStorage requires its own type.

**Strategy:** Add `clone_into(TypeStorage<GFConcept<IR,OR>>&)` as a new pure virtual to
`GFConcept`. Implement it in `GFModelCommon` to call the existing `GFStorage` path.
Then replace `GFStorage`'s internals with a `TypeStorage<GFConcept<IR,OR>>` member.

**Step 1: Add `clone_into(TypeStorage<GFConcept<IR,OR>>&)` to `GFConcept`**

In the primary `GFConcept<IR, OR>` template (around line 84–92), the new virtual must
be added alongside the existing ones. Update both the primary template and the `<IR,1>`
partial specialisation:

Primary template — add after `clone_into(GFStorage<IR, OR>&)`:
```cpp
virtual void clone_into_ts(TypeStorage<GFConcept<IR, OR>>&) const = 0;
```

Partial spec `<IR, 1>` — same addition:
```cpp
virtual void clone_into_ts(TypeStorage<GFConcept<IR, 1>>&) const = 0;
```

**Step 2: Implement `clone_into_ts` in `GFModelCommon`**

Add after the existing `clone_into` out-of-line definitions (around line 460):

```cpp
template <int IR, int OR, GFStorable<IR, OR> T>
void GFModelCommon<IR, OR, T>::clone_into_ts(TypeStorage<GFConcept<IR, OR>>& s) const {
    s.emplace<GFModel<IR, OR, T>>(this->data_);
}
```

Also add the declaration inside `GFModelCommon`:
```cpp
void clone_into_ts(TypeStorage<GFConcept<IR, OR>>&) const override;
```

**Step 3: Replace `GFStorage<IR,OR>` internals**

Replace the entire `class GFStorage { ... };` block with:

```cpp
template <int IR, int OR>
class GFStorage {
    TypeStorage<GFConcept<IR, OR>> storage_;

public:
    GFStorage() noexcept = default;

    // C++20 concept constraint: "T does not satisfy GFStorable<IR,OR>" on bad types.
    template <GFStorable<IR, OR> T>
    void emplace(T obj) {
        storage_.emplace<GFModel<IR, OR, std::decay_t<T>>>(std::move(obj));
    }

    // Copy/move/dtor: delegate entirely to TypeStorage.
    GFStorage(const GFStorage&)            = default;
    GFStorage(GFStorage&&)                 = default;
    GFStorage& operator=(const GFStorage&) = default;
    GFStorage& operator=(GFStorage&&)      = default;
    ~GFStorage()                           = default;

    [[nodiscard("Check empty() before calling get()")]]
    bool empty() const noexcept { return storage_.empty(); }

    GFConcept<IR, OR>&  get()     const noexcept { return storage_.get(); }
    GFConcept<IR, OR>*  get_ptr() const noexcept { return storage_.get_ptr(); }
};
```

Also remove the out-of-line `GFModelCommon::clone_into` and `clone_into_dynamic` definitions
that called `s.emplace(this->data_)` — they still work since `GFStorage::emplace` now
delegates to `TypeStorage`. No change needed to those bodies.

**Step 4: Build**

```bash
conda activate tycho
cd build && ninja -j6 all 2>&1 | tail -30
```

**Step 5: Run tests**

```bash
cd /path/to/tycho && python run_examples.py
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: 38/38 pass, "Optimal Solution Found".

**Step 6: Commit**

```bash
git add src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h
git commit -m "refactor: GFStorage now wraps TypeStorage<GFConcept<IR,OR>> internally"
```

---

## Task 5: Strip dead boilerplate + delete rubber_types files

**Files:**
- Modify: `src/VectorFunctions/VectorFunctionTypeErasure/DenseFunctionSpecs.h`
- Modify: `src/VectorFunctions/VectorFunctionTypeErasure/SizingSpecs.h`
- Modify: `src/Utils/Tycho_Utils.h`
- Delete: `src/VectorFunctions/VectorFunctionTypeErasure/DeepCopySpecs.h`
- Delete: `src/Utils/TypeErasure.h`
- Delete: `notices/rubber_types-mit.txt`

**Step 1: Strip dead `Model<>` and `ExternalInterface<>` from `DenseFunctionSpecs.h`**

In `DenseFunctionSpecs.h`, the `DenseFunctionSpec<IR,OR>` struct has three inner structs:
`Concept`, `Model<Holder>`, and `ExternalInterface<Container>`.

`Model<Holder>` and `ExternalInterface<Container>` are dead — they were used by rubber_types
composition and are no longer instantiated by anything. Only `Concept` is still used (as a
non-virtual base for `GFConcept`).

Delete the entire `Model<Holder>` template (all ~200 lines of virtual overrides using
`rubber_types::model_get`) and the entire `ExternalInterface<Container>` template.
Keep only the outer `DenseFunctionSpec<IR,OR>` struct, its type aliases, and its `Concept`
inner struct.

Verify by searching: `grep -n "rubber_types" src/VectorFunctions/VectorFunctionTypeErasure/DenseFunctionSpecs.h`
Expected: zero matches after the edit.

**Step 2: Strip dead `Model<>` and `ExternalInterface<>` from `SizingSpecs.h`**

Same approach: delete `Model<Holder>` and `ExternalInterface<Container>` from `SizableSpec`.
Keep only `Concept`. File should shrink from ~63 lines to ~37 lines.

Verify: `grep -n "rubber_types" src/VectorFunctions/VectorFunctionTypeErasure/SizingSpecs.h`
Expected: zero matches.

**Step 3: Remove `TypeErasure.h` include from `Tycho_Utils.h`**

In `src/Utils/Tycho_Utils.h`, find and delete the line:
```cpp
#include "TypeErasure.h"
```

**Step 4: Delete `DeepCopySpecs.h`**

```bash
git rm src/VectorFunctions/VectorFunctionTypeErasure/DeepCopySpecs.h
```

**Step 5: Delete `TypeErasure.h`**

```bash
git rm src/Utils/TypeErasure.h
```

**Step 6: Delete the rubber_types license notice**

```bash
git rm notices/rubber_types-mit.txt
```

**Step 7: Build — full clean**

```bash
conda activate tycho
cd build && ninja -j6 all 2>&1 | tail -30
```

Expected: clean build with zero rubber_types references. If you see "TypeErasure.h: No such
file or directory", there is a remaining include somewhere — find it with:
```bash
grep -r "TypeErasure" src/
```
and remove it.

**Step 8: Run tests**

```bash
cd /path/to/tycho && python run_examples.py
./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

Expected: 38/38 pass, "Optimal Solution Found".

**Step 9: Final verification — zero rubber_types**

```bash
grep -r "rubber_types" src/    # must be empty
grep -r "TypeErasure.h" src/   # must be empty
grep -r "DeepCopySpecs" src/   # must be empty
```

All three must return zero matches.

**Step 10: Commit**

```bash
git add src/VectorFunctions/VectorFunctionTypeErasure/DenseFunctionSpecs.h \
        src/VectorFunctions/VectorFunctionTypeErasure/SizingSpecs.h \
        src/Utils/Tycho_Utils.h
git commit -m "chore: strip dead rubber_types boilerplate and delete TypeErasure.h, DeepCopySpecs.h, rubber_types notice"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `ninja -j6 all` — clean build, no warnings about rubber_types
- [ ] `python run_examples.py` — 38/38 pass
- [ ] `./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp` — "Optimal Solution Found"
- [ ] `grep -r "rubber_types" src/` — zero matches
- [ ] `grep -r "TypeErasure.h" src/` — zero matches
- [ ] `grep -r "DeepCopySpecs" src/` — zero matches
- [ ] `ls notices/rubber_types-mit.txt` — file does not exist

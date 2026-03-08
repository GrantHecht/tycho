# PR 8 Design — Full rubber_types Elimination

**Date:** 2026-03-08
**Goal:** Remove all remaining `rubber_types::TypeErasure` usage from the codebase, generalize
the SBO type-erasure pattern developed for `GFStorage` into a reusable `TypeStorage<C>` utility,
and delete `TypeErasure.h` along with its MIT license notice.

---

## Background

PR 6 replaced rubber_types for `GenericFunction<IR,OR>` (the main VJP type-erased object) with
the new flat-vtable `GFStorage<IR,OR>` / `GFConcept<IR,OR>` / `GFModel<IR,OR,T>` system. That
left rubber_types still in use in three places:

| Remaining rubber_types usage | What it provides |
|---|---|
| `GenericConditional<IR>`, `GenericComparative<IR>` | Type-erased boolean predicates (stop functions for integrators) |
| `ConstraintInterface` | Type-erased solver constraint (what PSIOPT calls at solve time) |
| `ObjectiveInterface` | Type-erased solver objective |

Additionally, the `Model<>` and `ExternalInterface<>` inner structs in `DenseFunctionSpecs.h`,
`SizingSpecs.h`, and `SolverInterfaceSpecs.h` became dead code in PR 6 (GFModel directly
delegates every virtual; these rubber_types boilerplate structs are never instantiated).

---

## Design

### Part 1 — `TypeStorage<C, SBO_CAP>` (`src/Utils/TypeStorage.h`)

A general-purpose SBO container with value semantics that replaces the
`shared_ptr<const Concept>` pattern in rubber_types.

**Convention:** `C` must be an abstract class that declares
`virtual void clone_into(TypeStorage<C, SBO_CAP>&) const = 0` among its pure virtuals.
The `Model<T>` the caller writes implements this as `s.emplace<Model<T>>(data_)`.

**C++20 concept constraining the base type:**
```cpp
template <typename C, std::size_t N>
concept TypeStorageBase = requires(const C& c, TypeStorage<C, N>& s) {
    { c.clone_into(s) };
};

template <TypeStorageBase<SBO_CAP> C, std::size_t SBO_CAP = 128>
class TypeStorage { ... };
```

**Key fields and methods:**
```cpp
template <TypeStorageBase<SBO_CAP> C, std::size_t SBO_CAP = 128>
class TypeStorage {
    static constexpr std::size_t SBO_ALIGN = alignof(std::max_align_t);
    enum class Kind : uint8_t { Empty, Inline, Heap };

    Kind kind_ = Kind::Empty;
    union {
        alignas(SBO_ALIGN) std::byte buf_[SBO_CAP];
        C* ptr_;
    };

public:
    TypeStorage() noexcept = default;

    // emplace: caller specifies both Model type and T value.
    // Each use site defines its own Model<T> (e.g. ConditionalModel<IR,T> vs ConstraintModel<T>).
    template <typename Model, typename T>
    void emplace(T obj);   // SBO if sizeof(Model) <= SBO_CAP, else heap

    // Deep-copy via C::clone_into virtual
    TypeStorage(const TypeStorage&);
    TypeStorage(TypeStorage&&) noexcept;
    TypeStorage& operator=(const TypeStorage&);
    TypeStorage& operator=(TypeStorage&&) noexcept;
    ~TypeStorage();

    [[nodiscard]] bool empty()    const noexcept;
    C&                get()       const noexcept;
    C*                get_ptr()   const noexcept;
};
```

**Difference from GFStorage:** `emplace<Model, T>` takes both the Model type and T explicitly.
GFStorage's `emplace<T>` deduces `GFModel<IR,OR,T>` internally — after refactoring, it calls
`TypeStorage::emplace<GFModel<IR,OR,T>, T>` under the hood.

**SBO_CAP:** defaults to 128 bytes for all uses. `ConstraintInterface`/`ObjectiveInterface`
store large solver functions (e.g. shooting defects). If any exceed 128 bytes the heap fallback
is correct — no correctness risk, only a missed SBO opportunity. Verify during implementation.

---

### Part 2 — Conditional Migration (`ConditionalTypeErasure.h`)

New file: `src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h`

Replaces the rubber_types guts of `GenericConditional.h` and `GenericComparative.h`.
Both types share the same interface, so they share one Concept/Model pair.

**`ConditionalBase<IR>`** — flat abstract base:
```cpp
template <int IR>
struct ConditionalBase {
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;
    template <class Scalar> using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;

    virtual ~ConditionalBase() = default;
    virtual std::string name()  const = 0;
    virtual int         IRows() const = 0;
    virtual bool compute(ConstVectorBaseRef<InType> x) const = 0;
    virtual void clone_into(TypeStorage<ConditionalBase<IR>>&) const = 0;
};
```

**`ConditionalModel<IR, T>`** — implements all virtuals:
```cpp
template <int IR, typename T>
struct ConditionalModel final : ConditionalBase<IR> {
    T data_;
    explicit ConditionalModel(T t) : data_(std::move(t)) {}
    std::string name()  const override { return data_.name(); }
    int         IRows() const override { return data_.IRows(); }
    bool compute(ConstVectorBaseRef<InType> x) const override {
        InType xt(x.derived()); return data_.compute(xt);
    }
    void clone_into(TypeStorage<ConditionalBase<IR>>& s) const override {
        s.emplace<ConditionalModel<IR, T>>(data_);
    }
};
```

**`GenericConditional<IR>`** — thin value-semantic wrapper:
```cpp
template <int IR>
struct GenericConditional {
    static const bool IsConditional = true;
    static const int  IRC = IR;

    TypeStorage<ConditionalBase<IR>> storage;

    GenericConditional() = default;
    template <class T> GenericConditional(const T& t) {
        storage.emplace<ConditionalModel<IR, T>>(t);
    }
    // Copy/move: TypeStorage value semantics handle automatically

    std::string name()  const { return storage.get().name(); }
    int         IRows() const { return storage.get().IRows(); }
    template <class InT>
    bool compute(const Eigen::MatrixBase<InT>& x) const {
        typename ConditionalBase<IR>::InType xt(x.derived());
        return storage.get().compute(xt);
    }
};

// GenericComparative: same interface and storage
template <int IR> using GenericComparative = GenericConditional<IR>;
```

The `ConditionalSpec<IR>` struct and `reset_container` call in `GenericConditional.h` are removed.
`Integrator.h`'s `StopFuncType = GenericConditional<-1>` and all binding call sites are unaffected.

---

### Part 3 — Solver Interface Migration (`SolverInterfaceSpecs.h`)

`ConstraintInterface` and `ObjectiveInterface` each get a flat abstract base class, a templated
Model, and wrap `TypeStorage`. The existing `SolverConstraintSpec::Concept` and
`SolverObjectiveSpec::Concept` inner structs are reused as non-virtual bases (same flat-vtable
pattern as `GFConcept`) — PSIOPT sees identical virtual signatures.

**`ConstraintBase`** — flat abstract base, non-virtual MI of existing Spec Concepts:
```cpp
struct ConstraintBase
    : SolverConstraintSpec::Concept,
      SizableSpec::Concept {
    virtual ~ConstraintBase() = default;
    virtual void clone_into(TypeStorage<ConstraintBase>&) const = 0;
};
```

**`ConstraintModel<T>`** — delegates every virtual to `data_`, plus clone:
```cpp
template <typename T>
struct ConstraintModel final : ConstraintBase {
    T data_;
    explicit ConstraintModel(T t) : data_(std::move(t)) {}
    // SolverConstraintSpec overrides — all delegate to data_
    void constraints(...) const override { data_.constraints(...); }
    // ... (all remaining SolverConstraintSpec virtuals)
    // SizableSpec overrides
    std::string name()  const override { return data_.name(); }
    int         IRows() const override { return data_.IRows(); }
    int         ORows() const override { return data_.ORows(); }
    bool thread_safe()  const override { return data_.thread_safe(); }
    // Clone
    void clone_into(TypeStorage<ConstraintBase>& s) const override {
        s.emplace<ConstraintModel<T>>(data_);
    }
};
```

**`ConstraintInterface`** — value-semantic wrapper:
```cpp
struct ConstraintInterface {
    TypeStorage<ConstraintBase> storage;

    ConstraintInterface() = default;
    template <class T,
        std::enable_if_t<!std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>,
                                             std::decay_t<T>>, bool> = true>
    ConstraintInterface(const T& t) {
        storage.emplace<ConstraintModel<std::decay_t<T>>>(t);
    }
    // GenericFunction constructor: still uses pack_into (no change to GFModel)
    template <int IR, int OR>
    ConstraintInterface(const GenericFunction<IR, OR>& t) {
        t.func.get().pack_into_constraint_interface(*this);
    }

    // Forward all solver methods to storage.get()
    void constraints(...) const { storage.get().constraints(...); }
    // ... (all SolverConstraintSpec and SizableSpec methods forwarded)
};
```

`ObjectiveConcept` and `ObjectiveInterface` follow the same pattern, additionally inheriting
`SolverObjectiveSpec::Concept` and implementing its virtuals (`objective`, `gradient`, `hessian`).

**`GFModel::pack_into_constraint_interface`** (out-of-line in `GenericFunction.h`) updates to
directly emplace into the new TypeStorage-based ConstraintInterface:
```cpp
// Before
void pack_into_constraint_interface(ConstraintInterface& ci) const override {
    ci = ConstraintInterface(this->data_);
}
// After
void pack_into_constraint_interface(ConstraintInterface& ci) const override {
    ci.storage.emplace<ConstraintModel<T>>(this->data_);
}
```

**Copy semantics note:** rubber_types `ConstraintInterface` copies shared a `shared_ptr` (O(1)).
TypeStorage copy deep-clones via `clone_into`. `ConstraintFunction`/`ObjectiveFunction` construct
their interfaces once per solve — not on solver iterations — so this is not a hot path.

**Dead capability removed:** `DeepCopySpecs<ObjectiveInterface, ConstraintInterface>` gave
`ObjectiveInterface` the ability to deep-copy itself into a `ConstraintInterface`. A codebase-wide
search confirms `deep_copy_into` is never called externally; this capability is dead and not
reproduced in the new design.

---

### Part 4 — GFStorage Refactor (`GFTypeErasure.h`)

`GFStorage<IR,OR>` is refactored to hold a `TypeStorage<GFConcept<IR,OR>>` member rather than
reimplementing SBO mechanics. Public API unchanged — `GenericFunction.h` call sites unaffected.

`GFConcept<IR,OR>` gains `clone_into(TypeStorage<GFConcept<IR,OR>>&)` as its clone virtual
(satisfying `TypeStorageBase`). The existing `clone_into_dynamic(GFStorage<-1,-1>&)` for cross-IR
upcast remains a `GFConcept`-only virtual, unchanged.

```cpp
template <int IR, int OR>
class GFStorage {
    TypeStorage<GFConcept<IR, OR>> storage_;
public:
    template <GFStorable<IR, OR> T>
    void emplace(T obj) {
        storage_.emplace<GFModel<IR, OR, std::decay_t<T>>>(std::move(obj));
    }
    // Copy/move/dtor: delegate to TypeStorage (= default)
    [[nodiscard]] bool         empty()   const noexcept { return storage_.empty(); }
    GFConcept<IR, OR>&         get()     const noexcept { return storage_.get(); }
    GFConcept<IR, OR>*         get_ptr() const noexcept { return storage_.get_ptr(); }
};
```

---

### Part 5 — Spec File Cleanup + Deletion List

**`DenseFunctionSpecs.h` and `SizingSpecs.h`:** delete dead `Model<Holder>` and
`ExternalInterface<Container>` inner structs from each spec. Keep `Concept` inner structs —
they remain the non-virtual bases for `GFConcept`, `ConstraintBase`, and `ObjectiveBase`.

**`SolverInterfaceSpecs.h`:** remove `#include "DeepCopySpecs.h"` and all `DeepCopySpecs`
usage. Strip dead `SolverConstraintSpec::Model`/`ExternalInterface` and
`SolverObjectiveSpec::Model`/`ExternalInterface`.

**Files deleted:**

| File | Reason |
|---|---|
| `src/Utils/TypeErasure.h` | rubber_types — fully replaced by TypeStorage |
| `src/VectorFunctions/VectorFunctionTypeErasure/DeepCopySpecs.h` | Only served rubber_types deep-copy; TypeStorage value semantics replace it |
| `notices/rubber_types-mit.txt` | No rubber_types code remains in the repo |

**Includes removed:**

| File | Include removed |
|---|---|
| `src/pch.h` | `#include "Utils/TypeErasure.h"` |
| `src/Utils/Tycho_Utils.h` | `#include "TypeErasure.h"` |
| `src/VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h` | `#include "DeepCopySpecs.h"` |

**Includes added:**

| File | Include added |
|---|---|
| `src/pch.h` | `#include "Utils/TypeStorage.h"` |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h` | `#include "ConditionalTypeErasure.h"` |

---

## Files Changed Summary

| File | Action |
|---|---|
| `src/Utils/TypeStorage.h` | **new** — TypeStorageBase concept + TypeStorage<C,N> |
| `src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h` | **new** — ConditionalBase<IR>, ConditionalModel<IR,T> |
| `src/Utils/TypeErasure.h` | **deleted** |
| `src/VectorFunctions/VectorFunctionTypeErasure/DeepCopySpecs.h` | **deleted** |
| `notices/rubber_types-mit.txt` | **deleted** |
| `src/Utils/Tycho_Utils.h` | Remove TypeErasure.h include |
| `src/pch.h` | Swap TypeErasure.h → TypeStorage.h |
| `src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h` | GFStorage wraps TypeStorage; GFConcept gains clone_into |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h` | pack_into_* bodies updated for new ConstraintInterface storage |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h` | Migrated to ConditionalTypeErasure.h |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericComparative.h` | Becomes alias for GenericConditional |
| `src/VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h` | ConstraintInterface + ObjectiveInterface migrated to TypeStorage; dead spec boilerplate stripped |
| `src/VectorFunctions/VectorFunctionTypeErasure/DenseFunctionSpecs.h` | Strip dead Model<>/ExternalInterface<> inner structs |
| `src/VectorFunctions/VectorFunctionTypeErasure/SizingSpecs.h` | Strip dead Model<>/ExternalInterface<> inner structs |

---

## Implementation Order

1. Write `TypeStorage.h`; verify it compiles in isolation with a trivial Concept stub
2. Write `ConditionalTypeErasure.h`; migrate `GenericConditional.h` and `GenericComparative.h`;
   run all 38 examples
3. Migrate `ConstraintInterface` + `ObjectiveInterface` in `SolverInterfaceSpecs.h`;
   update `GFModel::pack_into_*` bodies in `GenericFunction.h`; run all 38 examples
4. Refactor `GFStorage` to wrap `TypeStorage<GFConcept<IR,OR>>`; run all 38 examples
5. Strip dead `Model<>`/`ExternalInterface<>` boilerplate from Spec files; delete
   `DeepCopySpecs.h`, `TypeErasure.h`, `notices/rubber_types-mit.txt`; run all 38 examples

---

## Verification

- `ninja -j6 all` — clean build, no new warnings
- `python run_examples.py` — all 38 examples pass
- `./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp` — "Optimal Solution Found"
- `grep -r "rubber_types" src/` — zero matches
- `grep -r "TypeErasure.h" src/` — zero matches

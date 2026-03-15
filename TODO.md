# Tycho — Modernization TODO

## Completed PRs

| PR   | Description                                                       | Status            |
| ---- | ----------------------------------------------------------------- | ----------------- |
| PR 1 | Rename ASSET/asset_asrl → Tycho/tycho                             | **DONE** (merged) |
| PR 2 | Decouple C++ from Python bindings                                 | **DONE** (merged) |
| PR 3 | Migrate pybind11 → nanobind                                       | **DONE** (merged) |
| PR 4 | Separate extensions C++ from binding code                         | **DONE** (merged) |
| PR 5 | Add per-file provenance headers to all ASSET-derived source files | **DONE** (merged) |
| PR 6 | Type erasure refactor (flat vtable + C++20 bump)                  | **DONE** (merged) |
| PR 7 | Fix GFStorage performance regression (shared_ptr)                 | **DONE** (merged) |
| PR 8 | C++20 concept adoption in the CRTP hierarchy                      | **DONE** (implemented) |
| PR 9 | Full rubber_types elimination                                      | **DONE** (implemented) |

### PR 6+7 Summary

PR 6 replaced `rubber_types::TypeErasure` for `GenericFunction<IR,OR>` with a custom flat-vtable
system (`GFConcept<IR,OR>` + `GFModel<IR,OR,T>` + `GFStorage<IR,OR>`) and bumped C++ standard
to C++20. It also eliminated double-erasure in solver interfaces via `pack_into_*` dispatch.

PR 6's SBO + value semantics caused a ~15-20x regression. PR 7 fixed it by switching GFStorage
to `shared_ptr<GFConcept>`, restoring O(1) copies (1.56s vs 29.63s regression vs 1.91s baseline).

**Current GFStorage architecture:**
```
GFStorage<IR,OR>
  shared_ptr<GFConcept<IR,OR>>       (O(1) copy via refcount)
    GFConcept: flat vtable, non-virtual MI of Spec::Concepts
      GFModel<IR,OR,T>: stores T directly
```

**Remaining rubber_types usage** (eliminated in PR 9):
- `GenericConditional<IR>`, `GenericComparative<IR>` — integrator stop functions
- `ConstraintInterface`, `ObjectiveInterface` — solver interface wrappers
- Dead `Model<>`/`ExternalInterface<>` boilerplate in Spec files

---

> **PR 8 is complete** — implemented as specified below with no runtime behavior change.
> **PR 9 is complete** — rubber_types fully eliminated; `TypeErasure.h` and `DeepCopySpecs.h` deleted.

## PR 8 — C++20 Concept Adoption in the CRTP Hierarchy

**Status:** **DONE** (implemented)

### Goal

Apply C++20 concepts to the VectorFunction CRTP layer. No runtime behavior change.

### New File: `src/VectorFunctions/VectorFunctionConcepts.h`

**Group 1 — Scalar type concepts** (replace `DetectSuperScalar.h` type traits):
```cpp
template <typename T> concept IsSuperScalar = /* Eigen Array<Scalar,N,1> for N > 1 */;
template <typename T> concept IsDoubleScalar = std::same_as<T, double>;
```

**Group 2 — Static capability flags** (wrap existing `static const bool` pattern):
```cpp
template <typename T> concept Vectorizable   = requires { requires bool(T::IsVectorizable); };
template <typename T> concept LinearVF       = requires { requires bool(T::IsLinearFunction); };
template <typename T> concept HasDiagonalJac = requires { requires bool(T::HasDiagonalJacobian); };
template <typename T> concept IsGenericVF    = requires { requires bool(T::IsGenericFunction); };
```

**Group 3 — Sizing concepts:**
```cpp
template <typename T> concept StaticallySized  = requires { requires (T::InputRows > 0); requires (T::OutputRows > 0); };
template <typename T> concept DynamicallySized = requires { requires (T::InputRows < 0 || T::OutputRows < 0); };
```

**Group 4 — Expression composition** (turn silent dimension mismatches into clear compile errors):
```cpp
template <typename T>
concept VFSized = requires(const T& t) { { t.IRows() } -> std::same_as<int>; { t.ORows() } -> std::same_as<int>; };

template <typename Inner, typename Outer>
concept Composable = VFSized<Inner> && VFSized<Outer> && requires {
    requires (Inner::OutputRows < 0 || Outer::InputRows < 0 || Inner::OutputRows == Outer::InputRows);
};

template <typename F1, typename F2>
concept Stackable = VFSized<F1> && VFSized<F2> && requires {
    requires (F1::InputRows < 0 || F2::InputRows < 0 || F1::InputRows == F2::InputRows);
};
```

### Other Changes

- **`DetectSuperScalar.h`**: Replace `Is_SuperScalar<T>` struct with `IsSuperScalar<T>` concept;
  keep backward-compat `struct Is_SuperScalar : std::bool_constant<IsSuperScalar<T>> {}`
- **`ComputableBase.h`**: `Is_SuperScalar<Scalar>::value` → `IsSuperScalar<Scalar>`;
  `!Derived::IsVectorizable` → `!Vectorizable<Derived>`
- **`[[nodiscard]]`** on `IRows()`, `ORows()`, `is_linear()`, `input_domain()`, `name()`,
  `thread_safe()` in `InputOutputSize`, `DomainHolder`, `DenseFunctionBase`
- **`OperatorOverloads.h`**: Add `requires Composable<Inner, Outer>` on `operator()`;
  add `requires Stackable<F1, F2>` on `stack(...)` and `StackedOutputs`
- **`using enum DenseDerivativeMode`** at local scope in derivative strategy headers
- **`FunctionRegistry.h`**: `HasTychoBind<T>` concept on `Build_Register<T>` (requires
  `typename TychoBind<T>::BuildTag`)

### Explicit Non-Scope

- No deducing-this migration (separate future refactor)
- No concept constraints mid-CRTP-chain (circular dependency risk)
- No modules (PCH already effective)

### Verification

- `ninja -j6 all` — clean build, no new warnings
- `python scripts/run_examples.py` — 38/38 pass
- `brachistochrone_cpp` — "Optimal Solution Found"
- Trigger dimension mismatch → compiler names `Composable` constraint

---

## PR 9 — Full rubber_types Elimination

### Goal

Remove all remaining `rubber_types::TypeErasure` usage and delete `TypeErasure.h` + notice.

### Architecture

```
GFStorage<IR,OR>:           UNCHANGED — shared_ptr (PR 7)
GenericConditional<IR>:     TypeStorage<ConditionalBase<IR>>    (value semantics, not hot)
GenericComparative<IR>:     type alias for GenericConditional<IR>
ConstraintInterface:        TypeStorage<ConstraintBase>          (value semantics, built once per solve)
ObjectiveInterface:         TypeStorage<ObjectiveBase>           (value semantics, built once per solve)
```

### Performance Note

GFStorage stays with shared_ptr — GenericFunction copies happen thousands of times during
collocation transcription (lesson from PR 6/7). Conditionals, ConstraintInterface, and
ObjectiveInterface are NOT on hot copy paths: conditionals are integrator stop functions;
solver interfaces are constructed once per solve setup and stored in vectors.

---

### Task 1: `src/Utils/TypeStorage.h`

General-purpose SBO container with value semantics. Replaces `rubber_types::TypeErasure` for
all remaining use sites. GFStorage does NOT use this (it keeps shared_ptr).

`TypeStorage` is the canonical value-semantic type erasure container going forward — any future
use site that needs polymorphism without heap-allocation overhead should reach for this rather than
re-implementing the pattern.

**Convention:** The base class `C` must declare
`virtual void clone_into(TypeStorage<C, SBO_CAP>&) const = 0`. Each use site defines its own
`Model<T>` that implements this as `s.emplace<Model<T>>(data_)`.

**Key difference from GFStorage:** `emplace<Model, T>(obj)` takes the Model type explicitly
(each use site has its own Model: `ConditionalModel<IR,T>`, `ConstraintModel<T>`, etc.).

```cpp
template <typename C, std::size_t N>
concept TypeStorageBase = requires(const C& c, TypeStorage<C, N>& s) { { c.clone_into(s) }; };

template <typename C, std::size_t SBO_CAP = 128>
class TypeStorage {
    static constexpr std::size_t SBO_ALIGN = alignof(std::max_align_t);
    enum class Kind : uint8_t { Empty, Inline, Heap };

    Kind kind_ = Kind::Empty;
    union { alignas(SBO_ALIGN) std::byte buf_[SBO_CAP]; C* ptr_; };

public:
    TypeStorage() noexcept = default;
    template <typename Model, typename T> void emplace(T obj);  // SBO or heap
    TypeStorage(const TypeStorage& o);             // deep-copy via clone_into
    TypeStorage(TypeStorage&& o) noexcept;         // memcpy (inline) or ptr swap (heap)
    TypeStorage& operator=(const TypeStorage&);
    TypeStorage& operator=(TypeStorage&&) noexcept;
    ~TypeStorage();
    [[nodiscard]] bool empty() const noexcept;
    C& get() const noexcept;
    C* get_ptr() const noexcept;
};
```

**Files:** Create `src/Utils/TypeStorage.h`. In `src/pch.h`, swap `#include "Utils/TypeErasure.h"`
→ `#include "Utils/TypeStorage.h"`.

---

### Task 2: Migrate `GenericConditional` / `GenericComparative`

Both are type-erased boolean predicates (integrator stop functions) with the same 3-method
interface. They share one Concept/Model pair.

**New file: `src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h`**

```cpp
template <int IR, typename T>
concept ConditionalStorable = requires(const T& t,
    const Eigen::Ref<const Eigen::Matrix<double, IR, 1>>& x) {
    { t.name() } -> std::same_as<std::string>;
    { t.IRows() } -> std::same_as<int>;
    { t.compute(x) } -> std::same_as<bool>;
};

template <int IR>
struct ConditionalBase {
    using InType = Eigen::Ref<const Eigen::Matrix<double, IR, 1>>;
    virtual ~ConditionalBase() = default;
    virtual std::string name() const = 0;
    virtual int IRows() const = 0;
    virtual bool compute(const Eigen::MatrixBase<InType>& x) const = 0;
    virtual void clone_into(TypeStorage<ConditionalBase<IR>>&) const = 0;
};

template <int IR, typename T>
struct ConditionalModel final : ConditionalBase<IR> {
    T data_;
    explicit ConditionalModel(T t) : data_(std::move(t)) {}
    // ... delegates all virtuals to data_, clone_into calls s.emplace<ConditionalModel<IR,T>>(data_)
};
```

**`GenericConditional.h`:** Replace rubber_types guts with `TypeStorage<ConditionalBase<IR>> storage`.
Constructor uses `storage.emplace<ConditionalModel<IR, T>>(t)`. Copy/move handled by TypeStorage.

**`GenericComparative.h`:** Replace with `template <int IR> using GenericComparative = GenericConditional<IR>;`

**Potential issues:** If any call site uses `obj.get_container()` or `reset_container()` (old
rubber_types API), it will fail to compile — remove those calls.

---

### Task 3: Migrate `ConstraintInterface` / `ObjectiveInterface`

The largest change. These are the types PSIOPT calls at solve time (hot path for evaluation,
but NOT hot for copy). Virtual signatures that PSIOPT sees are unchanged.

**ConstraintBase** — flat abstract base, non-virtual MI (same pattern as GFConcept):
```cpp
struct ConstraintBase : SolverConstraintSpec::Concept, SizableSpec::Concept {
    virtual ~ConstraintBase() = default;
    virtual void clone_into(TypeStorage<ConstraintBase>&) const = 0;
};

template <typename T>
struct ConstraintModel final : ConstraintBase {
    T data_;
    // ... delegates all SolverConstraintSpec + SizableSpec virtuals to data_
    void clone_into(TypeStorage<ConstraintBase>& s) const override {
        s.emplace<ConstraintModel<T>>(data_);
    }
};
```

**ConstraintInterface** — value-semantic wrapper:
```cpp
struct ConstraintInterface {
    TypeStorage<ConstraintBase> storage;
    ConstraintInterface() = default;
    template <class T, /* SFINAE: not Eigen */> ConstraintInterface(const T& t) {
        storage.emplace<ConstraintModel<std::decay_t<T>>>(t);
    }
    template <int IR, int OR>
    ConstraintInterface(const GenericFunction<IR, OR>& t) {
        t.func.get().pack_into_constraint_interface(*this);  // unchanged dispatch
    }
    // Forward all solver calls to storage.get()
};
```

**ObjectiveBase/ObjectiveModel/ObjectiveInterface** — same pattern, additionally inherits
`SolverObjectiveSpec::Concept` and implements `objective`, `objective_gradient`,
`objective_gradient_hessian`.

**pack_into_* body updates in `GFTypeErasure.h`:**
```cpp
// Before (constructs fresh rubber_types TypeErasure from T — heap alloc + copy)
void pack_into_constraint_interface(ConstraintInterface& ci) const override {
    ci = ConstraintInterface(this->data_);
}
// After (direct emplace into TypeStorage — avoids extra copy through CI constructor)
void pack_into_constraint_interface(ConstraintInterface& ci) const override {
    ci.storage.emplace<ConstraintModel<T>>(this->data_);
}
```

Same change for `pack_into_objective_interface` → `oi.storage.emplace<ObjectiveModel<T>>(...)`.

Note: `ConstraintModel` and `ObjectiveModel` are defined in `SolverInterfaceSpecs.h`, which
`GFTypeErasure.h` already includes — no additional includes needed.

**Dead code to delete:**
- `SolverInterfaceSelector` template (defined but never used anywhere)
- `DeepCopySpecs<ObjectiveInterface, ConstraintInterface>` — `deep_copy_into` is never called;
  dead capability not reproduced
- `SolverConstraintSpec::Model<>` and `ExternalInterface<>` inner structs
- `SolverObjectiveSpec::Model<>` and `ExternalInterface<>` inner structs
- `#include "DeepCopySpecs.h"` from `SolverInterfaceSpecs.h`

---

### Task 4: Strip dead boilerplate + delete rubber_types files

**`DenseFunctionSpecs.h`:** Delete dead `Model<Holder>` (~200 lines of virtual overrides using
`rubber_types::model_get`) and `ExternalInterface<Container>`. Keep only the outer
`DenseFunctionSpec<IR,OR>` struct, its type aliases, and `Concept`.

**`SizingSpecs.h`:** Same — delete `Model<Holder>` and `ExternalInterface<Container>`. Keep
only `SizableSpec` struct and `Concept`.

**`Tycho_Utils.h`:** Remove `#include "TypeErasure.h"`.

**Delete files:**
```bash
git rm src/Utils/TypeErasure.h
git rm src/VectorFunctions/VectorFunctionTypeErasure/DeepCopySpecs.h
```

> **Note:** `notices/rubber_types-mit.txt` is **not** deleted. CLAUDE.md prohibits deleting files
> under `notices/`. The notice stays even after the dependency is removed.

---

### Files Changed Summary

| File                                                                     | Action                                       |
| ------------------------------------------------------------------------ | -------------------------------------------- |
| `src/Utils/TypeStorage.h`                                                | **new** — SBO container with value semantics |
| `src/VectorFunctions/VectorFunctionTypeErasure/ConditionalTypeErasure.h` | **new**                                      |
| `src/Utils/TypeErasure.h`                                                | **deleted**                                  |
| `src/VectorFunctions/VectorFunctionTypeErasure/DeepCopySpecs.h`          | **deleted**                                  |
| `src/pch.h`                                                              | Swap TypeErasure.h → TypeStorage.h           |
| `src/Utils/Tycho_Utils.h`                                                | Remove TypeErasure.h include                 |
| `src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h`          | Update pack_into_* bodies                    |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h`     | Use ConditionalTypeErasure                   |
| `src/VectorFunctions/VectorFunctionTypeErasure/GenericComparative.h`     | Becomes alias                                |
| `src/VectorFunctions/VectorFunctionTypeErasure/SolverInterfaceSpecs.h`   | Full migration + cleanup                     |
| `src/VectorFunctions/VectorFunctionTypeErasure/DenseFunctionSpecs.h`     | Strip dead Model/ExternalInterface           |
| `src/VectorFunctions/VectorFunctionTypeErasure/SizingSpecs.h`            | Strip dead Model/ExternalInterface           |
| `src/Bindings/TychoModule.cpp`                                           | Remove `using namespace rubber_types;`       |

### Verification

- `ninja -j6 all` — clean build, no new warnings
- `python scripts/run_examples.py` — 38/38 pass
- `brachistochrone_cpp` — "Optimal Solution Found"
- `grep -r "rubber_types" src/` — zero matches (only `notices/rubber_types-mit.txt` remains)
- `grep -r "TypeErasure.h" src/` — zero matches
- `grep -r "DeepCopySpecs" src/` — zero matches

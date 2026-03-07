# Tycho — Modernization TODO

## Completed PRs

| PR | Description | Status |
|----|-------------|--------|
| PR 1 | Rename ASSET/asset_asrl → Tycho/tycho | **DONE** (merged) |
| PR 2 | Decouple C++ from Python bindings | **DONE** (merged) |
| PR 3 | Migrate pybind11 → nanobind | **DONE** (merged) |
| PR 4 | Separate extensions C++ from binding code | **DONE** (merged) |
| PR 5 | Add per-file provenance headers to all ASSET-derived source files | **DONE** (merged) |

---

## PR 6 — Type Erasure Refactor: Flat Concept + SBO + C++20 Bump

### Goal

Replace the `rubber_types::TypeErasure` composition used for `GFTE<IR,OR>` (GenericFunction's
internal type erasure) with a purpose-built, flat implementation, and bump the project to C++20.
The current design has three performance costs:

1. **Virtual inheritance overhead** -- `MergeSpecs` chains 5 specs via `public virtual
   ConceptOf<Specs>...`, adding vbase offset tables and extra indirection to every vtable lookup.
2. **Heap allocation on every construction** -- `shared_ptr<const Concept>` always calls
   `make_shared<Model<Holder<T>>>()`, even for tiny built-in types like `Segment` or `Norm`.
3. **Double-erasure in ConstraintInterface** -- `ConstraintInterface(GenericFunction<IR,OR>)`
   stores `t.func` (GFTE) rather than the underlying concrete type, causing two virtual dispatches
   per solver call instead of one.

Additionally, the new code written in this PR is the ideal first site for targeted C++20 adoption:
a `GFStorable` concept constrains `GFStorage::emplace` and replaces opaque template errors with
actionable diagnostics; `std::construct_at`, `[[nodiscard]]`, and `using enum` improve the
`GFStorage` implementation. Broader CRTP-level concept adoption is deferred to PR 7.

The fix is isolated to the GFTE layer. `rubber_types::TypeErasure` is left in place for
`ConstraintInterface`, `ObjectiveInterface`, and all other uses. The existing Spec files
(`DenseFunctionSpecs.h`, `SizingSpecs.h`, `SolverInterfaceSpecs.h`) are **not modified** --
they remain the canonical reference for which methods belong to which concern group.

### Target Architecture

```
Current:
  GFTE (rubber_types::TypeErasure<5 merged specs>)
    -> shared_ptr<Concept>  (heap, refcounted)
    -> Concept uses virtual MI  (multiple vbase pointers per Model)
  ConstraintInterface -> stores GFTE -> 2 virtual dispatches per solver call

Target:
  GFStorage<IR,OR>  (inline buffer or heap, value/clone semantics)
    -> GFConcept<IR,OR>: inherits Spec::Concepts non-virtually -> flat vtable, no vbase tables
    -> GFHolder<T> + GFModel<IR,OR,T>: stores T directly, no rubber_types middleman
  ConstraintInterface -> stores T directly via pack_into -> 1 virtual dispatch per solver call
```

### CMake: Bump to C++20

In `CMakeLists.txt`, change the C++ standard target:

```cmake
# Before
target_compile_features(tycho_core INTERFACE cxx_std_17)

# After
target_compile_features(tycho_core INTERFACE cxx_std_20)
```

Apply the same change to `pch`, `pch_bindings`, and any other targets that set a standard
explicitly. Verify with `ninja -j6 all` that the full build is clean under C++20.

### New File: `src/VectorFunctions/VectorFunctionTypeErasure/GFTypeErasure.h`

This file has five clearly separated sections, each with a single responsibility. The existing
Spec files (`DenseFunctionSpecs.h`, `SizingSpecs.h`, `SolverInterfaceSpecs.h`) are **not
modified** -- they remain the canonical reference for which methods belong to which concern group.

---

#### Section 1: `GFHolder<T>` -- owns one T, nothing else

```cpp
template <class T>
struct GFHolder {
    T data_;
    explicit GFHolder(T t) : data_(std::move(t)) {}
};
```

---

#### Section 2: `GFStorable` concept -- constrains what can be stored

A C++20 concept that constrains `GFStorage::emplace<T>` and `GenericFunction(const T&)`.
Instead of an opaque 40-line template backtrace when T is wrong, the compiler emits:
`"T does not satisfy GFStorable<6, 3>"`.

```cpp
template <typename T, int IR, int OR>
concept GFStorable = requires(const T& t) {
    { t.IRows() }         -> std::same_as<int>;
    { t.ORows() }         -> std::same_as<int>;
    { t.is_linear() }     -> std::same_as<bool>;
    { t.input_domain() };
    { t.name() }          -> std::same_as<std::string>;
    { T::IsVectorizable } -> std::convertible_to<bool>;
};
// GFModel compilation enforces the full virtual interface; this concept catches the
// most common mistake (wrong type entirely) at the call site before instantiation.
```

---

#### Section 3: `GFConcept<IR, OR>` -- flat abstract class via non-virtual inheritance

`GFConcept` inherits the existing Spec `Concept` classes **non-virtually**. This is valid because
`DenseFunctionSpec::Concept`, `SizableSpec::Concept`, and `SolverConstraintSpec::Concept` share
no common base -- so non-virtual multiple inheritance is diamond-free and produces a single flat
vtable with no vbase offset overhead. The Spec files remain the single source of truth for their
method signatures; `GFConcept` adds only the three new virtuals needed for copy/pack operations.

```cpp
// Forward declarations to break the circular dependency with SolverInterfaceSpecs.h
// (full definitions available by the time GFModel bodies are instantiated in GenericFunction.h)
struct ConstraintInterface;
struct ObjectiveInterface;

template <int IR, int OR>
struct GFConcept
    : DenseFunctionSpec<IR, OR>::Concept,   // non-virtual: no shared base -> no diamond
      SizableSpec::Concept,
      SolverConstraintSpec::Concept {

    virtual ~GFConcept() = default;

    // New virtuals not covered by any existing Spec:
    virtual void clone_into(GFStorage<IR, OR>&)         const = 0;  // same-type copy
    virtual void clone_into_dynamic(GFStorage<-1, -1>&) const = 0;  // upcast copy
    virtual void pack_into_constraint_interface(ConstraintInterface&) const = 0;
};

// OR == 1: additionally inherits SolverObjectiveSpec::Concept (non-virtually)
template <int IR>
struct GFConcept<IR, 1>
    : DenseFunctionSpec<IR, 1>::Concept,
      SizableSpec::Concept,
      SolverConstraintSpec::Concept,
      SolverObjectiveSpec::Concept {

    virtual ~GFConcept() = default;

    virtual void clone_into(GFStorage<IR, 1>&)           const = 0;
    virtual void clone_into_dynamic(GFStorage<-1, -1>&)  const = 0;
    virtual void pack_into_constraint_interface(ConstraintInterface&) const = 0;
    virtual void pack_into_objective_interface(ObjectiveInterface&)   const = 0;
};
```

---

#### Section 4: `GFModel<IR, OR, T>` -- implements all overrides, grouped by spec

Override bodies are grouped with section comments matching the source Spec files. If a new pure
virtual is added to any Spec, the compiler fails to instantiate `GFModel` (abstract class),
pointing directly to the right section. `pack_into_*` bodies are **declared here but defined
out-of-line at the bottom of `GenericFunction.h`** after `ConstraintInterface` and
`ObjectiveInterface` are fully defined, breaking the circular include without any runtime cost.

```cpp
template <int IR, int OR, GFStorable<IR, OR> T>   // C++20: concept constraint on T
struct GFModel final : GFHolder<T>, GFConcept<IR, OR> {
    using GFHolder<T>::GFHolder;

    // ---- DenseFunctionSpec overrides (see DenseFunctionSpecs.h) ----
    DomainMatrix input_domain()       const override { return this->data_.input_domain(); }
    bool         is_linear()          const override { return this->data_.is_linear(); }
    void enable_vectorization(bool b) const override { this->data_.enable_vectorization(b); }
    void compute(ConstVectorBaseRef<InType> x,
                 ConstVectorBaseRef<OutType> fx_) const override { this->data_.compute(x, fx_); }
    // ... (remaining DenseFunctionSpec overrides, same pattern) ...

    // ---- SizableSpec overrides (see SizingSpecs.h) ----
    std::string name()        const override { return this->data_.name(); }
    int         IRows()       const override { return this->data_.IRows(); }
    int         ORows()       const override { return this->data_.ORows(); }
    bool        thread_safe() const override { return this->data_.thread_safe(); }

    // ---- SolverConstraintSpec overrides (see SolverInterfaceSpecs.h) ----
    void constraints(...) const override { this->data_.constraints(...); }
    // ... (remaining SolverConstraintSpec overrides, same pattern) ...

    // ---- GFTypeErasure: copy and pack operations ----
    void clone_into(GFStorage<IR, OR>& s)        const override { s.emplace(this->data_); }
    void clone_into_dynamic(GFStorage<-1,-1>& s) const override { s.emplace(this->data_); }
    void pack_into_constraint_interface(ConstraintInterface&) const override; // body in GenericFunction.h
};
```

`sizeof(GFModel<IR,OR,T>)` = 8 (vptr) + `sizeof(T)`. No `Holder` wrapper. No
`rubber_types::model_get`.

---

#### Section 5: `GFStorage<IR, OR>` -- SBO container, value semantics

`GFStorage` is a **value type** with deep-copy semantics (clone-on-copy via
`GFConcept::clone_into`). This replaces the `shared_ptr<const Concept>` shared-ownership model.
For SBO-eligible types, clone is a placement-new into the inline buffer -- no heap allocation.
The explicit Rule-of-Five makes the ownership model unambiguous. C++20 features applied:
`GFStorable` concept on `emplace`, `std::construct_at` replacing raw placement new,
`[[nodiscard]]` with reason strings on key accessors, and `using enum Kind` for readability.

```cpp
// SBO_CAP is sized to cover all common built-in VF types without heap fallback.
// sizeof(GFModel<IR,OR,T>) = sizeof(vptr) + sizeof(T); most built-ins are 8-72 bytes.
// 128 bytes gives comfortable headroom. Widen if instrumentation reveals misses.
template <int IR, int OR>
class GFStorage {
    static constexpr std::size_t SBO_CAP   = 128;
    static constexpr std::size_t SBO_ALIGN = alignof(std::max_align_t);

    using Concept = GFConcept<IR, OR>;
    enum class Kind : uint8_t { Empty, Inline, Heap };

    Kind kind_ = Kind::Empty;
    union {
        alignas(SBO_ALIGN) std::byte buf_[SBO_CAP];
        Concept*                     ptr_;
    };

    Concept* concept() const noexcept {
        using enum Kind;
        return kind_ == Inline
            ? std::launder(reinterpret_cast<Concept*>(buf_))
            : ptr_;
    }

public:
    GFStorage() noexcept = default;

    // C++20: concept constraint gives "T does not satisfy GFStorable<IR,OR>" on bad types
    template <GFStorable<IR, OR> T>
    void emplace(T obj) {
        using Model = GFModel<IR, OR, std::decay_t<T>>;
        destroy();
        if constexpr (sizeof(Model) <= SBO_CAP && alignof(Model) <= SBO_ALIGN) {
            std::construct_at(reinterpret_cast<Model*>(buf_), std::move(obj)); // C++20
            kind_ = Kind::Inline;
        } else {
            ptr_ = new Model(std::move(obj));
            kind_ = Kind::Heap;
        }
    }

    // Deep-copy via virtual clone_into (placement-new for inline types, heap alloc for large ones)
    GFStorage(const GFStorage& o)            { if (!o.empty()) o.concept()->clone_into(*this); }
    GFStorage(GFStorage&& o) noexcept;       // moves ptr or memcpys inline buf, sets o to Empty
    GFStorage& operator=(const GFStorage& o);
    GFStorage& operator=(GFStorage&& o) noexcept;
    ~GFStorage() { destroy(); }

    // [[nodiscard]] with reason: calling get() on empty GFStorage is undefined behaviour
    [[nodiscard("Check empty() before calling get()")]]
    bool     empty()   const noexcept { return kind_ == Kind::Empty; }
    Concept& get()     const noexcept { return *concept(); }
    Concept* get_ptr() const noexcept { return concept(); }

private:
    void destroy() noexcept {
        using enum Kind;
        if      (kind_ == Inline) concept()->~Concept();
        else if (kind_ == Heap)   delete ptr_;
        kind_ = Kind::Empty;
    }
};
```

---

### SBO Buffer Coverage

| Concrete type | `sizeof(T)` | `sizeof(GFModel)` | Inline? |
|---|---|---|---|
| `Segment<6,3,0>` | ~8 | ~16 | yes |
| `Arguments<14>` | ~8 | ~16 | yes |
| `Norm<3>` | ~8 | ~16 | yes |
| `Scaled<Norm<3>>` | ~24 | ~32 | yes |
| `PythonFunction` | ~64 | ~72 | yes |
| Large expression tree | varies | varies | heap fallback |

### Eliminating the Double-Erasure in ConstraintInterface

`pack_into_constraint_interface` is declared in `GFConcept` (forward declaration of
`ConstraintInterface` suffices) and its body is defined out-of-line at the bottom of
`GenericFunction.h` after `ConstraintInterface` is fully defined. In `SolverInterfaceSpecs.h`,
the GenericFunction constructor changes to call through it:

```cpp
// Before: stores GFTE -- two virtual dispatches per solver call
template <int IR, int OR>
ConstraintInterface(const GenericFunction<IR, OR> &t) : Base(t.func) {}

// After: stores concrete T via pack_into -- one virtual dispatch per solver call
template <int IR, int OR>
ConstraintInterface(const GenericFunction<IR, OR> &t) {
    t.func.get().pack_into_constraint_interface(*this);
}
```

`GFModel<IR,OR,T>::pack_into_constraint_interface` calls `ConstraintInterface(this->data_)`,
storing T directly. The solver call chain becomes:

```
Before:  CI virtual dispatch -> GFTE ExternalInterface -> GFTE virtual dispatch -> T
After:   CI virtual dispatch -> T
```

Same fix applies to `ObjectiveInterface` via `pack_into_objective_interface`.

### Copy Semantics Change

The current `GenericFunction(const GenericFunction<IR,OR>&)` shares the underlying `shared_ptr`
(O(1)). `GFStorage` clone-on-copy calls `clone_into`, which calls `s.emplace(this->data_)` --
a `std::construct_at` placement into the SBO buffer for small types (a few cache lines, no heap
allocation) or a heap allocation for large types. No code in the library deliberately relies on
two GenericFunction copies sharing mutable state (`enable_vectorization` already behaves
inconsistently under shared ownership). Clone-on-copy is correct.

### Files Changed

| File | Action |
|---|---|
| `CMakeLists.txt` | Bump C++ standard to C++20 on all targets |
| `VectorFunctionTypeErasure/GFTypeErasure.h` | **new** -- GFStorable concept, GFHolder, GFConcept, GFModel, GFStorage |
| `VectorFunctionTypeErasure/GenericFunction.h` | Replace `TE func` with `GFStorage`; define `pack_into_*` bodies out-of-line at bottom |
| `VectorFunctionTypeErasure/SolverInterfaceSpecs.h` | Update ConstraintInterface and ObjectiveInterface GenericFunction constructors |
| `Utils/TypeErasure.h` | **no change** |
| `DenseFunctionSpecs.h`, `SizingSpecs.h`, `DeepCopySpecs.h` (specs) | **no change** -- Spec files remain canonical |

### Implementation Order

1. Bump CMakeLists.txt to C++20; verify clean build with `ninja -j6 all`
2. Write `GFTypeErasure.h` with all five sections; verify it compiles against a test instantiation
   (instantiate `GFModel<6,3,Segment<6,3,0>>` and `GFStorage<6,3>` to shake out any issues)
3. Swap `GFTE`/`GenTE` in `GenericFunction.h` for `GFStorage`; run all 38 examples
4. Update `ConstraintInterface`/`ObjectiveInterface` constructors; run all 38 examples again
5. Tune `SBO_CAP` -- add a debug-mode log in `emplace()` for heap fallbacks and widen if needed

### Verification

- `ninja -j6 all` -- clean build under C++20
- `python run_examples.py` -- all 38 examples pass
- `./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp` -- reports "Optimal Solution Found"

---

## PR 7 -- Broader C++20 Concept Adoption in the CRTP Hierarchy

### Goal

Apply C++20 concepts systematically to the VectorFunction CRTP layer. This PR does not change
any runtime behavior -- every `if constexpr` branch and expression template composition rule that
exists today continues to work identically. The gains are:

1. **Actionable compile errors** -- dimension mismatches in `NestedFunction` and `StackedOutputs`
   currently produce cryptic Eigen assertion failures or deep template backtraces. Concept
   constraints catch them at the composition call site with a clear message.
2. **Eliminated type trait boilerplate** -- `Is_SuperScalar<T>::value` and similar structs in
   `DetectSuperScalar.h` are replaced by proper concepts, used directly in `if constexpr` and
   `requires` clauses throughout `ComputableBase`.
3. **Documented interface contracts** -- the implicit requirements on `Derived` in the CRTP bases
   become explicit `requires` expressions that serve as executable documentation.
4. **`[[nodiscard]]` on metadata methods** -- `IRows()`, `ORows()`, `is_linear()`,
   `input_domain()`, and `name()` are pure queries; accidentally discarding them is always a bug.

### Explicit Non-Scope

- **No deducing-this migration** -- already documented in `doc/VectorFunction.md` as a separate,
  large future refactor.
- **No concept constraints on the deep CRTP base chain itself** (`VectorFunction`,
  `DenseFunctionBase`, `DenseDerivatives`, etc.) -- the risk of circular concept dependencies in
  a deeply layered CRTP hierarchy is high. Concepts are applied at call sites and composition
  boundaries, not mid-inheritance-chain.
- **No changes to derivative strategy dispatch** -- the `DenseFirstDerivatives` /
  `DenseSecondDerivatives` partial specializations on `DenseDerivativeModes` already work
  correctly and are not improved by concept constraints.
- **No modules** -- the PCH infrastructure already addresses build times effectively.

### New File: `src/VectorFunctions/VectorFunctionConcepts.h`

Single source of truth for all VectorFunction-related concepts. Included by the headers that need
them. Organized into four groups:

#### Group 1: Scalar type concepts (replace `DetectSuperScalar.h` type traits)

```cpp
// Replaces Is_SuperScalar<T>::value throughout ComputableBase and elsewhere.
// T is a SuperScalar if it is an Eigen Array<Scalar, N, 1> for some N > 1.
template <typename T>
concept IsSuperScalar = /* see DetectSuperScalar.h -- replaces Is_SuperScalar<T> */;

// Plain double scalar (the non-SuperScalar case)
template <typename T>
concept IsDoubleScalar = std::same_as<T, double>;
```

#### Group 2: Static capability flag concepts (replace `static const bool` pattern)

These wrap the existing static bool flags as concepts, improving readability at call sites
without changing the underlying mechanism.

```cpp
template <typename T>
concept Vectorizable = requires { requires bool(T::IsVectorizable); };

template <typename T>
concept LinearVF = requires { requires bool(T::IsLinearFunction); };

template <typename T>
concept HasDiagonalJac = requires { requires bool(T::HasDiagonalJacobian); };

template <typename T>
concept IsGenericVF = requires { requires bool(T::IsGenericFunction); };
```

#### Group 3: Sizing concepts

```cpp
// IRows/ORows are known at compile time and positive
template <typename T>
concept StaticallySized = requires {
    requires (T::InputRows  > 0);
    requires (T::OutputRows > 0);
};

template <typename T>
concept DynamicallySized = requires {
    requires (T::InputRows < 0 || T::OutputRows < 0);
};
```

#### Group 4: Expression composition compatibility concepts

These are the highest-value concepts in this PR -- they turn silent dimension mismatches into
clear compile errors at the expression-building call site.

```cpp
// VFSized<T>: T has runtime IRows()/ORows() accessors (minimum interface for composition)
template <typename T>
concept VFSized = requires(const T& t) {
    { t.IRows() } -> std::same_as<int>;
    { t.ORows() } -> std::same_as<int>;
};

// Composable<Inner, Outer>: Inner.ORows == Outer.IRows (for NestedFunction / operator())
// Dynamic sizes (-1) are exempt -- checked at runtime by the existing assertions.
template <typename Inner, typename Outer>
concept Composable = VFSized<Inner> && VFSized<Outer> && requires {
    requires (Inner::OutputRows < 0 || Outer::InputRows  < 0 ||
              Inner::OutputRows == Outer::InputRows);
};

// Stackable<F1, F2>: same input dimension (for StackedOutputs and stack operators)
template <typename F1, typename F2>
concept Stackable = VFSized<F1> && VFSized<F2> && requires {
    requires (F1::InputRows < 0 || F2::InputRows < 0 ||
              F1::InputRows == F2::InputRows);
};
```

### Changes by File

#### `DetectSuperScalar.h`

Replace the `Is_SuperScalar<T>` struct with the `IsSuperScalar<T>` concept from
`VectorFunctionConcepts.h`. Keep a backward-compatible alias for any remaining call sites:

```cpp
// Backward-compat shim -- remove once all call sites are updated
template <class T>
struct Is_SuperScalar : std::bool_constant<IsSuperScalar<T>> {};
```

Update all `Is_SuperScalar<Scalar>::value` references in `ComputableBase.h` to
`IsSuperScalar<Scalar>`.

#### `ComputableBase.h`

Replace static-bool dispatch with concept-based dispatch:

```cpp
// Before
if constexpr (!Derived::IsVectorizable) {
    if constexpr (Is_SuperScalar<Scalar>::value) { ... }
}

// After
if constexpr (!Vectorizable<Derived>) {
    if constexpr (IsSuperScalar<Scalar>) { ... }
}
```

Add `[[nodiscard]]` to the pure metadata methods on `InputOutputSize`:

```cpp
[[nodiscard]] int IRows() const noexcept;
[[nodiscard]] int ORows() const noexcept;
```

And on `DomainHolder`:

```cpp
[[nodiscard]] DomainMatrix input_domain() const;
```

#### `DenseFunctionBase.h`

Add `[[nodiscard]]` to metadata methods:

```cpp
[[nodiscard]] bool        is_linear()    const noexcept;
[[nodiscard]] std::string name()         const;
[[nodiscard]] bool        thread_safe()  const noexcept;
```

#### `OperatorOverloads.h`

Add `Composable` and `Stackable` concept constraints to the expression-building free functions
and `operator()`:

```cpp
// Before
template <class Outer, class Inner>
auto operator()(const Outer& outer, const Inner& inner) { ... }

// After
template <class Outer, class Inner>
    requires Composable<Inner, Outer>
auto operator()(const Outer& outer, const Inner& inner) { ... }
```

Apply `Stackable` similarly to `stack(...)` and the `StackedOutputs` variadic constructors.

#### `DenseFunctionBase.h` -- `using enum DenseDerivativeModes`

In files that use `DenseDerivativeModes` values heavily (the derivative strategy headers), add
`using enum DenseDerivativeModes;` at local scope to reduce verbosity:

```cpp
// Before
if (Jm == DenseDerivativeModes::AutodiffFwd) { ... }

// After (with using enum at function scope)
using enum DenseDerivativeModes;
if (Jm == AutodiffFwd) { ... }
```

#### `Bindings/FunctionRegistry.h`

Add a concept check on `Build_Register<T>` to give a clear error when `TychoBind<T>` has not
been specialized:

```cpp
template <typename T>
concept HasTychoBind = requires { typename TychoBind<T>::BuildTag; };
// BuildTag is a trivial tag type added to every TychoBind specialization.

template <HasTychoBind T>
static void Build_Register(nb::module_& m, const std::string& name) {
    TychoBind<T>::Build(m, name);
}
```

### Implementation Order

1. Write `VectorFunctionConcepts.h`; ensure it compiles in isolation
2. Update `DetectSuperScalar.h`; update `ComputableBase.h` call sites; run all 38 examples
3. Add `[[nodiscard]]` to metadata methods; run all 38 examples (expect zero warnings with
   `-Wno-unused-result` removed from build flags, or audit discarded results)
4. Add concept constraints to `OperatorOverloads.h`; verify expression composition still works
5. Add `using enum DenseDerivativeModes` where beneficial; run all 38 examples
6. Add `HasTychoBind` concept to `FunctionRegistry.h`; run all 38 examples

### Verification

- `ninja -j6 all` -- clean build, no new warnings
- `python run_examples.py` -- all 38 examples pass
- `./build/examples/cpp_examples/brachistochrone/brachistochrone_cpp` -- reports "Optimal Solution Found"
- Intentionally trigger a dimension mismatch in a test expression (e.g., compose two functions
  with mismatched IR/OR) and confirm the compiler error names the `Composable` constraint

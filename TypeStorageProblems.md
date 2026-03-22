# TypeStorage memcpy-move Issues

## Background

`TypeStorage<C, SBO_CAP>` (`src/Utils/TypeStorage.h`) is an SBO (Small Buffer Optimization)
container that stores polymorphic objects either inline (in a 128-byte buffer) or on the heap.
**Inline objects are moved via `memcpy`**, which requires the stored type to be trivially
relocatable (no self-referential pointers).

This assumption is violated by two standard library types commonly embedded in stored models:

1. **`std::string`** — libstdc++ SSO has a pointer into the object itself
2. **`std::map`** — both libstdc++ and libc++ have a sentinel node with self-referential pointers

Whether a given `Model` type triggers the bug depends on whether it fits in the 128-byte SBO.
libc++ types are significantly smaller than libstdc++ types, so models that safely exceed 128
bytes on libstdc++ (going to the heap path) can fit inline on libc++ (triggering the memcpy path).

## Issue 1: `ConditionalStatement::name_` (FIXED — commit `404c983`)

**Affected stdlib:** libstdc++ (GCC)
**Status:** Fixed by removing the dead `std::string name_` member

`ConditionalStatement` had an unused `std::string name_` member (never written to, `name()`
always returned empty). `ConditionalModel<-1, ConditionalStatement<Segment<-1,1,-1>, Segment<-1,1,-1>>>`
was 120 bytes — fits in the 128-byte SBO. On libstdc++, `std::string` SSO has a self-referential
pointer, so memcpy-moving it corrupted the pointer. The destructor then called `free()` on a
stack address.

**Symptoms:** `free(): invalid size`, `double free or corruption (out)`, segfault during
cleanup of `GenericConditional` objects. Triggered by any conditional expression (e.g., `m > 1`).

**Fix:** Removed the dead `std::string name_` and `std::string name_` from `ConditionalStatement`
and `ConstantConditional`. The `name()` method now returns `{}` directly.

## Issue 2: `ODESize::_XtUPidxs` (FIXED — PR #25)

**Affected stdlib:** libc++ (LLVM/Clang)
**Status:** Fixed — replaced `std::map` with `FlatMap<std::string, Eigen::VectorXi>` (PR #25)

`ODESize<XV,UV,PV>` (`src/OptimalControl/ODESizes.h:93`) contains:
```cpp
std::map<std::string, Eigen::VectorXi> _XtUPidxs;
```

This map stores variable index groups (e.g., state/control/parameter indices by name). It is
used during ODE setup and phase construction but is **not accessed during solver evaluation**.
However, it is embedded in every ODE type, which is stored inside `LGLDefects<DODE, CS>`,
which gets type-erased into `ConstraintModel<T>` via `TypeStorage<ConstraintBase, 128>`.

### Why it crashes on libc++ but not libstdc++

The `std::map` sentinel node has self-referential pointers on **both** stdlibs. The difference
is size:

| Type | libstdc++ | libc++ |
|------|-----------|--------|
| `std::map<string, VectorXi>` | 48 bytes | 24 bytes |
| `std::string` | 32 bytes | 24 bytes |

The 32-byte size difference means `ConstraintModel<LGLDefects<BrachODE, 2>>` **exceeds** the
128-byte SBO on libstdc++ (→ heap path, safe) but **fits inside** on libc++ (→ memcpy path,
crash).

### Crash details

**Backtrace** (gdb, libc++ build):
```
#0  __memmove_avx_unaligned_erms ()                              — libc.so.6
#1  std::__1::__tree::__construct_node(...)                      — map insert
#2  std::__1::map::insert(...)                                   — map range insert
#3  ConstraintModel<LGLDefects<BrachODE, 2>>::clone_into(TypeStorage<ConstraintBase, 128>&)
#4  ConstraintFunction::thread_split(int)
#5  NonLinearProgram::analyzeThreading()
#6  NonLinearProgram::make_NLP()
#7  ODEPhaseBase::transcribe()
```

**ASAN report:** `attempting free on address which was not malloc()-ed: 0x7fff... Address is
located in stack of thread T0`

### Performance implications

This size difference also explains the **11.6x slowdown** in `BM_Phase_Construct_16seg` on
Linux (libstdc++) vs macOS (libc++) observed in PR #23 benchmarks:

| Benchmark | macOS (libc++) | Linux (libstdc++) |
|-----------|---------------|-------------------|
| `BM_Phase_Construct_16seg` | 54 us | 625 us |
| `BM_Phase_Construct_64seg` | 226 us | 808 us |

On macOS, `ConstraintModel` fits inline → fast memcpy clone. On Linux, it exceeds 128 bytes
→ heap allocation (`new`/`delete`) on every `clone_into` during phase construction.

**Note:** The macOS inline path is also memcpy-moving a `std::map`, which is technically
undefined behavior. It happens not to crash on Apple's libc++ but is not guaranteed to be safe.

### Potential fixes

1. **Factor `_XtUPidxs` out of `ODESize`** — move the index map to a side table not carried
   through type erasure. The map is only used during setup, not evaluation.

2. **Increase SBO_CAP** — e.g., to 256 bytes so all models fit inline on both stdlibs. But
   this only works if the memcpy-move is actually safe for all embedded types (it isn't for
   `std::map`).

3. **Replace memcpy-move with virtual dispatch** — add a `move_into(TypeStorage&)` virtual
   method. Correct by construction but adds overhead per move.

4. **Use `SBO_CAP = 0` for `ConstraintBase`/`ObjectiveBase`** — forces heap allocation,
   avoiding memcpy entirely. Simple but gives up the SBO performance benefit.

5. **Replace `std::map` with a flat, trivially-relocatable container** — e.g.,
   `std::vector<std::pair<...>>` or a custom sorted vector. Preserves SBO performance.

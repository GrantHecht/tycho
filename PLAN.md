# Bump Stack Allocator: Full Rewrite

## Summary

Full rewrite of the bump stack allocator (`MemoryManagement.h`) with:
- Save/restore semantics (replaces broken allocate/deallocate)
- SIMD-aligned allocations via `EIGEN_MAX_ALIGN_BYTES`
- `std::memset` zeroing (replaces Eigen expression templates)
- `Eigen::Map<T, AlignedMax>` for aligned SIMD loads/stores
- High-water learning (overflow triggers arena growth on full restore)
- `std::vector` overflow (replaces `std::list`)
- ISA-aware `DefaultSuperScalar` (SSE/NEON=2, AVX=4, AVX-512=8)
- Dead code removal (~170 lines of JumpTable paths, `critical_size` param)
- `using MemoryManager = BumpAllocator` alias for compatibility

## Files Changed

| File | Change |
|------|--------|
| `src/Utils/MemoryManagement.h` | Full rewrite (~250 lines replaces ~446) |
| `src/Utils/MemoryManagement.cpp` | Updated thread-local definitions |
| `src/TypeDefs/EigenTypes.h` | ISA-aware DefaultSuperScalar |
| `src/Bindings/MemoryManagerBind.h` | Updated for BumpAllocator |
| `src/Bindings/MemoryManagerBind.cpp` | Removed enable/disable, added resize |
| `src/Bindings/UtilsBind.cpp` | Updated TychoBind reference |
| `src/VectorFunctions/ComputableBase.h` | Removed TYCHO_MEMORYMAN guard |
| `src/VectorFunctions/CommonFunctions/CrossProduct.h` | Removed TYCHO_MEMORYMAN guards + fallbacks |
| `src/VectorFunctions/CommonFunctions/*.h` (18 files) | Removed `critical_size` arg |
| `src/OptimalControl/LGLDefects.h` | Removed `critical_size` arg |
| `src/OptimalControl/TrapezoidalDefects.h` | Removed `critical_size` arg |
| `src/Integrators/RKSteppers.h` | Removed `critical_size` arg |
| `src/Integrators/Integrator.h` | Removed `critical_size` arg |
| `CMakeLists.txt` | Removed TYCHO_MEMORYMAN definition |
| `bench/cpp/utils/bench_utils.cpp` | Added 6 allocate_run microbenchmarks |
| `tests/cpp/utils/test_memory_management.cpp` | Rewrote tests for BumpAllocator |

#pragma once

// Portable always-inline attribute. Use on short, frequently-inlined hot-path
// wrappers where the compiler's default inlining heuristic can miss.
// GCC/Clang: [[gnu::always_inline]] (C++11 attribute, clang-cl supports it).
// MSVC:     __forceinline (pre-C++11 keyword; ignores the attribute syntax).
// Other:    plain inline fallback.
#if defined(__GNUC__) || defined(__clang__)
#define TYCHO_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define TYCHO_ALWAYS_INLINE __forceinline
#else
#define TYCHO_ALWAYS_INLINE inline
#endif

// Portable noinline attribute. Use to keep a function-call boundary stable
// (e.g. so Enzyme's activity analysis doesn't have to walk through Eigen's
// expression templates inside the callee).
// GCC/Clang: __attribute__((noinline)) (placement-permissive; can sit between
//            `inline` and the return type without binding to the type).
// MSVC:     __declspec(noinline).
// Other:    no-op fallback.
#if defined(__GNUC__) || defined(__clang__)
#define TYCHO_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define TYCHO_NOINLINE __declspec(noinline)
#else
#define TYCHO_NOINLINE
#endif
